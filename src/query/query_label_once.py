#!/usr/bin/env python3
from __future__ import annotations

"""
Label scientific search queries with one prompt per chunk.

This script is intentionally not batch API based. It groups queries into
chunks, sends one prompt per chunk, and asks the model to return one label
plus a short reason for each query.

Default behavior:
  - chunk size: 100 queries
  - scheme: six labels
  - output: one JSONL row per query with id/label/reason

Example:
  python -m src.query.query_label_once \
    --input data_251117/query/query_rewrite_batch_output.jsonl \
    --output data_251117/query/query_label_once.jsonl \
    --scheme six
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


SIX_LABELS = ["evidence-based", "comparison", "experience", "reason", "instruction", "debate"]
FOUR_LABELS = ["factual_retrieval", "comparison", "summarization", "causal_reasoning"]


SYSTEM_PROMPT_SIX = """You are a strict query labeler for scientific search queries.

Label each query with exactly one category from this set:
- evidence-based
- comparison
- experience
- reason
- instruction
- debate

Definitions:
- evidence-based: asks for facts, models, methods, datasets, metrics, definitions, parameters, setups, formulas, examples, or direct retrieval of scientific evidence.
- comparison: asks which option is better, how two things differ, trade-offs, contrasts, performance differences, or ranking between alternatives.
- experience: asks about human behavior, user behavior, annotator behavior, reader behavior, observations, experiences, or field-study style evidence.
- reason: asks why, motivation, cause, influence, effect, implications, explanation of an outcome, or the reason something happened.
- instruction: asks how to do something, operate, assemble, troubleshoot, calibrate, implement, or follow a procedure.
- debate: asks normative, controversial, argumentative, or stance-seeking questions, including whether something should be done or whether a claim is valid/unsafe/transparent.

Rules:
- Choose the label that matches the primary intent.
- If a query can fit multiple labels, prefer the most specific one.
- If the query is essentially "find a model/method/paper about X", use evidence-based.
- Return JSON only with this exact schema:
  {"labels":[{"custom_id":"...","label":"...","reason":"..."}]}
- Keep `reason` short, ideally 5-15 words, and explain the main cue for the label.
- Preserve input order in the output labels list.
- Keep labels exactly as written above, with lowercase and hyphenation preserved.
"""


SYSTEM_PROMPT_FOUR = """You are a strict query labeler for scientific search queries.

Label each query with exactly one category from this set:
- factual_retrieval
- comparison
- summarization
- causal_reasoning

Definitions:
- factual_retrieval: asks for facts, models, methods, datasets, metrics, definitions, parameters, setups, formulas, examples, or direct retrieval of scientific evidence.
- comparison: asks which option is better, how two things differ, trade-offs, contrasts, performance differences, or ranking between alternatives.
- summarization: asks to summarize, condense, list main points, extract key insights, or synthesize information from a document/study/result set.
- causal_reasoning: asks why, motivation, cause, influence, effect, implications, explanation of an outcome, or the reason something happened.

Rules:
- Choose the label that matches the primary intent.
- If a query can fit multiple labels, prefer the most specific one.
- If the query is essentially "find a model/method/paper about X", use factual_retrieval.
- Return JSON only with this exact schema:
  {"labels":[{"custom_id":"...","label":"...","reason":"..."}]}
- Keep `reason` short, ideally 5-15 words, and explain the main cue for the label.
- Preserve input order in the output labels list.
- Keep labels exactly as written above, with lowercase and underscores preserved.
"""


def load_queries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_query_from_response(row: dict[str, Any]) -> str:
    response = row.get("response") or {}
    body = response.get("body") if isinstance(response, dict) else {}
    choices = body.get("choices") if isinstance(body, dict) else []
    if isinstance(choices, list) and choices:
        first = choices[0] or {}
        message = first.get("message") or {}
        content = str(message.get("content") or "").strip()
        if content:
            try:
                obj = json.loads(content)
                if isinstance(obj, dict):
                    query = obj.get("query")
                    if isinstance(query, str) and query.strip():
                        return query.strip()
            except Exception:
                pass
            return content.splitlines()[0].strip()
    return ""


def _extract_query_text(row: dict[str, Any], query_field: str = "query") -> str:
    value = row.get(query_field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    for fallback_key in ("query", "rewritten_query", "before_query", "after_query"):
        value = row.get(fallback_key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    response_query = _extract_query_from_response(row)
    if response_query:
        return response_query
    return ""


def _make_custom_id(row: dict[str, Any], index: int) -> str:
    custom_id = str(row.get("custom_id") or row.get("id") or "").strip()
    return custom_id or f"q{index:04d}"


def _slice_records(records: list[dict[str, Any]], start: int) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError("start must be >= 0")
    return records[start:]


def _build_payload(records: list[dict[str, Any]], query_field: str = "query") -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for idx, row in enumerate(records):
        query = _extract_query_text(row, query_field=query_field)
        if not query:
            continue
        payload.append(
            {
                "custom_id": _make_custom_id(row, idx),
                "query": query,
            }
        )
    return payload


def _chunked(items: list[dict[str, str]], chunk_size: int) -> list[list[dict[str, str]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _build_prompt(payload: list[dict[str, str]], scheme: str) -> tuple[str, str]:
    system_prompt = SYSTEM_PROMPT_SIX if scheme == "six" else SYSTEM_PROMPT_FOUR
    user_prompt = (
        "Label the following scientific search queries.\n\n"
        "Queries:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )
    return system_prompt, user_prompt


def _extract_json_text(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            return None
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            return None
    return None


def _parse_labels(text: str) -> list[dict[str, str]]:
    obj = _extract_json_text(text)
    if isinstance(obj, dict):
        labels = obj.get("labels")
    else:
        labels = obj
    out: list[dict[str, str]] = []
    if isinstance(labels, list):
        for item in labels:
            if not isinstance(item, dict):
                continue
            custom_id = str(item.get("custom_id") or "").strip()
            label = str(item.get("label") or "").strip()
            reason = str(item.get("reason") or "").strip()
            if custom_id and label:
                out.append({"custom_id": custom_id, "label": label, "reason": reason})
    return out


def _build_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def _write_prompt(path: Path, system_prompt: str, user_prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("### SYSTEM ###\n")
        f.write(system_prompt)
        f.write("\n\n### USER ###\n")
        f.write(user_prompt)


def _write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_output_rows(handle: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _run_spinner(stop_event: threading.Event, prefix: str = "Waiting") -> None:
    frames = ["|", "/", "-", "\\"]
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{prefix} {frames[idx % len(frames)]}")
        sys.stdout.flush()
        time.sleep(0.12)
        idx += 1
    sys.stdout.write("\r")
    sys.stdout.flush()


def _approx_token_count(text: str, model: str) -> int:
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Rough fallback if tiktoken is unavailable.
        return max(1, len(text) // 4)


def _estimate_chat_tokens(system_prompt: str, user_prompt: str, model: str) -> int:
    # Simple, practical approximation.
    return _approx_token_count(system_prompt, model) + _approx_token_count(user_prompt, model) + 12


def main() -> None:
    parser = argparse.ArgumentParser(description="Label queries with one prompt per chunk.")
    parser.add_argument("--input", type=Path, default=Path("data_251117/query/query_rewrite_batch_output.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data_251117/query/query_label_once.jsonl"))
    parser.add_argument("--prompt-out-dir", type=Path, default=None)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--scheme", choices=["six", "four"], default="six")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    rows = load_queries(args.input)
    selected = _slice_records(rows, args.start)
    payload = _build_payload(selected)
    if not payload:
        raise SystemExit("No query payload found.")

    chunks = _chunked(payload, args.chunk_size)
    print(f"total_queries={len(payload)}")
    print(f"chunk_size={args.chunk_size}")
    print(f"num_chunks={len(chunks)}")

    client = None if args.dry_run else _build_client()

    output_handle = None
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output.open("w", encoding="utf-8")

    try:
        for chunk_idx, chunk in enumerate(chunks, start=1):
            system_prompt, user_prompt = _build_prompt(chunk, args.scheme)
            est_tokens = _estimate_chat_tokens(system_prompt, user_prompt, args.model)
            print(f"chunk={chunk_idx}/{len(chunks)} queries={len(chunk)} estimated_input_tokens={est_tokens}")

            if args.prompt_out_dir is not None:
                args.prompt_out_dir.mkdir(parents=True, exist_ok=True)
                _write_prompt(args.prompt_out_dir / f"chunk_{chunk_idx:02d}.txt", system_prompt, user_prompt)

            if args.dry_run:
                continue

            stop_event = threading.Event()
            spinner = threading.Thread(target=_run_spinner, args=(stop_event, f"Waiting chunk {chunk_idx}"), daemon=True)
            spinner.start()
            t0 = time.time()
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                )
            finally:
                stop_event.set()
                spinner.join(timeout=1.0)

            elapsed = time.time() - t0
            response_text = str(response.choices[0].message.content or "").strip()
            label_rows = _parse_labels(response_text)
            label_map = {row["custom_id"]: row for row in label_rows}

            chunk_rows: list[dict[str, Any]] = []
            for item in chunk:
                custom_id = item["custom_id"]
                parsed = label_map.get(custom_id, {})
                chunk_rows.append(
                    {
                        "chunk_id": f"chunk-{chunk_idx:02d}",
                        "custom_id": custom_id,
                        "query": item["query"],
                        "label": parsed.get("label", ""),
                        "reason": parsed.get("reason", ""),
                        "estimated_input_tokens": est_tokens,
                        "response_text": response_text,
                        "elapsed_sec": round(elapsed, 2),
                    }
                )

            if output_handle is not None:
                _append_output_rows(output_handle, chunk_rows)
            print(f"chunk={chunk_idx} returned_labels={len(label_rows)} elapsed_sec={elapsed:.2f}")

        if output_handle is not None:
            print(f"saved_output={args.output}")
    finally:
        if output_handle is not None:
            output_handle.close()


if __name__ == "__main__":
    main()
