#!/usr/bin/env python3
from __future__ import annotations

"""
Build OpenAI Batch input for labeling scientific search queries.

The script supports chunking many queries into a single request, so you can
send up to 500 queries at once and ask the model to return one label per query.

Workflow:
  1) Build batch input:
     python -m src.query.batch_query_label build \
       --input data_251117/query/query_rewrite_batch_output.jsonl \
       --output data_251117/query/query_label_batch_input.jsonl \
       --chunk-size 500

  2) Submit with existing batch runner:
     python -m src.llm.batch \
       data_251117/query/query_label_batch_input.jsonl \
       data_251117/query/query_label_batch_output.jsonl

  3) Parse batch output:
     python -m src.query.batch_query_label parse \
       --input data_251117/query/query_label_batch_output.jsonl \
       --output data_251117/query/query_label_polished.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


SIX_CLASS_LABELS = ["evidence-based", "comparison", "experience", "reason", "instruction", "debate"]
FOUR_CLASS_LABELS = ["factual_retrieval", "comparison", "summarization", "causal_reasoning"]


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
- Keep labels exactly as written above, with lowercase and hyphenation preserved.
- Return JSON only, with this exact schema:
  {"labels":[{"custom_id":"...","label":"..."}]}
- Preserve input order in the output labels list.
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
- Keep labels exactly as written above, with lowercase and underscores preserved.
- Return JSON only, with this exact schema:
  {"labels":[{"custom_id":"...","label":"..."}]}
- Preserve input order in the output labels list.
"""


USER_TEMPLATE = """Label the following scientific search queries.

Queries:
{queries_json}
"""


def load_queries(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _extract_query_from_response(rec: dict[str, Any]) -> str:
    response = rec.get("response") or {}
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


def _extract_query_text(rec: dict[str, Any], query_field: str) -> str:
    value = rec.get(query_field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    for fallback_key in ("query", "rewritten_query", "before_query", "after_query"):
        value = rec.get(fallback_key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    response_query = _extract_query_from_response(rec)
    if response_query:
        return response_query
    return ""


def chunk_records(records: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]


def build_prompt_payload(chunk: list[dict[str, Any]], query_field: str, chunk_idx: int) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for idx, rec in enumerate(chunk, start=1):
        query = _extract_query_text(rec, query_field)
        if not query:
            continue
        custom_id = str(rec.get("custom_id") or rec.get("id") or f"chunk-{chunk_idx:04d}-q{idx:03d}")
        payload.append({"custom_id": custom_id, "query": query})
    return payload


def build_batch_input(
    records: Iterable[dict[str, Any]],
    output_path: Path,
    model: str,
    chunk_size: int,
    label_scheme: str,
    query_field: str,
) -> None:
    records_list = list(records)
    chunks = chunk_records(records_list, chunk_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = SYSTEM_PROMPT_SIX if label_scheme == "six" else SYSTEM_PROMPT_FOUR

    with output_path.open("w", encoding="utf-8") as f:
        for chunk_idx, chunk in enumerate(chunks):
            payload = build_prompt_payload(chunk, query_field=query_field, chunk_idx=chunk_idx)
            if not payload:
                continue
            item = {
                "custom_id": f"label-{chunk_idx:04d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": USER_TEMPLATE.format(queries_json=json.dumps(payload, ensure_ascii=False, indent=2))},
                    ],
                    "temperature": 0.0,
                    "max_tokens": max(256, 12 * len(payload)),
                },
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _extract_json_text(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
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
            return {}
    return {}


def parse_labels_response(text: str) -> list[dict[str, str]]:
    obj = _extract_json_text(text)
    labels = obj.get("labels")
    if isinstance(labels, list):
        out: list[dict[str, str]] = []
        for item in labels:
            if not isinstance(item, dict):
                continue
            custom_id = str(item.get("custom_id") or "").strip()
            label = str(item.get("label") or "").strip()
            if custom_id and label:
                out.append({"custom_id": custom_id, "label": label})
        if out:
            return out

    # Fallback: allow a plain list of {"custom_id":..., "label":...}
    if isinstance(obj, list):
        out = []
        for item in obj:
            if not isinstance(item, dict):
                continue
            custom_id = str(item.get("custom_id") or "").strip()
            label = str(item.get("label") or "").strip()
            if custom_id and label:
                out.append({"custom_id": custom_id, "label": label})
        if out:
            return out

    return []


def extract_response_text(row: dict[str, Any]) -> str:
    response = row.get("response") or {}
    body = response.get("body") if isinstance(response, dict) else {}
    choices = body.get("choices") if isinstance(body, dict) else []
    if isinstance(choices, list) and choices:
        first = choices[0] or {}
        message = first.get("message") or {}
        return str(message.get("content") or "").strip()
    return ""


def _extract_prompt_payload(row: dict[str, Any]) -> list[dict[str, Any]]:
    body = row.get("body") or {}
    messages = body.get("messages") if isinstance(body, dict) else []
    if not isinstance(messages, list):
        return []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        text = str(msg.get("content") or "").strip()
        if not text:
            continue
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            continue
        try:
            payload = json.loads(text[start : end + 1])
        except Exception:
            continue
        if isinstance(payload, list):
            out: list[dict[str, Any]] = []
            for item in payload:
                if isinstance(item, dict):
                    out.append(item)
            return out
    return []


def _build_request_lookup(input_path: Path) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            custom_id = str(row.get("custom_id") or "").strip()
            if not custom_id:
                continue
            lookup[custom_id] = _extract_prompt_payload(row)
    return lookup


def parse_batch_output(input_path: Path, output_path: Path, batch_input_path: Path) -> None:
    request_lookup = _build_request_lookup(batch_input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            response_text = extract_response_text(row)
            labels = parse_labels_response(response_text)
            request_items = request_lookup.get(str(row.get("custom_id") or "").strip(), [])
            request_lookup_map = {
                str(item.get("custom_id") or "").strip(): str(item.get("query") or "").strip()
                for item in request_items
            }
            label_map = {
                str(item.get("custom_id") or "").strip(): str(item.get("label") or "").strip()
                for item in labels
                if str(item.get("custom_id") or "").strip()
            }

            for item in request_items:
                custom_id = str(item.get("custom_id") or "").strip()
                if not custom_id:
                    continue
                label = label_map.get(custom_id, "")
                if not label:
                    continue
                out = {
                    "chunk_custom_id": row.get("custom_id"),
                    "custom_id": custom_id,
                    "query": request_lookup_map.get(custom_id, ""),
                    "label": label,
                    "response_text": response_text,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or parse batch input for query labeling.")
    subparsers = parser.add_subparsers(dest="command")

    build = subparsers.add_parser("build", help="Build batch input JSONL")
    build.add_argument("--input", type=Path, default=Path("data_251117/query/query_rewrite_batch_output.jsonl"))
    build.add_argument("--output", type=Path, default=Path("data_251117/query/query_label_batch_input.jsonl"))
    build.add_argument("--model", default="gpt-4o-mini")
    build.add_argument("--chunk-size", type=int, default=500)
    build.add_argument("--scheme", choices=["six", "four"], default="six")
    build.add_argument("--query-field", default="query")

    parse = subparsers.add_parser("parse", help="Parse OpenAI batch output JSONL")
    parse.add_argument("--input", type=Path, required=True)
    parse.add_argument("--batch-input", type=Path, default=Path("data_251117/query/query_label_batch_input.jsonl"))
    parse.add_argument("--output", type=Path, default=Path("data_251117/query/query_label_polished.jsonl"))

    args = parser.parse_args()

    if args.command == "parse":
        parse_batch_output(args.input, args.output, batch_input_path=args.batch_input)
        print(f"saved_output={args.output}")
        return

    input_path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    records = load_queries(input_path)
    if not records:
        raise SystemExit("No query records found.")

    build_batch_input(
        records=records,
        output_path=args.output,
        model=args.model,
        chunk_size=args.chunk_size,
        label_scheme=args.scheme,
        query_field=args.query_field,
    )
    print(f"saved_batch_input={args.output}")
    print(f"rows={len(records)}")
    print(f"chunk_size={args.chunk_size}")
    print(f"scheme={args.scheme}")


if __name__ == "__main__":
    main()
