#!/usr/bin/env python3
"""Map a user query to nugget schema headers via LLM (keywords per relevant column)."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
load_dotenv(os.path.join(_repo_root, ".env"), override=False)

from src.config import OUTPUT_DIR
from src.llm.batch import main_batch_query
from src.evaluate.query_csv_to_qrels_run import build_qrels_and_run, discover_csv_paths

# Keep in sync with src/evaluate/card2nugget_extraction.py OUTPUT_HEADERS
NUGGET_SCHEMA_HEADERS = [
    "Model",
    "Base_model",
    "Dataset",
    "Train_dataset",
    "Test_dataset",
    "Model_hyperparameters",
    "Model_variant_type",
    "Metric",
    "Metric_value",
]

OUTPUT_HEADER_KEYWORD_JSON = os.path.join(OUTPUT_DIR, "evaluate", "query_header_keyword_mapping.json")
OUTPUT_QRELS = os.path.join(OUTPUT_DIR, "evaluate", "real_subtopic.qrels")
OUTPUT_RUN = os.path.join(OUTPUT_DIR, "evaluate", "real_initial.run")
OUTPUT_MATCH_DEBUG_JSON = os.path.join(OUTPUT_DIR, "evaluate", "query_csv_match_debug.json")
BATCH_DIR = Path(OUTPUT_DIR) / "evaluate" / "batch"
EVAL_DIR = Path(OUTPUT_DIR) / "evaluate"

PROMPT_PATH = Path("src/evaluate/query2nugget_prompts.yaml")
PROMPT_KEY = "query_to_nugget_headers"
TEXT_MODEL = os.getenv("MODELSEARCHDEMO_TEXT_EXTRACTION_MODEL", "gpt-4o-mini")


def _load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        return ""
    data = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8")) or {}
    return str(data.get(PROMPT_KEY, "")).strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                pass
    return None


def _build_prompt_for_query(query: str) -> str:
    template = _load_prompt_template()
    if not template:
        raise RuntimeError(f"Missing prompt key {PROMPT_KEY} in {PROMPT_PATH}")
    headers_yaml = yaml.safe_dump(NUGGET_SCHEMA_HEADERS, allow_unicode=True).strip()
    return template.replace("[[HEADERS_YAML]]", headers_yaml).replace("[[USER_QUERY]]", (query or "").strip())


def _finalize_map_response(query: str, prompt: str, raw: str) -> dict[str, Any]:
    parsed = _extract_json_object(raw)
    if not parsed:
        return {
            "query": query,
            "related": [],
            "header_list": [],
            "error": "parse_failed",
            "prompt": prompt,
            "raw_response": raw,
        }
    out = _validate_and_normalize(parsed, query)
    out["prompt"] = prompt
    out["raw_response"] = raw
    return out


def _validate_and_normalize(
    parsed: dict[str, Any],
    user_query: str,
) -> dict[str, Any]:
    allowed = set(NUGGET_SCHEMA_HEADERS)
    related_raw = parsed.get("related")
    if not isinstance(related_raw, list):
        related_raw = []

    normalized: list[dict[str, Any]] = []
    for item in related_raw:
        if not isinstance(item, dict):
            continue
        h = str(item.get("header", "")).strip()
        if h not in allowed:
            continue
        kws = item.get("keywords")
        if isinstance(kws, str):
            kws = [kws]
        elif not isinstance(kws, list):
            kws = []
        keywords = [str(x).strip() for x in kws if str(x).strip()]
        normalized.append({"header": h, "keywords": keywords})

    q = str(parsed.get("query", "")).strip() or user_query
    return {
        "query": q,
        "related": normalized,
        "header_list": [x["header"] for x in normalized],
    }


def _extract_text_from_batch_line(item: dict[str, Any]) -> tuple[str, str]:
    resp = item.get("response") or {}
    body = resp.get("body") if isinstance(resp, dict) else {}
    if isinstance(body, dict):
        choices = body.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            content = message.get("content", "")
            if isinstance(content, str):
                return content, ""
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(str(c.get("text", "")))
                return "\n".join(parts), ""
    err = item.get("error")
    return "", (str(err).strip() if err else "") or "batch_output_parse_error"


def map_queries_via_batch(queries: list[str], *, model: str | None = None) -> list[dict[str, Any]]:
    """OpenAI Batch API: one chat completion per query. Writes input/output jsonl under evaluate/batch/."""
    clean = [(q or "").strip() for q in queries if (q or "").strip()]
    if not clean:
        return []

    if not os.getenv("OPENAI_API_KEY"):
        out: list[dict[str, Any]] = []
        for q in clean:
            try:
                prompt = _build_prompt_for_query(q)
            except RuntimeError as e:
                prompt = ""
                out.append(
                    {
                        "query": q,
                        "related": [],
                        "header_list": [],
                        "error": str(e),
                        "prompt": prompt,
                        "raw_response": "",
                    }
                )
                continue
            out.append(
                {
                    "query": q,
                    "related": [],
                    "header_list": [],
                    "error": "skip_no_api_key",
                    "prompt": prompt,
                    "raw_response": "",
                }
            )
        return out

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    in_path = BATCH_DIR / f"query2nugget_batch_input_{ts}.jsonl"
    out_path = BATCH_DIR / f"query2nugget_batch_output_{ts}.jsonl"

    prompts: list[str] = []
    with open(in_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(clean):
            prompt = _build_prompt_for_query(q)
            prompts.append(prompt)
            payload = {
                "custom_id": f"{i:06d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model or TEXT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 800,
                    "temperature": 0.0,
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    main_batch_query(str(in_path), str(out_path))

    by_idx: dict[int, tuple[str, str]] = {}
    if out_path.is_file():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = str(obj.get("custom_id", "")).strip()
                if not cid.isdigit():
                    continue
                idx = int(cid)
                text, note = _extract_text_from_batch_line(obj)
                by_idx[idx] = (text, note)

    results: list[dict[str, Any]] = []
    for i, q in enumerate(clean):
        prompt = prompts[i]
        if i not in by_idx:
            results.append(
                {
                    "query": q,
                    "related": [],
                    "header_list": [],
                    "error": "batch_missing_line",
                    "prompt": prompt,
                    "raw_response": "",
                }
            )
            continue
        raw, err_note = by_idx[i]
        if err_note:
            results.append(
                {
                    "query": q,
                    "related": [],
                    "header_list": [],
                    "error": err_note,
                    "prompt": prompt,
                    "raw_response": raw,
                }
            )
            continue
        if not (raw or "").strip():
            results.append(
                {
                    "query": q,
                    "related": [],
                    "header_list": [],
                    "error": "batch_empty_response",
                    "prompt": prompt,
                    "raw_response": raw,
                }
            )
            continue
        row = _finalize_map_response(q, prompt, raw)
        row["batch_input_jsonl"] = str(in_path.resolve())
        row["batch_output_jsonl"] = str(out_path.resolve())
        results.append(row)
    return results


def _save_qrels_and_run(
    *,
    mapping_results: list[dict[str, Any]],
    csv_roots: list[Path],
    qrels_path: Path,
    run_path: Path,
    debug_path: Path,
    subtopic: str,
) -> tuple[int, int]:
    csv_paths = discover_csv_paths(csv_roots)
    if not csv_paths:
        raise RuntimeError(f"No CSV files found under: {[str(x) for x in csv_roots]}")
    qrels_rows, run_rows, debug = build_qrels_and_run(mapping_results, csv_paths, subtopic=subtopic)

    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    with open(qrels_path, "w", encoding="utf-8") as f:
        for qid, st, doc_id, rel in qrels_rows:
            f.write(f"{qid} {st} {doc_id} {rel}\n")
    with open(run_path, "w", encoding="utf-8") as f:
        for qid, q0, doc_id, rank, score, tag in run_rows:
            f.write(f"{qid} {q0} {doc_id} {rank} {score} {tag}\n")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump({"csv_paths": [str(p) for p in csv_paths], "queries": debug}, f, ensure_ascii=False, indent=2)

    return len(qrels_rows), len(run_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map a user query to nugget schema headers (LLM) with per-header keyword lists.",
    )
    parser.add_argument("--query", default=None, help="Single user search query.")
    parser.add_argument("--queries-file", default=None, help="Text file, one query per line.")
    parser.add_argument(
        "--output",
        default=OUTPUT_HEADER_KEYWORD_JSON,
        help=f"JSON output path (default: {OUTPUT_HEADER_KEYWORD_JSON})",
    )
    parser.add_argument("--model", default=None, help="OpenAI chat model (default from env or gpt-4o-mini).")
    parser.add_argument(
        "--build-qrels-run",
        action="store_true",
        help="Also build qrels/run by matching mapping JSON against card2nugget CSV files.",
    )
    parser.add_argument(
        "--csv-root",
        action="append",
        default=None,
        help="Directory to scan for card2nugget CSV files (repeatable). Default: evaluate/ and evaluate/batch/.",
    )
    parser.add_argument("--qrels-output", default=OUTPUT_QRELS, help=f"Qrels output path (default: {OUTPUT_QRELS})")
    parser.add_argument("--run-output", default=OUTPUT_RUN, help=f"Run output path (default: {OUTPUT_RUN})")
    parser.add_argument(
        "--match-debug-output",
        default=OUTPUT_MATCH_DEBUG_JSON,
        help=f"Debug JSON for CSV matching (default: {OUTPUT_MATCH_DEBUG_JSON})",
    )
    parser.add_argument("--subtopic", default="1", help="Subtopic id written into qrels (default: 1).")
    args = parser.parse_args()

    queries: list[str] = []
    if args.query:
        queries.append(args.query.strip())
    if args.queries_file:
        p = Path(args.queries_file)
        if not p.is_file():
            parser.error(f"--queries-file not found: {p}")
        with open(p, encoding="utf-8") as f:
            queries.extend(line.strip() for line in f if line.strip())
    queries = [q for q in queries if q]
    if not queries:
        parser.error("Provide --query and/or --queries-file.")

    print("[mode] OpenAI Batch (always)")
    results = map_queries_via_batch(queries, model=args.model)

    payload: dict[str, Any] = {"queries": results} if len(results) > 1 else results[0]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"saved = {out_path.resolve()}")
    for r in results:
        err = r.get("error", "")
        hl = r.get("header_list", [])
        print(f"query = {r.get('query', '')!r}  |  headers = {hl}  |  note = {err or 'ok'}")

    if args.build_qrels_run:
        roots = [Path(p) for p in args.csv_root] if args.csv_root else [EVAL_DIR, BATCH_DIR]
        qrels_lines, run_lines = _save_qrels_and_run(
            mapping_results=results,
            csv_roots=roots,
            qrels_path=Path(args.qrels_output),
            run_path=Path(args.run_output),
            debug_path=Path(args.match_debug_output),
            subtopic=str(args.subtopic),
        )
        print(f"saved_qrels = {Path(args.qrels_output).resolve()}  | lines = {qrels_lines}")
        print(f"saved_run = {Path(args.run_output).resolve()}  | lines = {run_lines}")
        print(f"saved_match_debug = {Path(args.match_debug_output).resolve()}")


if __name__ == "__main__":
    main()
