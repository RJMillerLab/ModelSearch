#!/usr/bin/env python3
"""Batch API wrapper for query2nugget mapping.

Core prompt construction, response normalization, and qrels/run builders live in
``query2nugget_mapping.py``. This module only owns OpenAI Batch-specific I/O.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from src.config import OUTPUT_DIR
from src.evaluate.query2nugget_mapping import (
    CARD2NUGGET_DIR,
    OUTPUT_HEADER_KEYWORD_JSON,
    OUTPUT_MATCH_DEBUG_JSON,
    OUTPUT_QRELS,
    OUTPUT_RUN,
    assemble_query2nugget_results,
    build_skip_no_api_key_rows,
    prepare_query2nugget_prompts,
)
from src.evaluate.query2nugget_match import save_qrels_and_run
from src.evaluate.query2nugget_prompting import TEXT_MODEL
from src.llm.batch import main_batch_query

BATCH_DIR = Path(OUTPUT_DIR) / "evaluate" / "batch"


def extract_text_from_batch_line(item: dict[str, Any]) -> tuple[str, str]:
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


def query2nugget_batch_responses(prompts: list[str], *, model: str | None) -> tuple[dict[int, tuple[str, str]], Path, Path]:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    in_path = BATCH_DIR / f"query2nugget_batch_input_{ts}.jsonl"
    out_path = BATCH_DIR / f"query2nugget_batch_output_{ts}.jsonl"
    model_name = model or TEXT_MODEL
    with open(in_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            payload = {
                "custom_id": f"{i:06d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 800,
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
                obj = json.loads(line)
                cid = str(obj.get("custom_id", "")).strip()
                if not cid.isdigit():
                    continue
                by_idx[int(cid)] = extract_text_from_batch_line(obj)
    return by_idx, in_path, out_path


def map_queries_batch(
    queries: list[str],
    *,
    model: str | None = None,
) -> list[dict[str, Any]]:
    clean = [(q or "").strip() for q in queries if (q or "").strip()]
    if not clean:
        return []
    if not os.getenv("OPENAI_API_KEY"):
        return build_skip_no_api_key_rows(clean)

    clean, prompt_by_i, error_by_i, ok_order, ok_prompts = prepare_query2nugget_prompts(clean)
    if not ok_prompts:
        return [error_by_i[i] for i in sorted(error_by_i)]

    print("[query2nugget] llm_mode=batch (OpenAI Batch API)")
    response_by_sub, in_path, out_path = query2nugget_batch_responses(ok_prompts, model=model)
    return assemble_query2nugget_results(
        clean,
        prompt_by_i,
        error_by_i,
        response_by_sub,
        ok_order,
        source_label="batch",
        batch_paths=(in_path, out_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map user queries to nugget schema headers with the OpenAI Batch API.",
    )
    parser.add_argument("--query", default=None, help="Single user search query.")
    parser.add_argument("--queries-file", default=None, help="Text file, one query per line.")
    parser.add_argument(
        "--output",
        default=OUTPUT_HEADER_KEYWORD_JSON,
        help=f"JSON output path (default: {OUTPUT_HEADER_KEYWORD_JSON})",
    )
    parser.add_argument("--model", default=None, help="OpenAI chat model (default from env or gpt-5.4-mini).")
    parser.add_argument(
        "--build-qrels-run",
        action="store_true",
        help="Also build qrels/run by matching mapping JSON against card2nugget CSV files.",
    )
    parser.add_argument(
        "--csv-root",
        action="append",
        default=None,
        help="Directory to scan for card2nugget CSV files (repeatable). Default: data_*/card2nugget/.",
    )
    parser.add_argument("--qrels-output", default=OUTPUT_QRELS, help=f"Qrels output path (default: {OUTPUT_QRELS})")
    parser.add_argument("--run-output", default=OUTPUT_RUN, help=f"Run output path (default: {OUTPUT_RUN})")
    parser.add_argument("--match-debug-output", default=OUTPUT_MATCH_DEBUG_JSON, help=f"Debug JSON for CSV matching (default: {OUTPUT_MATCH_DEBUG_JSON})")
    parser.add_argument("--subtopic", default="1", help="Subtopic id written into qrels (default: 1).")
    parser.add_argument(
        "--match-build",
        choices=["structured", "llm_rerank"],
        default="structured",
        help="How to build qrels/run: structured=header-nonempty row match; llm_rerank=LLM picks row_idx from candidate rows.",
    )
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

    results = map_queries_batch(queries, model=args.model)
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
        roots = [Path(p) for p in args.csv_root] if args.csv_root else [CARD2NUGGET_DIR]
        qrels_lines, run_lines = save_qrels_and_run(
            mapping_results=results,
            csv_roots=roots,
            qrels_path=Path(args.qrels_output),
            run_path=Path(args.run_output),
            debug_path=Path(args.match_debug_output),
            subtopic=str(args.subtopic),
            match_build=args.match_build,
            rerank_model=args.model,
        )
        print(f"saved_qrels = {Path(args.qrels_output).resolve()}  | lines = {qrels_lines}")
        print(f"saved_run = {Path(args.run_output).resolve()}  | lines = {run_lines}")
        print(f"saved_match_debug = {Path(args.match_debug_output).resolve()}")


if __name__ == "__main__":
    main()
