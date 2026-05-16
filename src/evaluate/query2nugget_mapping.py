#!/usr/bin/env python3
"""Sync query2nugget mapping entrypoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
load_dotenv(os.path.join(_repo_root, ".env"), override=False)

from src.config import OUTPUT_DIR
from src.evaluate.query2nugget_prompting import (
    TEXT_MODEL,
    build_prompt_for_query,
    finalize_map_response,
)
from src.llm.model import query_openai, setup_openai

OUTPUT_HEADER_KEYWORD_JSON = os.path.join(OUTPUT_DIR, "evaluate", "query_header_keyword_mapping.json")
OUTPUT_QRELS = os.path.join(OUTPUT_DIR, "evaluate", "real_subtopic.qrels")
OUTPUT_RUN = os.path.join(OUTPUT_DIR, "evaluate", "real_initial.run")
OUTPUT_MATCH_DEBUG_JSON = os.path.join(OUTPUT_DIR, "evaluate", "query_csv_match_debug.json")
CARD2NUGGET_DIR = Path(OUTPUT_DIR) / "card2nugget"


def build_skip_no_api_key_rows(clean: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for q in clean:
        try:
            prompt = build_prompt_for_query(q)
        except RuntimeError as e:
            out.append({"query": q, "related": [], "header_list": [], "error": str(e), "prompt": "", "raw_response": ""})
            continue
        out.append({"query": q, "related": [], "header_list": [], "error": "skip_no_api_key", "prompt": prompt, "raw_response": ""})
    return out


def query2nugget_iter_responses(prompts: list[str], *, model: str | None) -> dict[int, tuple[str, str]]:
    by_idx: dict[int, tuple[str, str]] = {}
    setup_openai("", mode="openai")
    for i, prompt in enumerate(prompts):
        try:
            raw = query_openai(prompt, mode="openai", model=model or TEXT_MODEL, max_tokens=800) or ""
            by_idx[i] = (raw, "")
        except Exception as e:
            by_idx[i] = ("", str(e))
    return by_idx


def prepare_query2nugget_prompts(
    queries: list[str],
) -> tuple[list[str], dict[int, str], dict[int, dict[str, Any]], list[int], list[str]]:
    clean = [(q or "").strip() for q in queries if (q or "").strip()]
    error_by_i: dict[int, dict[str, Any]] = {}
    prompt_by_i: dict[int, str] = {}
    ok_order: list[int] = []
    ok_prompts: list[str] = []
    for i, q in enumerate(clean):
        try:
            prompt = build_prompt_for_query(q)
            prompt_by_i[i] = prompt
            ok_order.append(i)
            ok_prompts.append(prompt)
        except RuntimeError as e:
            error_by_i[i] = {"query": q, "related": [], "header_list": [], "error": str(e), "prompt": "", "raw_response": ""}
    return clean, prompt_by_i, error_by_i, ok_order, ok_prompts


def assemble_query2nugget_results(
    clean: list[str],
    prompt_by_i: dict[int, str],
    error_by_i: dict[int, dict[str, Any]],
    response_by_sub: dict[int, tuple[str, str]],
    ok_order: list[int],
    *,
    source_label: str,
    batch_paths: tuple[Path, Path] | None = None,
) -> list[dict[str, Any]]:
    by_orig: dict[int, tuple[str, str]] = {
        orig_i: response_by_sub[k]
        for k, orig_i in enumerate(ok_order)
        if k in response_by_sub
    }
    missing_key = f"{source_label}_missing_line"
    results: list[dict[str, Any]] = []
    for i, q in enumerate(clean):
        if i in error_by_i:
            results.append(error_by_i[i])
            continue
        prompt = prompt_by_i[i]
        if i not in by_orig:
            results.append({"query": q, "related": [], "header_list": [], "error": missing_key, "prompt": prompt, "raw_response": ""})
            continue
        raw, err_note = by_orig[i]
        if err_note:
            results.append({"query": q, "related": [], "header_list": [], "error": err_note, "prompt": prompt, "raw_response": raw})
            continue
        if not (raw or "").strip():
            results.append({"query": q, "related": [], "header_list": [], "error": f"{source_label}_empty_response", "prompt": prompt, "raw_response": raw})
            continue
        row = finalize_map_response(q, prompt, raw)
        if batch_paths:
            row["batch_input_jsonl"] = str(batch_paths[0].resolve())
            row["batch_output_jsonl"] = str(batch_paths[1].resolve())
        results.append(row)
    return results


def map_queries(
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

    print("[query2nugget] llm_mode=iter (sync chat per query)")
    response_by_sub = query2nugget_iter_responses(ok_prompts, model=model)
    return assemble_query2nugget_results(
        clean,
        prompt_by_i,
        error_by_i,
        response_by_sub,
        ok_order,
        source_label="iter",
    )


def _load_queries(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
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
    return queries


def _write_mapping_output(results: list[dict[str, Any]], output: str) -> None:
    payload: dict[str, Any] = {"queries": results} if len(results) > 1 else results[0]
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"saved = {out_path.resolve()}")
    for r in results:
        err = r.get("error", "")
        headers = r.get("header_list", [])
        print(f"query = {r.get('query', '')!r}  |  headers = {headers}  |  note = {err or 'ok'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Map user queries to nugget headers and value constraints.")
    parser.add_argument("--query", default=None, help="Single user search query.")
    parser.add_argument("--queries-file", default=None, help="Text file, one query per line.")
    parser.add_argument("--output", default=OUTPUT_HEADER_KEYWORD_JSON, help=f"JSON output path (default: {OUTPUT_HEADER_KEYWORD_JSON})")
    parser.add_argument("--model", default=None, help="OpenAI chat model (default from env or gpt-5.4-mini).")
    parser.add_argument("--build-qrels-run", action="store_true", help="Also build qrels/run from card2nugget CSV files.")
    parser.add_argument("--csv-root", action="append", default=None, help="Directory to scan for card2nugget CSV files.")
    parser.add_argument("--qrels-output", default=OUTPUT_QRELS, help=f"Qrels output path (default: {OUTPUT_QRELS})")
    parser.add_argument("--run-output", default=OUTPUT_RUN, help=f"Run output path (default: {OUTPUT_RUN})")
    parser.add_argument("--match-debug-output", default=OUTPUT_MATCH_DEBUG_JSON, help=f"Debug JSON output path (default: {OUTPUT_MATCH_DEBUG_JSON})")
    parser.add_argument("--subtopic", default="1", help="Subtopic id written into qrels.")
    parser.add_argument("--match-build", choices=["structured", "llm_rerank"], default="structured", help="qrels/run builder.")
    args = parser.parse_args()

    results = map_queries(_load_queries(args, parser), model=args.model)
    _write_mapping_output(results, args.output)

    if args.build_qrels_run:
        from src.evaluate.query2nugget_match import save_qrels_and_run

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
