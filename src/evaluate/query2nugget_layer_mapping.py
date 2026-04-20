#!/usr/bin/env python3
"""Map a user query to nugget schema headers via LLM (keywords per relevant column)."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
load_dotenv(os.path.join(_repo_root, ".env"), override=False)

from src.config import OUTPUT_DIR
from src.llm.model import query_openai, setup_openai

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

INPUT_NUGGETS_JSONL = os.path.join(OUTPUT_DIR, "evaluate", "modelcard_nuggets.jsonl")
OUTPUT_MAPPING_JSON = os.path.join(OUTPUT_DIR, "evaluate", "query_nugget_mapping.json")
OUTPUT_HEADER_KEYWORD_JSON = os.path.join(OUTPUT_DIR, "evaluate", "query_header_keyword_mapping.json")
OUTPUT_QRELS = os.path.join(OUTPUT_DIR, "evaluate", "real_subtopic.qrels")
OUTPUT_RUN = os.path.join(OUTPUT_DIR, "evaluate", "real_initial.run")

PROMPT_PATH = Path("src/evaluate/query2nugget_prompts.yaml")
PROMPT_KEY = "query_to_nugget_headers"
TEXT_MODEL = os.getenv("MODELSEARCHDEMO_TEXT_EXTRACTION_MODEL", "gpt-4o-mini")


def load_modelcard_nuggets(path: str = INPUT_NUGGETS_JSONL) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


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


def map_query_to_nugget_headers(
    query: str,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Call LLM: query + fixed headers → JSON with related headers and keyword lists."""
    query = (query or "").strip()
    if not query:
        raise ValueError("query is empty")

    template = _load_prompt_template()
    if not template:
        raise RuntimeError(f"Missing prompt key {PROMPT_KEY} in {PROMPT_PATH}")

    headers_yaml = yaml.safe_dump(NUGGET_SCHEMA_HEADERS, allow_unicode=True).strip()
    prompt = (
        template.replace("[[HEADERS_YAML]]", headers_yaml).replace("[[USER_QUERY]]", query)
    )

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "query": query,
            "related": [],
            "header_list": [],
            "error": "skip_no_api_key",
            "prompt": prompt,
            "raw_response": "",
        }

    setup_openai("", mode="openai")
    raw = query_openai(
        prompt,
        mode="openai",
        model=model or TEXT_MODEL,
        max_tokens=800,
        temperature=0.0,
    ) or ""

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


def map_query_to_nuggets(nuggets: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    """Legacy stub (jsonl pipeline). Not used by LLM header mapping CLI."""
    _ = nuggets
    return {}


def mapping_to_qrels_rows(mapping: dict[str, dict[str, object]]) -> list[tuple[str, str, str, int]]:
    return []


def mapping_to_run_rows(mapping: dict[str, dict[str, object]]) -> list[tuple[str, str, str, int, float, str]]:
    return []


def save_outputs(mapping: dict[str, dict[str, object]]) -> None:
    Path(OUTPUT_MAPPING_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MAPPING_JSON, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_QRELS, "w", encoding="utf-8") as f:
        for qid, subtopic, doc_id, rel in mapping_to_qrels_rows(mapping):
            f.write(f"{qid} {subtopic} {doc_id} {rel}\n")

    with open(OUTPUT_RUN, "w", encoding="utf-8") as f:
        for qid, q0, doc_id, rank, score, tag in mapping_to_run_rows(mapping):
            f.write(f"{qid} {q0} {doc_id} {rank} {score} {tag}\n")


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
        "--legacy-nuggets",
        action="store_true",
        help="Run legacy jsonl → mapping stub (writes empty qrels/run).",
    )
    args = parser.parse_args()

    if args.legacy_nuggets:
        nuggets = load_modelcard_nuggets()
        mapping = map_query_to_nuggets(nuggets)
        save_outputs(mapping)
        print(f"saved_mapping_json = {OUTPUT_MAPPING_JSON}")
        print(f"saved_qrels = {OUTPUT_QRELS}")
        print(f"saved_run = {OUTPUT_RUN}")
        return

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
        parser.error("Provide --query and/or --queries-file (or use --legacy-nuggets).")

    results: list[dict[str, Any]] = []
    for q in queries:
        results.append(map_query_to_nugget_headers(q, model=args.model))

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


if __name__ == "__main__":
    main()
