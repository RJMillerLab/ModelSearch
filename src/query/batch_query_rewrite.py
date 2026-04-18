#!/usr/bin/env python3
from __future__ import annotations

"""
Build OpenAI Batch input for polishing LitSearch-style queries.

Workflow:
  1) Generate batch input:
     python -m src.query.batch_query_rewrite \
       --input src/query/data/query/litsearch_query.jsonl \
       --output src/query/data/query/query_rewrite_batch_input.jsonl

  2) Submit with existing batch runner:
     python -m src.llm.batch \
       src/query/data/query/query_rewrite_batch_input.jsonl \
       src/query/data/query/query_rewrite_batch_output.jsonl

  3) Parse batch output:
     python -m src.query.batch_query_rewrite parse \
       --input src/query/data/query/query_rewrite_batch_output.jsonl \
       --output src/query/data/query/query_rewrite_polished.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "Polish scientific search queries with the smallest natural edit. "
    "Prefer replacing paper/studies/research papers/publications/articles/literature with model, method, approach, benchmark, or task when needed. "
    "Keep all other wording and structure unchanged when possible. "
    "Return JSON only: {\"query\":\"...\"}."
)

USER_TEMPLATE = "Query: {query}"


def load_queries(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_json_text(text: str) -> dict[str, Any]:
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


def extract_rewrite(text: str, fallback: str) -> str:
    obj = extract_json_text(text)
    if isinstance(obj, dict):
        query = obj.get("query")
        if isinstance(query, str) and query.strip():
            return query.strip()
    text = (text or "").strip()
    if text:
        return text.splitlines()[0].strip()
    return fallback


def build_batch_input(
    records: Iterable[dict[str, Any]],
    output_path: Path,
    model: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, rec in enumerate(records):
            query = str(rec.get("query") or "").strip()
            if not query:
                continue
            item = {
                "custom_id": f"rewrite-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TEMPLATE.format(query=query)},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 96,
                },
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_batch_output(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            custom_id = row.get("custom_id")
            response = row.get("response") or {}
            body = response.get("body") if isinstance(response, dict) else {}
            choices = body.get("choices") if isinstance(body, dict) else []

            response_text = ""
            if isinstance(choices, list) and choices:
                first = choices[0] or {}
                message = first.get("message") or {}
                response_text = str(message.get("content") or "")

            rewritten = extract_rewrite(response_text, "")
            if not rewritten:
                rewritten = ""

            out = {
                "custom_id": custom_id,
                "rewritten_query": rewritten,
                "response_text": response_text,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or parse batch input for query polishing.")
    subparsers = parser.add_subparsers(dest="command")

    build = subparsers.add_parser("build", help="Build batch input JSONL")
    build.add_argument("--input", type=Path, default=Path("src/query/data/query/litsearch_query.jsonl"))
    build.add_argument("--output", type=Path, default=Path("src/query/data/query/query_rewrite_batch_input.jsonl"))
    build.add_argument("--model", default="gpt-4o-mini")
    build.add_argument("--limit", type=int, default=None)

    parse = subparsers.add_parser("parse", help="Parse OpenAI batch output JSONL")
    parse.add_argument("--input", type=Path, required=True)
    parse.add_argument("--output", type=Path, default=Path("src/query/data/query/query_rewrite_polished.jsonl"))

    args = parser.parse_args()

    if args.command == "parse":
        parse_batch_output(args.input, args.output)
        print(f"saved_output={args.output}")
        return

    input_path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    records = load_queries(input_path)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No query records found.")

    build_batch_input(records, args.output, args.model)
    print(f"saved_batch_input={args.output}")
    print(f"rows={len(records)}")


if __name__ == "__main__":
    main()
