#!/usr/bin/env python3
"""
python -m src.query.query_rewrite_batch \
  --input src/query/data/query/litsearch_query.jsonl \
  --output src/query/data/query/query_rewrite_batch_input.jsonl \
  --mode batch_input \
  --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

from src.llm.model import query_openai, setup_openai


DEFAULT_SYSTEM_PROMPT = (
    "Rewrite scientific search queries into model-oriented queries. "
    "Use the smallest natural edit. "
    "Return JSON only: {\"query\":\"...\"}."
)

DEFAULT_USER_TEMPLATE = "Query: {query}"


def load_queries(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_prompt(query: str) -> str:
    return DEFAULT_USER_TEMPLATE.format(query=query.strip())


def extract_json_object(text: str) -> dict[str, Any]:
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


def normalize_output_query(response_text: str, fallback_query: str) -> str:
    obj = extract_json_object(response_text)
    if isinstance(obj, dict):
        query = obj.get("query")
        if isinstance(query, str) and query.strip():
            return query.strip()
    text = (response_text or "").strip()
    if text:
        return text.splitlines()[0].strip()
    return fallback_query


def make_batch_input(
    records: Sequence[dict[str, Any]],
    output_path: Path,
    model: str,
    system_prompt: str,
    user_template: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, rec in enumerate(records):
            query = str(rec.get("query") or "").strip()
            if not query:
                continue
            req = {
                "custom_id": f"rewrite-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_template.format(query=query)},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 256,
                },
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def run_direct(
    records: Sequence[dict[str, Any]],
    output_path: Path,
    llm_model: str,
    system_prompt: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    setup_openai("", mode="openai")

    with output_path.open("w", encoding="utf-8") as f:
        for idx, rec in enumerate(records):
            query = str(rec.get("query") or "").strip()
            if not query:
                continue
            prompt = f"{system_prompt}\n\n{build_prompt(query)}"
            response_text = query_openai(
                prompt,
                mode="openai",
                model=llm_model,
                max_tokens=256,
                temperature=0.0,
            )
            rewritten = normalize_output_query(response_text, query)
            row = {
                "id": idx,
                "query_set": rec.get("query_set"),
                "original_query": query,
                "rewritten_query": rewritten,
                "response_text": response_text,
                "corpusids": rec.get("corpusids"),
                "specificity": rec.get("specificity"),
                "quality": rec.get("quality"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{idx + 1}/{len(records)}] done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite literature-search queries into model-oriented queries.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/query/data/query/litsearch_query.jsonl"),
        help="Input JSONL containing query records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/query/data/query/query_rewrite_output.jsonl"),
        help="Output JSONL for rewritten queries, or batch input file when --mode batch_input.",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "batch_input"],
        default="direct",
        help="direct = call model for each query now; batch_input = write an OpenAI Batch input file.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name for direct mode or batch input generation.",
    )
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used for rewrite.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick tests.",
    )
    args = parser.parse_args()

    load_dotenv()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    records = load_queries(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No query records found.")

    if args.mode == "batch_input":
        make_batch_input(
            records=records,
            output_path=args.output,
            model=args.model,
            system_prompt=args.system_prompt,
            user_template=DEFAULT_USER_TEMPLATE,
        )
        print(f"saved_batch_input={args.output}")
        print(f"rows={len(records)}")
    else:
        run_direct(
            records=records,
            output_path=args.output,
            llm_model=args.model,
            system_prompt=args.system_prompt,
        )
        print(f"saved_output={args.output}")
        print(f"rows={len(records)}")


if __name__ == "__main__":
    main()
