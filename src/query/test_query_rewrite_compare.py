#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("src/query/data/query/query_rewrite_batch_input.jsonl")
DEFAULT_OUTPUT = Path("src/query/data/query/query_rewrite_batch_output.jsonl")
DEFAULT_MD = Path("src/query/data/query/query_rewrite_compare_sample.md")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_query_from_output(row: dict[str, Any]) -> str:
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


def extract_query_from_input(row: dict[str, Any]) -> str:
    body = row.get("body") or {}
    messages = body.get("messages") if isinstance(body, dict) else []
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "user":
                text = str(msg.get("content") or "").strip()
                if text.startswith("Query:"):
                    return text[len("Query:") :].strip()
                return text
    return ""


def build_output_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        custom_id = str(row.get("custom_id") or "").strip()
        if custom_id:
            lookup[custom_id] = row
    return lookup


def _escape_md(text: str) -> str:
    return text.replace("|", r"\|").replace("\n", " ").strip()


def write_md(rows: list[dict[str, Any]], md_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Query Rewrite Comparison Sample\n\n")
        f.write("| custom_id | before_query | after_query |\n")
        f.write("| --- | --- | --- |\n")
        for row in rows:
            f.write(
                "| {cid} | {before} | {after} |\n".format(
                    cid=_escape_md(str(row.get("custom_id", ""))),
                    before=_escape_md(str(row.get("before_query", ""))),
                    after=_escape_md(str(row.get("after_query", ""))),
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare query rewrite batch input and output for the first N rows.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    if not output_path.exists():
        raise SystemExit(f"Output file not found: {output_path}")

    input_rows = load_jsonl(input_path)
    output_rows = load_jsonl(output_path)
    output_lookup = build_output_lookup(output_rows)

    sample_rows: list[dict[str, Any]] = []
    for row in input_rows[: max(0, args.limit)]:
        custom_id = str(row.get("custom_id") or "").strip()
        before_query = extract_query_from_input(row)
        output_row = output_lookup.get(custom_id, {})
        after_query = extract_query_from_output(output_row) if output_row else ""
        sample_rows.append(
            {
                "custom_id": custom_id,
                "before_query": before_query,
                "after_query": after_query,
            }
        )

    write_md(sample_rows, args.md)


if __name__ == "__main__":
    main()
