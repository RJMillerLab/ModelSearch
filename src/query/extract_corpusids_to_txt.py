#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_corpusids(jsonl_path: Path) -> list[str]:
    corpusids: list[str] = []
    seen: set[str] = set()

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            values = record.get("corpusids") or []
            if not isinstance(values, list):
                continue
            for value in values:
                cid = str(value).strip()
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                corpusids.append(cid)

    return corpusids


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract unique corpusids from a LitSearch JSONL file.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/query/data/query/litsearch_query.jsonl"),
        help="Path to the LitSearch query JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/query/data/query/new_corpusids.txt"),
        help="Path to the output TXT file.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    corpusids = load_corpusids(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(corpusids) + ("\n" if corpusids else ""), encoding="utf-8")

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"unique_corpusids={len(corpusids)}")


if __name__ == "__main__":
    main()
