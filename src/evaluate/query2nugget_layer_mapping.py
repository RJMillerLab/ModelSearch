#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.config import OUTPUT_DIR


INPUT_NUGGETS_JSONL = os.path.join(OUTPUT_DIR, "evaluate", "modelcard_nuggets.jsonl")
OUTPUT_MAPPING_JSON = os.path.join(OUTPUT_DIR, "evaluate", "query_nugget_mapping.json")
OUTPUT_QRELS = os.path.join(OUTPUT_DIR, "evaluate", "real_subtopic.qrels")
OUTPUT_RUN = os.path.join(OUTPUT_DIR, "evaluate", "real_initial.run")


def load_modelcard_nuggets(path: str = INPUT_NUGGETS_JSONL) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def map_query_to_nuggets(nuggets: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    """Build query -> nuggets mapping."""
    # TODO: paste the tested query-to-nugget mapping logic here.
    return {}


def mapping_to_qrels_rows(mapping: dict[str, dict[str, object]]) -> list[tuple[str, str, str, int]]:
    """Return qrels rows as (qid, subtopic, doc_id, rel)."""
    # TODO: convert query-nugget mapping to qrels rows.
    return []


def mapping_to_run_rows(mapping: dict[str, dict[str, object]]) -> list[tuple[str, str, str, int, float, str]]:
    """Return run rows as (qid, Q0, doc_id, rank, score, tag)."""
    # TODO: convert query-nugget mapping to run rows.
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
    nuggets = load_modelcard_nuggets()
    mapping = map_query_to_nuggets(nuggets)
    save_outputs(mapping)
    print(f"saved_mapping_json={OUTPUT_MAPPING_JSON}")
    print(f"saved_qrels={OUTPUT_QRELS}")
    print(f"saved_run={OUTPUT_RUN}")


if __name__ == "__main__":
    main()
