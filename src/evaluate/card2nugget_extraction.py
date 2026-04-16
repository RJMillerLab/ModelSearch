#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.config import CARD_CONTENT_RAW, OUTPUT_DIR


SHARD_GLOB = "train-0000*-of-00006.parquet"
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "evaluate", "modelcard_nuggets.jsonl")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "evaluate", "modelcard_nuggets.json")
SINGLE_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "evaluate", "single_modelcard_nuggets.json")


def load_model_cards():
    """Load only modelId and card from CARD_CONTENT_RAW parquet shards."""
    import pandas as pd

    parquet_paths = sorted(Path(CARD_CONTENT_RAW).glob(SHARD_GLOB))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet shards matched {Path(CARD_CONTENT_RAW) / SHARD_GLOB}")
    return pd.concat(
        (pd.read_parquet(path, columns=["modelId", "card"]) for path in parquet_paths),
        ignore_index=True,
    )


def extract_nuggets_from_card(model_id: str, card: str) -> list[dict[str, str]]:
    """Extract nuggets from one model card."""
    # TODO: paste the tested single-card nugget extraction logic here.
    return []


def extract_nuggets_batch() -> list[dict[str, str]]:
    df = load_model_cards()
    records: list[dict[str, str]] = []
    for row in df[["modelId", "card"]].itertuples(index=False):
        if row.modelId is None or row.card is None:
            continue
        model_id = str(row.modelId).strip()
        card = str(row.card).strip()
        if not model_id or not card:
            continue
        records.extend(extract_nuggets_from_card(model_id, card))
    return records


def save_batch_outputs(records: list[dict[str, str]]) -> None:
    Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    grouped: dict[str, dict[str, list[dict[str, str]]]] = {}
    for record in records:
        model_id = str(record["modelId"])
        grouped.setdefault(model_id, {"nuggets": []})["nuggets"].append(
            {
                "nugget_id": record["nugget_id"],
                "text": record["text"],
            }
        )
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)


def save_single_output(model_id: str, card: str, nuggets: list[dict[str, str]]) -> None:
    Path(SINGLE_OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "modelId": model_id,
        "card": card,
        "nuggets": nuggets,
    }
    with open(SINGLE_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_batch() -> None:
    records = extract_nuggets_batch()
    save_batch_outputs(records)
    print(f"saved_jsonl={OUTPUT_JSONL}")
    print(f"saved_json={OUTPUT_JSON}")
    print(f"nuggets={len(records)}")


def run_single(args: argparse.Namespace) -> None:
    nuggets = extract_nuggets_from_card(args.model_id, args.card)
    save_single_output(args.model_id, args.card, nuggets)
    print(f"saved_json={SINGLE_OUTPUT_JSON}")
    print(f"nuggets={len(nuggets)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract nuggets from model cards.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("batch", help="Run card-to-nugget extraction for all raw model cards.")

    single_parser = subparsers.add_parser("single", help="Run card-to-nugget extraction for one card.")
    single_parser.add_argument("--model-id", required=True)
    single_parser.add_argument("--card", required=True)

    args = parser.parse_args()
    if args.command == "batch":
        run_batch()
    elif args.command == "single":
        run_single(args)


if __name__ == "__main__":
    main()
