#!/usr/bin/env python3
"""Dump archived Hugging Face model-card markdown snapshots from the dataset parquet."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import duckdb

from src.config import CARD_CONTENT_RAW, OUTPUT_DIR

DEFAULT_MODEL_CARD_SNAPSHOT_DIR = Path(OUTPUT_DIR) / "card2nugget" / "modelcards"
RAW_SHARD_GLOB = "*.parquet"


def safe_modelcard_snapshot_name(model_id: str) -> str:
    return str(model_id).strip().replace("/", "__") + ".md"


def snapshot_path_for_model_id(model_id: str, output_dir: Path | str = DEFAULT_MODEL_CARD_SNAPSHOT_DIR) -> Path:
    return Path(output_dir) / safe_modelcard_snapshot_name(model_id)


def dump_modelcard_snapshots(
    model_ids: Iterable[str],
    *,
    raw_card_dir: Path | str = CARD_CONTENT_RAW,
    output_dir: Path | str = DEFAULT_MODEL_CARD_SNAPSHOT_DIR,
) -> dict[str, Path]:
    ids = sorted({str(mid).strip() for mid in model_ids if str(mid).strip()})
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ids:
        return {}

    parquet_glob = str(Path(raw_card_dir) / RAW_SHARD_GLOB)
    placeholders = ", ".join("?" for _ in ids)
    query = f"""
        SELECT modelId, card
        FROM read_parquet(?, union_by_name=true)
        WHERE CAST(modelId AS VARCHAR) IN ({placeholders})
    """
    with duckdb.connect(":memory:") as con:
        rows = con.execute(query, [parquet_glob, *ids]).fetchall()

    written: dict[str, Path] = {}
    for model_id_raw, card_raw in rows:
        model_id = str(model_id_raw or "").strip()
        if not model_id or model_id in written:
            continue
        snapshot_path = snapshot_path_for_model_id(model_id, out_dir)
        snapshot_path.write_text(str(card_raw or ""), encoding="utf-8")
        written[model_id] = snapshot_path
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump archived model-card markdown snapshots by model id.")
    parser.add_argument("model_ids", nargs="+", help="Hugging Face model ids, e.g. org/model.")
    parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_CARD_SNAPSHOT_DIR))
    parser.add_argument("--raw-card-dir", default=str(CARD_CONTENT_RAW))
    args = parser.parse_args()

    written = dump_modelcard_snapshots(args.model_ids, raw_card_dir=args.raw_card_dir, output_dir=args.output_dir)
    for model_id, path in written.items():
        print(f"{model_id}\t{path.resolve()}")


if __name__ == "__main__":
    main()
