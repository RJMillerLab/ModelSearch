"""
Build a txt file of model_id that have tables (non-empty csv_basename in relationship data).

Uses MODEL_TO_TABLES_EXPLODE_PARQUET (same index as load_modelid_to_csvlist / Card2Tab2Card).
Build the parquet first::

  python scripts/build_model_to_tables_explode_parquet.py

Aligns with the same explode parquet as `load_modelid_to_csvlist` / table search. Useful for offline allowlists and debugging.

Usage:
  python -m src.utils.build_valid_model_ids_txt --output data/valid_model_ids_with_tables_hugging.txt --resources hugging
"""

import os
import sys
import argparse

import duckdb

# Run from repo root so src imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import INDEX_TABLE, MODEL_TO_TABLES_EXPLODE_PARQUET
from src.utils import list_model_ids_with_tables_from_explode, list_model_ids_with_tables_from_explode_filtered_by_modellake_db


def main():
    parser = argparse.ArgumentParser(description="Extract model IDs that have tables from exploded parquet → one model_id per line.")
    parser.add_argument("--resources", nargs="+", default=["hugging", "github", "arxiv"], choices=["hugging", "github", "arxiv", "llm"], help="Resource labels (same as Card2Tab2Card --resources).")
    parser.add_argument("--output", required=True, help="Output txt path (e.g. data/valid_model_ids_with_tables.txt)")
    parser.add_argument("--duckdb_path", default="", help="Optional DuckDB path. If provided, only keep tables whose filename exists in modellake_index.")
    parser.add_argument("--index_table", default=INDEX_TABLE, help=f"DuckDB index table name (default: {INDEX_TABLE})")
    parser.add_argument("--filename_col", default="filename", help="DuckDB column name containing the table filename (default: filename)")
    parser.add_argument("--explode_parquet", default=MODEL_TO_TABLES_EXPLODE_PARQUET, help="Exploded parquet path to read (default: config.MODEL_TO_TABLES_EXPLODE_PARQUET)")
    args = parser.parse_args()

    if args.duckdb_path:
        db_abs = os.path.abspath(args.duckdb_path)
        with duckdb.connect(db_abs, read_only=True) as con:
            n_files = con.execute(
                f"SELECT COUNT(DISTINCT regexp_extract(CAST({args.filename_col} AS VARCHAR), '[^/]+$')) FROM {args.index_table} WHERE {args.filename_col} IS NOT NULL"
            ).fetchone()[0]
        print(f"DB unique filenames (basename): {n_files}")

        # Baseline: without filename filter, only resource filter + non-empty csv_basename
        baseline_ids = list_model_ids_with_tables_from_explode(
            resources=args.resources,
            explode_parquet=args.explode_parquet,
        )
        print(f"Baseline (no duckdb filename filter): {len(baseline_ids)} model IDs")

        valid_ids = list_model_ids_with_tables_from_explode_filtered_by_modellake_db(
            resources=args.resources,
            db_path=args.duckdb_path,
            index_table=args.index_table,
            filename_col=args.filename_col,
            explode_parquet=args.explode_parquet,
        )

        filtered_out = max(0, len(baseline_ids) - len(valid_ids))
        print(f"Filtered out: {filtered_out} model IDs (kept {len(valid_ids)})")
    else:
        valid_ids = list_model_ids_with_tables_from_explode(args.resources)

    out_abs = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for mid in valid_ids:
            f.write(mid + "\n")

    print(f"Wrote {len(valid_ids)} model IDs to {args.output}")


if __name__ == "__main__":
    main()
