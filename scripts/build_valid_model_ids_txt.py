#!/usr/bin/env python3
"""
Build data/valid_model_ids_with_tables.txt from relationship parquet (Part 1 / training).
Run once; inference (demo backend) only loads this txt.

Extracts model_id that have at least one row with csv_basename in the parquet.
One model_id per line. Used by backend "Narrow down" to pick seed from query2modelcard results.
"""

import os
import sys
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_PARQUET = "data_citationlake/processed/modelcard_step3_dedup.parquet"
DEFAULT_OUTPUT = "data/valid_model_ids_with_tables.txt"


def main():
    parser = argparse.ArgumentParser(
        description="Build valid_model_ids_with_tables.txt from relationship parquet (training step)."
    )
    parser.add_argument(
        "--parquet",
        default=DEFAULT_PARQUET,
        help=f"Relationship parquet path (default: {DEFAULT_PARQUET})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output txt path, one model_id per line (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only print parquet schema (column names and dtypes) and exit.",
    )
    args = parser.parse_args()

    parquet_abs = os.path.join(REPO_ROOT, args.parquet) if not os.path.isabs(args.parquet) else args.parquet
    out_abs = os.path.join(REPO_ROOT, args.output) if not os.path.isabs(args.output) else args.output

    if not os.path.isfile(parquet_abs):
        print(f"Error: parquet not found: {parquet_abs}", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    df = pd.read_parquet(parquet_abs, columns=None)
    if args.schema_only:
        print("Parquet schema:")
        for c in df.columns:
            print(f"  {c!r}: {df[c].dtype}")
        print(f"Rows: {len(df)}")
        sys.exit(0)
    cols = list(df.columns)
    # Model ID column: try common names (case-sensitive then case-insensitive)
    id_candidates = ["modelId", "model_id", "modelID"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col is None:
        low = {c.lower(): c for c in df.columns}
        id_col = low.get("modelid") or low.get("model_id")
    # Table/file column: any that look like csv basename or table reference
    base_candidates = ["csv_basename", "csvBasename", "basename", "filename", "table_basename", "csv_path", "table_id"]
    base_col = next((c for c in base_candidates if c in df.columns), None)
    if base_col is None:
        low = {c.lower(): c for c in df.columns}
        base_col = low.get("csv_basename") or low.get("basename") or low.get("filename")
    if id_col is None or base_col is None:
        print("Error: parquet must have a model-id column and a table/csv column.", file=sys.stderr)
        print("Schema (columns and dtypes):", file=sys.stderr)
        for c in cols:
            print(f"  {c!r}: {df[c].dtype}", file=sys.stderr)
        print("Expected model-id: one of modelId, model_id, modelID", file=sys.stderr)
        print("Expected table/csv: one of csv_basename, basename, filename", file=sys.stderr)
        sys.exit(1)

    has_base = df[base_col].notna() & (df[base_col].astype(str).str.strip() != "")
    valid = sorted(df.loc[has_base, id_col].astype(str).str.strip().unique())

    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as f:
        for mid in valid:
            f.write(mid + "\n")

    print(f"Wrote {len(valid)} valid model IDs to {out_abs}")


if __name__ == "__main__":
    main()
