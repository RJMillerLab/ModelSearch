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
    args = parser.parse_args()

    parquet_abs = os.path.join(REPO_ROOT, args.parquet) if not os.path.isabs(args.parquet) else args.parquet
    out_abs = os.path.join(REPO_ROOT, args.output) if not os.path.isabs(args.output) else args.output

    if not os.path.isfile(parquet_abs):
        print(f"Error: parquet not found: {parquet_abs}", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    df = pd.read_parquet(parquet_abs, columns=None)
    id_col = "modelId" if "modelId" in df.columns else ("model_id" if "model_id" in df.columns else None)
    base_col = "csv_basename" if "csv_basename" in df.columns else None
    if id_col is None or base_col is None:
        print(f"Error: parquet must have modelId (or model_id) and csv_basename columns", file=sys.stderr)
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
