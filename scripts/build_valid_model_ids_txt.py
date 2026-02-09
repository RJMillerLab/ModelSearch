#!/usr/bin/env python3
"""
Build data/valid_model_ids_with_tables.txt from relationship parquet (Part 1 / training).
Run once; inference (demo backend) only loads this txt.

Extracts modelId that have at least one non-empty table list (schema: modelId + hugging/github/html/llm table list columns).
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

    # Hardcoded for data_citationlake/processed/modelcard_step3_dedup.parquet schema:
    # modelId (object) + table list columns: hugging_table_list_dedup, github_table_list_dedup,
    # html_table_list_mapped_dedup, llm_table_list_mapped_dedup (object, may be list or str)
    ID_COL = "modelId"
    TABLE_LIST_COLS = [
        "hugging_table_list_dedup",
        "github_table_list_dedup",
        "html_table_list_mapped_dedup",
        "llm_table_list_mapped_dedup",
    ]
    if ID_COL not in df.columns:
        print(f"Error: parquet missing column {ID_COL!r}. Schema: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    existing_table_cols = [c for c in TABLE_LIST_COLS if c in df.columns]
    if not existing_table_cols:
        print(f"Error: parquet missing any of {TABLE_LIST_COLS}. Schema: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    def _has_tables(row):
        for c in existing_table_cols:
            v = row[c]
            if pd.isna(v):
                continue
            if isinstance(v, (list, tuple)):
                if len(v) > 0:
                    return True
                continue
            s = str(v).strip()
            if s and s.lower() not in ("nan", "none", "[]", ""):
                return True
        return False

    mask = df.apply(_has_tables, axis=1)
    valid = sorted(df.loc[mask, ID_COL].astype(str).str.strip().unique())

    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as f:
        for mid in valid:
            f.write(mid + "\n")

    print(f"Wrote {len(valid)} valid model IDs to {out_abs}")


if __name__ == "__main__":
    main()
