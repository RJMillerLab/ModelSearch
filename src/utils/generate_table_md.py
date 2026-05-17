#!/usr/bin/env python3
"""
Generate markdown files for table comparison/viewing.

Usage:
  python -m src.utils.generate_table_md --table_ids 3690 46228 --output table_comparison.md
  python -m src.utils.generate_table_md --model_id google-bert/bert-base-uncased --output model_tables.md
  python -m src.utils.generate_table_md --csv_files 9ba9372b41_table1.csv a48291951a_table1.csv --output csv_preview.md
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List
import pandas as pd

from src.config import MODELLAKE_DB, CLASSIFICATION_JSON
from src.utils import resolve_table_path
from . import (
    get_repo_root,
    load_classifications,
    get_tables_metadata,
    table_to_markdown,
    get_model_tables_from_db,
)

def generate_markdown(
    table_ids: Optional[List[int]] = None,
    model_id: Optional[str] = None,
    csv_files: Optional[List[str]] = None,
    db_path: str = None,
    classification_json: Optional[str] = None,
    output_path: str = "table_comparison.md",
    max_rows: int = 50,
) -> None:
    repo_root = get_repo_root()
    db_path = db_path or MODELLAKE_DB
    classification_json = classification_json or CLASSIFICATION_JSON
    classifications = {}
    if classification_json and os.path.exists(classification_json):
        classifications = load_classifications(classification_json)

    tables_metadata = []
    if model_id:
        tables_metadata = get_model_tables_from_db(model_id, db_path)
        if not tables_metadata:
            print(f"No tables found for model {model_id}")
            return
    elif table_ids:
        tables_metadata = get_tables_metadata(table_ids, db_path)
    elif csv_files:
        seen = set()
        for csv in csv_files:
            filename = os.path.basename(str(csv).strip())
            if not filename or filename in seen:
                continue
            seen.add(filename)
            tables_metadata.append(
                {
                    "tableid": f"csv:{filename}",
                    "filename": filename,
                    "table_group": "N/A",
                    "table_type": "N/A",
                }
            )
    else:
        print("Must provide --table_ids, --model_id, or --csv_files")
        return

    if not tables_metadata:
        print("No tables to write")
        return

    lines = []
    if model_id:
        lines.append(f"# Tables for Model: `{model_id}`\n\n**Total tables:** {len(tables_metadata)}\n\n---\n")
    elif csv_files:
        lines.append(f"# CSV Preview\n\n**Total tables:** {len(tables_metadata)}\n\n---\n")
    else:
        lines.append(f"# Table Comparison\n\n**Total tables:** {len(tables_metadata)}\n\n---\n")

    for i, meta in enumerate(tables_metadata, 1):
        tid, filename = meta["tableid"], meta["filename"]
        lines.append(f"\n## Table {i}: ID `{tid}`\n")
        lines.append(f"- **Filename:** `{filename}`")
        lines.append(f"- **Table Group:** `{meta.get('table_group', 'N/A')}`")
        lines.append(f"- **Table Type:** `{meta.get('table_type', 'N/A')}`")
        if tid in classifications:
            lines.append(f"- **Classification:** `{classifications[tid]}`")
        lines.append("")
        full_path = resolve_table_path(filename)
        df = pd.read_csv(full_path, nrows=max_rows)

        if df is not None:
            lines.append(f"### Preview ({len(df)} rows, {len(df.columns)} columns)\n")
            lines.append(table_to_markdown(df, max_rows=max_rows))
            lines.append("\n### Columns\n")
            for col in df.columns:
                lines.append(f"- `{col}` ({df[col].dtype}): {df[col].notna().sum()}/{len(df)} non-null")
        else:
            lines.append("⚠️  **CSV file not found**\n")
        lines.append("---\n")

    out = Path(output_path) if os.path.isabs(output_path) else repo_root / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Generated: {out}")


def main():
    ap = argparse.ArgumentParser(description="Generate markdown for table(s) by ID, model ID, or CSV files")
    ap.add_argument("--table_ids", type=int, nargs="+", default=None)
    ap.add_argument("--model_id", type=str, default=None)
    ap.add_argument("--csv_files", type=str, nargs="+", default=None, help="CSV basenames or paths")
    ap.add_argument("--output", type=str, default="table_comparison.md")
    ap.add_argument("--max_rows", type=int, default=50)
    args = ap.parse_args()
    selected_modes = sum(bool(v) for v in [args.table_ids, args.model_id, args.csv_files])
    if selected_modes == 0:
        ap.error("Provide --table_ids, --model_id, or --csv_files")
    if selected_modes > 1:
        ap.error("Use only one of: --table_ids, --model_id, --csv_files")
    generate_markdown(
        table_ids=args.table_ids,
        model_id=args.model_id,
        csv_files=args.csv_files,
        db_path=MODELLAKE_DB,
        classification_json=CLASSIFICATION_JSON,
        output_path=args.output,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
