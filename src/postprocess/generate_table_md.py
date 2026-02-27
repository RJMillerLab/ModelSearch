#!/usr/bin/env python3
"""
Generate markdown files for table comparison/viewing.

**用途**：按 table_ids 或 model_id 直接生成「表对比 / 某模型关联表」的 Markdown（不依赖日志）。
在 docs/build_index.md 里会用到，例如表格对比、按 model 看表。

Usage:
  python -m src.postprocess.generate_table_md --table_ids 3690 46228 --output table_comparison.md
  python -m src.postprocess.generate_table_md --model_id google-bert/bert-base-uncased --output model_tables.md
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List

from .table_md_common import (
    REPO_ROOT,
    resolve_path,
    load_classifications,
    get_table_metadata,
    load_table_csv,
    table_to_markdown,
    get_model_tables_from_db,
)


def generate_markdown(
    table_ids: Optional[List[int]] = None,
    model_id: Optional[str] = None,
    db_path: str = "data/modellake.db",
    classification_json: Optional[str] = "data/table_classifications.json",
    relationship_parquet: str = "data_citationlake/processed/modelcard_step3_dedup.parquet",
    output_path: str = "table_comparison.md",
    max_rows: int = 50,
) -> None:
    classifications = {}
    if classification_json:
        p = resolve_path(classification_json)
        if p.exists():
            classifications = load_classifications(str(p))

    tables_metadata = []
    if model_id:
        db = str(resolve_path(db_path))
        tables_metadata = get_model_tables_from_db(model_id, db, relationship_parquet)
        if not tables_metadata:
            print(f"No tables found for model {model_id}")
            return
    elif table_ids:
        db = str(resolve_path(db_path))
        for tid in table_ids:
            m = get_table_metadata(tid, db)
            if m:
                tables_metadata.append(m)
    else:
        print("Must provide --table_ids or --model_id")
        return

    if not tables_metadata:
        print("No tables to write")
        return

    lines = []
    if model_id:
        lines.append(f"# Tables for Model: `{model_id}`\n\n**Total tables:** {len(tables_metadata)}\n\n---\n")
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
        df = load_table_csv(filename, max_rows=max_rows)
        if df is not None:
            lines.append(f"### Preview ({len(df)} rows, {len(df.columns)} columns)\n")
            lines.append(table_to_markdown(df, max_rows=max_rows))
            lines.append("\n### Columns\n")
            for col in df.columns:
                lines.append(f"- `{col}` ({df[col].dtype}): {df[col].notna().sum()}/{len(df)} non-null")
        else:
            lines.append("⚠️  **CSV file not found**\n")
        lines.append("---\n")

    out = Path(output_path) if os.path.isabs(output_path) else REPO_ROOT / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Generated: {out}")


def main():
    ap = argparse.ArgumentParser(description="Generate markdown for table(s) by ID or model ID")
    ap.add_argument("--table_ids", type=int, nargs="+", default=None)
    ap.add_argument("--model_id", type=str, default=None)
    ap.add_argument("--db_path", default="data/modellake.db")
    ap.add_argument("--classification_json", default="data/table_classifications.json")
    ap.add_argument("--relationship_parquet", default="data_citationlake/processed/modelcard_step3_dedup.parquet")
    ap.add_argument("--output", "-o", default="table_comparison.md")
    ap.add_argument("--max_rows", type=int, default=50)
    args = ap.parse_args()
    if not args.table_ids and not args.model_id:
        ap.error("Provide --table_ids or --model_id")
    classification_json = None
    if args.classification_json:
        p = resolve_path(args.classification_json)
        if p.exists():
            classification_json = str(p)
    generate_markdown(
        table_ids=args.table_ids,
        model_id=args.model_id,
        db_path=args.db_path,
        classification_json=classification_json,
        relationship_parquet=args.relationship_parquet,
        output_path=args.output,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
