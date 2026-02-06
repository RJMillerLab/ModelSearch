#!/usr/bin/env python3
"""
Generate markdown files for table comparison/viewing.

Usage:
  python -m src.postprocess.generate_table_md --table_ids 3690 46228 --output table_comparison.md
  python -m src.postprocess.generate_table_md --model_id google-bert/bert-base-uncased --output model_tables.md
"""

import os
import sys
import argparse
import pandas as pd
import duckdb
import json
from typing import List, Dict, Optional, Any
from pathlib import Path

# Repo root (this file is in src/postprocess/)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_classifications(json_path: str) -> Dict[int, str]:
    """Load classifications from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def _find_csv_file(filename: str) -> Optional[str]:
    """Find CSV file by basename in common data dirs."""
    basename = os.path.basename(filename)
    for d in [
        "data_citationlake/processed/deduped_hugging_csvs",
        "data_citationlake/processed/deduped_github_csvs",
        "data_citationlake/processed/tables_output",
        "data/raw",
    ]:
        p = _REPO_ROOT / d / basename
        if p.exists():
            return str(p)
    if os.path.exists(filename):
        return filename
    return None


def load_relationship_parquet(parquet_path: str) -> pd.DataFrame:
    path = parquet_path if os.path.isabs(parquet_path) else _REPO_ROOT / parquet_path
    return pd.read_parquet(path)


def get_table_metadata(
    tableid: int, db_path: str, index_table: str = "modellake_index"
) -> Optional[Dict[str, Any]]:
    """Get table metadata from modellake.db."""
    path = db_path if os.path.isabs(db_path) else _REPO_ROOT / db_path
    if not path.exists():
        return None
    con = duckdb.connect(str(path), read_only=True)
    try:
        q = f"SELECT DISTINCT tableid, filename, table_group, table_type FROM {index_table} WHERE tableid = ? AND rowid = -1 LIMIT 1"
        row = con.execute(q, [tableid]).fetchone()
        if not row:
            return None
        return {"tableid": row[0], "filename": row[1], "table_group": row[2], "table_type": row[3]}
    finally:
        con.close()


def load_table_csv(filename: str, max_rows: int = 50) -> Optional[pd.DataFrame]:
    csv_path = _find_csv_file(filename)
    if not csv_path:
        return None
    try:
        return pd.read_csv(csv_path, nrows=max_rows)
    except Exception:
        return None


def table_to_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "*Empty table*"
    df_display = df.head(max_rows)
    try:
        md = df_display.to_markdown(index=False, tablefmt="github")
    except ImportError:
        md = "```\n" + df_display.to_string(index=False) + "\n```"
    if len(df) > max_rows:
        md += f"\n\n*... and {len(df) - max_rows} more rows*"
    return md


def get_table_classification(tableid: int, classifications: Dict[int, str]) -> Optional[str]:
    return classifications.get(tableid)


def get_model_tables_from_db(
    model_id: str, db_path: str, relationship_parquet: str
) -> List[Dict[str, Any]]:
    """Get table metadata for all tables linked to model_id."""
    df = load_relationship_parquet(relationship_parquet)
    model_col = next((c for c in ["modelId", "model_id", "modelID"] if c in df.columns), None)
    csv_col = next(
        (c for c in ["csv_basename", "basename", "filename", "table_basename", "csv_path"] if c in df.columns),
        None,
    )
    if not model_col or not csv_col:
        return []
    basenames = df.loc[df[model_col] == model_id, csv_col].dropna().unique().tolist()
    if not basenames:
        return []
    path = db_path if os.path.isabs(db_path) else _REPO_ROOT / db_path
    con = duckdb.connect(str(path), read_only=True)
    try:
        placeholders = ",".join(["?" for _ in basenames])
        q = f"SELECT DISTINCT tableid, filename, table_group, table_type FROM modellake_index WHERE filename IN ({placeholders}) AND rowid = -1"
        rows = con.execute(q, basenames).fetchall()
        return [{"tableid": r[0], "filename": r[1], "table_group": r[2], "table_type": r[3]} for r in rows]
    finally:
        con.close()


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
        p = _REPO_ROOT / classification_json if not os.path.isabs(classification_json) else classification_json
        if p.exists():
            classifications = load_classifications(str(p))

    tables_metadata = []
    if model_id:
        db = db_path if os.path.isabs(db_path) else str(_REPO_ROOT / db_path)
        tables_metadata = get_model_tables_from_db(model_id, db, relationship_parquet)
        if not tables_metadata:
            print(f"No tables found for model {model_id}")
            return
    elif table_ids:
        db = db_path if os.path.isabs(db_path) else str(_REPO_ROOT / db_path)
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

    out = output_path if os.path.isabs(output_path) else _REPO_ROOT / output_path
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
    generate_markdown(
        table_ids=args.table_ids,
        model_id=args.model_id,
        db_path=args.db_path,
        classification_json=args.classification_json if (Path(args.classification_json).exists() or (_REPO_ROOT / args.classification_json).exists()) else None,
        relationship_parquet=args.relationship_parquet,
        output_path=args.output,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
