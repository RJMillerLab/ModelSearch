"""
Shared helpers for generating markdown from tables / modellake.db.
Used by generate_table_md (by table_ids or model_id) and generate_md_from_logs (from search logs).
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import duckdb

# Repo root (this file is in src/postprocess/)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_path(path: str) -> Path:
    if os.path.isabs(path):
        return Path(path)
    return REPO_ROOT / path


def load_classifications(json_path: str) -> Dict[int, str]:
    """Load table classification JSON: keys as int, values as label."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

def load_relationship_parquet(parquet_path: str) -> pd.DataFrame:
    path = resolve_path(parquet_path)
    return pd.read_parquet(str(path))


def get_table_metadata(
    tableid: int, db_path: str, index_table: str = "modellake_index"
) -> Optional[Dict[str, Any]]:
    """Get table metadata from modellake.db by tableid."""
    path = resolve_path(db_path)
    if not path.exists():
        return None
    with duckdb.connect(str(path), read_only=True) as con:
        q = f"SELECT DISTINCT tableid, filename, table_group, table_type FROM {index_table} WHERE tableid = ? AND rowid = -1 LIMIT 1"
        row = con.execute(q, [tableid]).fetchone()
        if not row:
            return None
        return {"tableid": row[0], "filename": row[1], "table_group": row[2], "table_type": row[3]}

def table_to_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render DataFrame as markdown code block (first max_rows)."""
    if df is None or df.empty:
        return "*Empty table*"
    df_display = df.head(max_rows)
    md = "```\n" + df_display.to_string(index=False) + "\n```"
    if len(df) > max_rows:
        md += f"\n\n*... and {len(df) - max_rows} more rows*"
    return md


def get_table_classification(tableid: int, classifications: Dict[int, str]) -> Optional[str]:
    return classifications.get(tableid)


def get_model_tables_from_db(
    model_id: str, db_path: str, relationship_parquet: str
) -> List[Dict[str, Any]]:
    """Get table metadata for all tables linked to model_id (from relationship parquet + modellake.db)."""
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
    path = resolve_path(db_path)
    if not path.exists():
        return []
    with duckdb.connect(str(path), read_only=True) as con:
        placeholders = ",".join(["?" for _ in basenames])
        q = f"SELECT DISTINCT tableid, filename, table_group, table_type FROM modellake_index WHERE filename IN ({placeholders}) AND rowid = -1"
        rows = con.execute(q, basenames).fetchall()
        return [{"tableid": r[0], "filename": r[1], "table_group": r[2], "table_type": r[3]} for r in rows]
