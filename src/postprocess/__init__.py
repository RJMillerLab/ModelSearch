# Postprocess: generate markdown from logs, table comparison MD. Pipeline + CSV search in .pipeline
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

__all__ = [
    "is_model_search_log",
    "is_table_search_log",
    "get_repo_root",
    "load_classifications",
    "get_tables_metadata",
    "table_to_markdown",
    "get_model_tables_from_db",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def is_model_search_log(log_name: str) -> bool:
    """True if this log is from a model-search pipeline (models first, then related tables)."""
    return any(kw in log_name for kw in ("card2card", "card2tab2card", "query2modelcard"))


def is_table_search_log(log_name: str) -> bool:
    """True if this log is from a table-search pipeline (tables only)."""
    return any(kw in log_name for kw in ("tab2tab",))


def get_repo_root() -> Path:
    """Repo root (ModelSearchDemo)."""
    return _REPO_ROOT

def load_classifications(json_path: str) -> Dict[int, str]:
    """Load table classification JSON: keys as int, values as label."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

def _row_to_table_metadata(row: Any) -> Dict[str, Any]:
    return {
        "tableid": row[0],
        "filename": row[1],
        "table_group": row[2],
        "table_type": row[3],
    }


def get_tables_metadata(
    tableids: List[int], db_path: str, index_table: str = "modellake_index"
) -> List[Dict[str, Any]]:
    import duckdb
    """Get table metadata from modellake.db by tableids, preserving input order."""
    normalized_tableids: List[int] = []
    seen = set()
    for tableid in tableids:
        if tableid is None:
            continue
        tableid_int = int(tableid)
        if tableid_int not in seen:
            seen.add(tableid_int)
            normalized_tableids.append(tableid_int)
    if not normalized_tableids:
        return []

    with duckdb.connect(db_path, read_only=True) as con:
        placeholders = ",".join(["?" for _ in normalized_tableids])
        q = (
            f"SELECT DISTINCT tableid, filename, table_group, table_type "
            f"FROM {index_table} "
            f"WHERE tableid IN ({placeholders}) AND rowid = -1"
        )
        rows = con.execute(q, normalized_tableids).fetchall()

    metadata_by_tableid = {int(row[0]): _row_to_table_metadata(row) for row in rows}
    return [
        metadata_by_tableid[tableid]
        for tableid in normalized_tableids
        if tableid in metadata_by_tableid
    ]

def table_to_markdown(df, max_rows: int = 50) -> str:
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


def get_model_tables_from_db(model_id: str, db_path: str) -> List[Dict[str, Any]]:
    """Get table metadata for all tables linked to model_id."""
    from src.utils import load_modelid_to_csvlist
    import duckdb
    basenames = load_modelid_to_csvlist(model_id)
    if not basenames:
        return []
    with duckdb.connect(db_path, read_only=True) as con:
        placeholders = ",".join(["?" for _ in basenames])
        q = f"SELECT DISTINCT tableid, filename, table_group, table_type FROM modellake_index WHERE filename IN ({placeholders}) AND rowid = -1"
        rows = con.execute(q, basenames).fetchall()
        return [_row_to_table_metadata(row) for row in rows]

