"""
Deprecated Scripts
"""
import os
from typing import Optional, Dict, Any, List
from collections import defaultdict
import duckdb
import pandas as pd

from src.config import TABLE_BASE_DIRS

_CACHED_BASENAME_TO_PATH: Optional[Dict[str, str]] = None


def _build_basename_index(table_base_dirs: List[str]) -> Dict[str, str]:
    """
    deprecated, used for building a cache of basename to path from local directories

    Usage:
    idx = _build_basename_index(dirs)
    if base in idx:
        return idx[base]
    """
    global _CACHED_BASENAME_TO_PATH
    if _CACHED_BASENAME_TO_PATH is not None:
        return _CACHED_BASENAME_TO_PATH
    index: Dict[str, str] = {}
    for base in table_base_dirs:
        abs_base = os.path.abspath(base)
        if not os.path.isdir(abs_base):
            continue
        try:
            for f in os.listdir(abs_base):
                if f.lower().endswith(".csv"):
                    if f not in index:
                        index[f] = os.path.join(abs_base, f)
        except OSError:
            continue
    _CACHED_BASENAME_TO_PATH = index
    return index

def _load_table_from_db(
    tableid: int,
    index_table: str = "modellake_index",
) -> Optional[pd.DataFrame]:
    """
    Load table content from modellake.db by tableid (no CSV).
    modellake_index: (tableid, rowid, colid, tokenized). rowid=-1 = header.
    Returns None if DB missing or schema not supported.

    Deprecated, because column type can not be preserved.
    """
    from src.config import MODELLAKE_DB
    db_path = MODELLAKE_DB
    if not db_path or not os.path.isfile(db_path):
        return None
    con = duckdb.connect(db_path, read_only=True)
    info = con.execute(f"DESCRIBE {index_table}").fetchall()
    col_names = [r[0] for r in info]
    if "tableid" not in col_names or "rowid" not in col_names or "colid" not in col_names:
        con.close()
        return None
    value_col = "tokenized" if "tokenized" in col_names else ("value" if "value" in col_names else None)
    if not value_col:
        con.close()
        return None
    headers = con.execute(
        f"SELECT colid, {value_col} FROM {index_table} WHERE tableid = ? AND rowid = -1 ORDER BY colid",
        [tableid],
    ).fetchall()
    if not headers:
        con.close()
        return None
    col_names_list = [str(h[1]) if h[1] is not None else f"col_{h[0]}" for h in headers]
    rows = con.execute(
        f"SELECT rowid, colid, {value_col} FROM {index_table} WHERE tableid = ? AND rowid >= 0 ORDER BY rowid, colid",
        [tableid],
    ).fetchall()
    con.close()
    if not rows:
        return pd.DataFrame(columns=col_names_list)
    row_data: Dict[int, Dict[int, Any]] = defaultdict(dict)
    for rowid, colid, val in rows:
        row_data[rowid][colid] = val
    row_ids = sorted(row_data.keys())
    data = [[row_data[rid].get(cid) for cid in [h[0] for h in headers]] for rid in row_ids]
    return pd.DataFrame(data, columns=col_names_list)