"""
Single entry point for loading a table: from modellake.db (by tableid) or from CSV path.
Try DB first when db_path + tableid are given; else resolve path and read CSV.
"""
import os
from typing import Optional, Dict, Any
from collections import defaultdict

import pandas as pd

# Base dirs for CSV lookup (same as table_integration)
TABLE_BASE_DIRS = [
    "data_citationlake/processed/deduped_hugging_csvs",
    "data_citationlake/processed/deduped_github_csvs",
    "data_citationlake/processed/tables_output",
]

_CACHED_BASENAME_TO_PATH: Optional[Dict[str, str]] = None


def _build_basename_index() -> Dict[str, str]:
    global _CACHED_BASENAME_TO_PATH
    if _CACHED_BASENAME_TO_PATH is not None:
        return _CACHED_BASENAME_TO_PATH
    index: Dict[str, str] = {}
    for base in TABLE_BASE_DIRS:
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


def resolve_table_path(basename: str) -> Optional[str]:
    """Resolve CSV basename to full path. Used by integration when only filename is known."""
    base = os.path.basename(basename)
    idx = _build_basename_index()
    if base in idx:
        return idx[base]
    for base_dir in TABLE_BASE_DIRS:
        p = os.path.join(base_dir, base)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def load_table_from_db(
    db_path: str,
    tableid: int,
    index_table: str = "modellake_index",
) -> Optional[pd.DataFrame]:
    """
    Load table content from modellake.db by tableid (no CSV).
    modellake_index: (tableid, rowid, colid, tokenized). rowid=-1 = header.
    Returns None if DB missing or schema not supported.
    """
    if not db_path or not os.path.isfile(db_path):
        return None
    try:
        import duckdb
    except ImportError:
        return None
    try:
        con = duckdb.connect(db_path, read_only=True)
        try:
            info = con.execute(f"DESCRIBE {index_table}").fetchall()
        except Exception:
            con.close()
            return None
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
    except Exception as e:
        print(f"⚠️  load_table_from_db({tableid}): {e}")
        return None


def load_table(
    path_or_basename: str,
    db_path: Optional[str] = None,
    tableid: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Load one table: try DB first (if db_path + tableid), else resolve path and read CSV.
    Call this everywhere you need table content (integration, etc.).
    """
    if db_path and tableid is not None:
        df = load_table_from_db(db_path, int(tableid))
        if df is not None:
            return df
    basename = os.path.basename(path_or_basename)
    resolved = resolve_table_path(basename)
    if resolved and os.path.exists(resolved):
        try:
            return pd.read_csv(resolved)
        except Exception as e:
            print(f"⚠️  Error loading {resolved}: {e}")
            return None
    if os.path.exists(path_or_basename):
        try:
            return pd.read_csv(path_or_basename)
        except Exception as e:
            print(f"⚠️  Error loading {path_or_basename}: {e}")
            return None
    possible_base_dirs = [
        "data_citationlake/processed/deduped_hugging_csvs",
        "data_citationlake/processed/deduped_github_csvs",
        "data_citationlake/processed/tables_output",
        "../CitationLake/data/processed/deduped_hugging_csvs",
        "../CitationLake/data/processed/deduped_github_csvs",
        "../CitationLake/data/processed/tables_output",
    ]
    for base_dir in possible_base_dirs:
        abs_base = os.path.abspath(base_dir)
        p = os.path.join(abs_base, basename)
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception as e:
                print(f"⚠️  Error loading {p}: {e}")
    print(f"⚠️  Table not found: {path_or_basename} (basename: {basename})")
    return None
