"""
Table to Table Search (Blend_internal) via in-process import.

The legacy subprocess wrapper is preserved in `src/search/tab2tab_CLI.py`.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

import pandas as pd

#from src.config import MODELLAKE_DB, TAB2TAB_OUTPUT_JSON, BLEND_INTERNAL_REPO, INDEX_TABLE
from src.config import *
from src.utils import resolve_table_path


def _normalize_header_token_for_index(s: str) -> str:
    """
    Match Blend_internal src.utils.df_to_index header path:
    col_name = str(df.columns[col_counter]).lower().strip()[:200]
    """
    return str(s).lower().strip()[:200]


def _normalize_cell_token_for_index(s: str) -> str:
    """
    Match Blend_internal src.utils.df_to_index body path (tokenized cell string).
    """
    t = (
        str(s)
        .lower()
        .replace("\\", "")
        .replace("'", "")
        .replace('"', "")
        .replace("\t", "")
        .replace("\n", "")
        .replace("\r", "")
        .strip()[:200]
    )
    if t in ("nan", "none", ""):
        return ""
    return t


def _extract_keyword_query_from_table(table_path: str) -> List[str]:
    """Build keyword query from table headers (normalized like DuckDB index)."""
    out: List[str] = []
    seen = set()
    try:
        df = pd.read_csv(table_path, nrows=0)
    except Exception:
        return []
    for c in df.columns:
        s = _normalize_header_token_for_index(str(c))
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _extract_single_column_query_from_table(table_path: str, max_rows_per_table: int = 100) -> List[str]:
    """Build single-column query from first-column values (normalized like DuckDB index)."""
    try:
        df = pd.read_csv(table_path, nrows=max_rows_per_table)
    except Exception:
        return []
    if len(df.columns) == 0:
        return []
    vals = df[df.columns[0]].dropna().astype(str).tolist()
    out: List[str] = []
    seen = set()
    for v in vals:
        s = _normalize_cell_token_for_index(v)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _load_blend_tab2tab_module():
    from others.Blend_internal.scripts import tab2tab as blend_tab2tab
    return blend_tab2tab


def _ensure_blend_exists() -> None:
    if not BLEND_INTERNAL_REPO or not os.path.exists(BLEND_INTERNAL_REPO):
        raise FileNotFoundError(
            "Blend_internal not found. Please clone it into `others/Blend_internal` "
            f"(expected at: {BLEND_INTERNAL_REPO})."
        )


def search_table2table(
    *,
    search_type: str,
    query: Any,
    k: int,
    db_path: Optional[str] = None,
    output_json: Optional[str] = None,
    parse_payload: bool = True,
) -> List[str]:
    """
    Run Blend_internal tab2tab in-process and return CSV **basenames** only.

    The query table basename is filtered out from results if present.
    """
    _ensure_blend_exists()
    blend_tab2tab = _load_blend_tab2tab_module()

    # Used for debug prints (keyword/single_column paths).
    query_tokens: List[str] = []
    blend_query: Any

    if not isinstance(query, str):
        raise ValueError("query must be a table name/path string")
    table_path = resolve_table_path(query) or (query if os.path.exists(query) else None)
    if not table_path:
        raise ValueError(f"No existing table found for query: {query!r}")

    if search_type in ("single_column", "keyword"):
        if search_type == "keyword":
            query_tokens = _extract_keyword_query_from_table(table_path)
        else:
            query_tokens = _extract_single_column_query_from_table(table_path)
        if not query_tokens:
            raise ValueError(f"Empty extracted query from table path: {table_path!r}")
        print(
            f"[tab2tab-debug] search_type={search_type} "
            f"table={os.path.basename(table_path)} "
            f"query_tokens={len(query_tokens)} "
            f"tokens_preview={query_tokens[:20]!r}",
            flush=True,
        )
        blend_query = query_tokens
    elif search_type in ("multi_column", "unionable"):
        try:
            blend_query = pd.read_csv(table_path)
        except Exception as e:
            raise ValueError(f"Failed to load query table for {search_type}: {table_path!r}: {e}") from e
    else:
        raise ValueError(f"Unknown search_type: {search_type}")

    raw_table_ids = blend_tab2tab.search_table2table(
        query=blend_query,
        search_type=search_type,
        k=k,
        db_path=db_path,
    )
    table_ids = [int(tid) for tid in (raw_table_ids or [])]
    table_names = blend_tab2tab._table_ids_to_table_names(table_ids, db_path=db_path)
    query_bn = os.path.basename(table_path)
    filenames = [os.path.basename(str(name).strip()) for name in table_names if str(name).strip()]
    filenames = [f for f in filenames if f != query_bn]

    if output_json:
        parent = os.path.dirname(os.path.abspath(output_json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(filenames, f, ensure_ascii=False, indent=2)

    if not parse_payload:
        return []
    return filenames

def main() -> None:
    """
    Minimal CLI for local testing (kept consistent with build_index.md usage).
    """
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Table to Table Search using Blend_internal (in-process wrapper)")
    parser.add_argument("--search_type", choices=["single_column", "multi_column", "keyword", "unionable"], required=True)
    parser.add_argument(
        "--query",
        default=None,
        required=True,
        help="Input table path/name. keyword/single_column extract normalized tokens (JSON array sent to Blend); multi_column/unionable pass table path directly.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output_json", default="", help="Optional output json path to write Blend_internal results.")
    parser.add_argument("--resources", nargs="+", default=["hugging"], choices=["hugging", "github", "arxiv"], help="Optional table resource filter on tab2tab results.")
    args = parser.parse_args()

    resources = [str(r).strip().lower() for r in (args.resources or []) if str(r).strip()]
    resource_set = set(resources)
    if resource_set == {'hugging'}:
        db_path = MODELLAKE_DB_HUGGING
    elif resource_set == {'hugging', 'github', 'arxiv'}:
        db_path = MODELLAKE_DB
    else:
        raise NotImplementedError(f"Unsupported resource combination: {resource_set}. Must be one of: {'hugging', 'github', 'arxiv'}")

    print(
        "[tab2tab CLI] artifacts: "
        f"resources={resources!r} | "
        f"modellake_db={os.path.abspath(db_path)} | "
        f"blend_repo={os.path.abspath(BLEND_INTERNAL_REPO or '')}",
        flush=True,
    )

    start = time.time()
    results = search_table2table(search_type=args.search_type, query=args.query, k=args.k, db_path=db_path, output_json=args.output_json)
    if args.output_json:
        print(f"Saved json: {args.output_json}")
    else:
        print(f"Found {len(results)} tables: {results}")
    print(f"Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()

