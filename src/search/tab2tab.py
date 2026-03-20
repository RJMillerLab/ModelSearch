"""
Table to Table Search (Testing Tool)

Minimal wrapper over Blend_internal for the 4 supported search types:
- single_column
- multi_column
- keyword
- unionable
"""

import os
import sys
import time
import json
import threading
import argparse
from typing import List, Optional, Iterable, Any

import pandas as pd

from src.config import MODELLAKE_DB, TAB2TAB_OUTPUT_JSON, BLEND_INTERNAL_REPO


blend_path_abs = os.path.abspath(BLEND_INTERNAL_REPO)
if blend_path_abs in sys.path:
    sys.path.remove(blend_path_abs)
sys.path.insert(0, blend_path_abs)

_SingleColumnJoinSearch = None
_MultiColumnJoinSearch = None
_KeywordSearch = None
_UnionSearch = None

_import_lock = threading.Lock()
_config_lock = threading.Lock()


def _update_blend_config(db_path: str) -> str:
    """Update Blend_internal config.ini with the correct db_path before importing."""
    with _config_lock:
        if not os.path.exists(blend_path_abs):
            raise FileNotFoundError("Blend_internal not found. Please clone it first: git clone git@github.com:DoraDong-2023/Blend_internal.git")

        import configparser

        config_path = os.path.join(blend_path_abs, "config", "config.ini")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Blend_internal config.ini not found at {config_path}")

        config = configparser.ConfigParser()
        config.read(config_path)

        if not os.path.isabs(db_path):
            modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            db_path_abs = os.path.abspath(os.path.join(modelsearch_root, db_path))
        else:
            db_path_abs = os.path.abspath(db_path)

        if "Database" not in config:
            config["Database"] = {}
        config["Database"]["path"] = db_path_abs
        config["Database"]["dbms"] = "duckdb"
        config["Database"]["index_table"] = "modellake_index"

        tmp_path = config_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            config.write(f)
        os.replace(tmp_path, config_path)
        return config_path


def _lazy_import_blend() -> None:
    """Lazy import Blend_internal functions after config is set."""
    global _SingleColumnJoinSearch, _MultiColumnJoinSearch, _KeywordSearch, _UnionSearch
    if _SingleColumnJoinSearch is not None:
        return

    with _import_lock:
        if _SingleColumnJoinSearch is not None:
            return

        # Ensure Blend path has highest priority.
        if blend_path_abs in sys.path:
            sys.path.remove(blend_path_abs)
        sys.path.insert(0, blend_path_abs)

        from src.Tasks.SingleColumnJoinSearch import SingleColumnJoinSearch
        from src.Tasks.MultiColumnJoinSearch import MultiColumnJoinSearch
        from src.Tasks.KeywordSearch import KeywordSearch
        from src.Tasks.UnionSearch import UnionSearch

        _SingleColumnJoinSearch = SingleColumnJoinSearch
        _MultiColumnJoinSearch = MultiColumnJoinSearch
        _KeywordSearch = KeywordSearch
        _UnionSearch = UnionSearch


def _prepare_db_path(db_path: Optional[str]) -> str:
    if db_path:
        return db_path
    default_path = MODELLAKE_DB
    modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.abspath(os.path.join(modelsearch_root, default_path))


def search_single_column(query_values: Iterable[Any], k: int = 10, db_path: Optional[str] = None) -> List[int]:
    _update_blend_config(_prepare_db_path(db_path))
    _lazy_import_blend()
    plan = _SingleColumnJoinSearch(query_values, k)
    return plan.run()


def search_multi_column(query_dataset: pd.DataFrame, k: int = 10, db_path: Optional[str] = None) -> List[int]:
    _update_blend_config(_prepare_db_path(db_path))
    _lazy_import_blend()
    plan = _MultiColumnJoinSearch(query_dataset, k)
    return plan.run()


def search_keyword(query_values: List[str], k: int = 10, db_path: Optional[str] = None) -> List[int]:
    _update_blend_config(_prepare_db_path(db_path))
    _lazy_import_blend()
    plan = _KeywordSearch(query_values, k)
    return plan.run()


def search_unionable(query_dataset: pd.DataFrame, k: int = 10, db_path: Optional[str] = None) -> List[int]:
    _update_blend_config(_prepare_db_path(db_path))
    _lazy_import_blend()
    plan = _UnionSearch(query_dataset, k)
    return plan.run()


def search_table2table(
    query: Any,
    search_type: str = "single_column",
    k: int = 10,
    db_path: Optional[str] = None,
) -> List[int]:
    """Unified interface for the 4 supported table-to-table search types."""
    if search_type == "single_column":
        if not isinstance(query, (list, tuple, pd.Series)):
            raise ValueError("For single_column search, query must be an iterable of values")
        return search_single_column(query, k, db_path=db_path)
    if search_type == "multi_column":
        if not isinstance(query, pd.DataFrame):
            raise ValueError("For multi_column search, query must be a pandas DataFrame")
        return search_multi_column(query, k, db_path=db_path)
    if search_type == "keyword":
        if not isinstance(query, list) or not all(isinstance(x, str) for x in query):
            raise ValueError("For keyword search, query must be a list of strings")
        return search_keyword(query, k, db_path=db_path)
    if search_type == "unionable":
        if not isinstance(query, pd.DataFrame):
            raise ValueError("For unionable search, query must be a pandas DataFrame")
        return search_unionable(query, k, db_path=db_path)

    raise ValueError("Unknown search_type: must be 'single_column', 'multi_column', 'keyword', or 'unionable'")


def main() -> None:
    """CLI entry point for tab2tab search (testing tool)."""
    parser = argparse.ArgumentParser(description="Table to Table Search using Blend_internal (Testing Tool)")
    parser.add_argument("--search_type", choices=["single_column", "multi_column", "keyword", "unionable"], required=True, help="Type of search to perform")
    parser.add_argument("--query", default=None, required=True, help="For single_column/keyword: comma-separated values; for multi_column/unionable: CSV path.")
    parser.add_argument("--k", type=int, default=10, help="Number of results to return")
    args = parser.parse_args()

    start_time = time.time()
    db_path = MODELLAKE_DB
    output_path = TAB2TAB_OUTPUT_JSON
    _update_blend_config(db_path)

    if args.search_type in ("single_column", "keyword"):
        query = [x.strip() for x in args.query.split(",")]
    elif args.search_type in ("multi_column", "unionable"):
        query = pd.read_csv(args.query)

    results = search_table2table(query, args.search_type, args.k, db_path=db_path)
    print(f"Found {len(results)} tables:")
    for i, table_id in enumerate(results, 1):
        print(f"  {i}. Table ID: {table_id}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    result_data = {
        "query": query if isinstance(query, list) else str(query),
        "search_type": args.search_type,
        "k": args.k,
        "results": [int(tid) for tid in results],
        "num_results": len(results),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Results saved to {output_path}")
    print(f"\nTotal time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()

