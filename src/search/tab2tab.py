"""
Table to Table Search (Blend_internal) via in-process import.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import pandas as pd

#from src.config import MODELLAKE_DB, TAB2TAB_OUTPUT_JSON, BLEND_INTERNAL_REPO, INDEX_TABLE
from src.config import *
from src.utils import resolve_table_path

def _load_blend_tab2tab_module():
    from others.Blend_internal.scripts import tab2tab as blend_tab2tab
    return blend_tab2tab


_BLEND_SRC_ISOLATION_LOCK = threading.RLock()


@contextmanager
def _blend_internal_src_isolation():
    """
    Strictly isolate Blend_internal's `src` during import+call.

    Collision source: both repos use the top-level package name `src`,
    so `src.utils` / `src.Operators.*` may resolve to the wrong code.
    """
    with _BLEND_SRC_ISOLATION_LOCK:
        if not BLEND_INTERNAL_REPO or not os.path.exists(BLEND_INTERNAL_REPO):
            yield
            return

        orig_sys_path = list(sys.path)
        orig_src_modules = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}

        # Force import resolution to Blend_internal's `src` directory.
        for k in list(orig_src_modules.keys()):
            sys.modules.pop(k, None)

        # Prepend so imports see Blend_internal first.
        if BLEND_INTERNAL_REPO in sys.path:
            sys.path.remove(BLEND_INTERNAL_REPO)
        sys.path.insert(0, BLEND_INTERNAL_REPO)

        try:
            yield
        finally:
            # Clean any Blend_internal `src*` modules loaded during the context.
            for k in list(sys.modules.keys()):
                if k == "src" or k.startswith("src."):
                    sys.modules.pop(k, None)

            # Restore our original `src*` module set.
            sys.path = orig_sys_path
            sys.modules.update(orig_src_modules)


def _ensure_blend_exists() -> None:
    if not BLEND_INTERNAL_REPO or not os.path.exists(BLEND_INTERNAL_REPO):
        raise FileNotFoundError(f"Blend_internal not found. Please clone it into `others/Blend_internal` (expected at: {BLEND_INTERNAL_REPO}).")


def search_table2table(
    *,
    search_type: str,
    query: Any,
    k: int,
    con_data: Any,
    augmentation_types: List[str],
    output_json: Optional[str] = None,
    parse_payload: bool = True,
) -> List[str]:
    """
    Run Blend_internal table_searcher in-process and return CSV **basenames** only.

    The query table basename is filtered out from results if present.
    """
    if not isinstance(query, str):
        raise ValueError("query must be a table name/path string")
    # Accept three query forms:
    # 1) basename resolvable in TABLE_BASE_DIRS
    # 2) existing filesystem path
    # 3) raw filename stored in Blend index (e.g. *_t.csv / *_s.csv)
    table_path = resolve_table_path(query) or (query if os.path.exists(query) else query.strip())
    if not table_path:
        raise ValueError(f"No existing or indexed table found for query: {query!r}")

    query_bn = os.path.basename(table_path)
    search_type_l = str(search_type).strip().lower()

    with _blend_internal_src_isolation():
        from others.Blend_internal.scripts.table_searcher import (
            TableKeywordSearcher,
            TableSingleColJoinableSearcher,
            TableUnionableSearcher,
            TableMultiColJoinableSearcher,
        )

        if search_type_l == "keyword":
            searcher = TableKeywordSearcher(index_table=INDEX_TABLE, augmentation_types=augmentation_types, con_data=con_data)
        elif search_type_l == "single_column":
            # Match original semantics: "cell values only" (no header tokens).
            searcher = TableSingleColJoinableSearcher(index_table=INDEX_TABLE, augmentation_types=augmentation_types, mode="with_header", con_data=con_data)
        elif search_type_l == "unionable":
            # Match original semantics: per-column overlap over "cell values".
            searcher = TableUnionableSearcher(index_table=INDEX_TABLE, augmentation_types=augmentation_types, mode="with_header", con_data=con_data)
        elif search_type_l == "multi_column":
            # Match original semantics: MultiColumnOverlap uses cell values only.
            searcher = TableMultiColJoinableSearcher(index_table=INDEX_TABLE, augmentation_types=augmentation_types, mode="with_header", con_data=con_data)
        else:
            raise ValueError(f"Unknown search_type: {search_type!r}. Expected one of: keyword/single_column/unionable/multi_column.")

        # Pass full path so external-mode tokenization always works; internal-mode
        # can still happen when Blend_internal finds the exact filename in index.
        filenames = searcher.search_table(query_filename=table_path, k=k)
        filenames = [str(f).strip() for f in (filenames or []) if str(f).strip()]

    if output_json:
        parent = os.path.dirname(os.path.abspath(output_json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(filenames, f, ensure_ascii=False, indent=2)

    if not parse_payload:
        return []
    return filenames


def search_table2table_with_scores(
    *,
    search_type: str,
    query: Any,
    k: int,
    con_data: Any,
    augmentation_types: List[str],
    output_json: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run Blend_internal table_searcher in-process and return scored ranked results.

    Returned items are dicts:
      - ``filename``: Blend DB stored filename (e.g. ``foo.csv`` / ``foo_t.csv`` / ``foo_s.csv``)
      - ``score``: primary numeric score for reranking (definition depends on ``search_type``)
      - ``rank``: 1-based rank within this single run (after searcher rerank)

    Notes:
    - For some searchers (e.g. multi-column overlap), Blend_internal only exposes ordering; we use an
      order-derived proxy score so callers can still re-rank.
    """
    if not isinstance(query, str):
        raise ValueError("query must be a table name/path string")
    # Accept both local paths and indexed filenames (same semantics as search_table2table).
    table_path = resolve_table_path(query) or (query if os.path.exists(query) else query.strip())
    if not table_path:
        raise ValueError(f"No existing or indexed table found for query: {query!r}")

    _ensure_blend_exists()

    search_type_l = str(search_type).strip().lower()
    with _blend_internal_src_isolation():
        from others.Blend_internal.scripts.table_searcher import (
            TableKeywordSearcher,
            TableSingleColJoinableSearcher,
            TableUnionableSearcher,
            TableMultiColJoinableSearcher,
        )

        if search_type_l == "keyword":
            searcher = TableKeywordSearcher(index_table=INDEX_TABLE, augmentation_types=augmentation_types, con_data=con_data)
        elif search_type_l == "single_column":
            searcher = TableSingleColJoinableSearcher(index_table=INDEX_TABLE, augmentation_types=augmentation_types, mode="with_header", con_data=con_data)
        elif search_type_l == "unionable":
            searcher = TableUnionableSearcher(index_table=INDEX_TABLE, augmentation_types=augmentation_types, mode="with_header", con_data=con_data)
        elif search_type_l == "multi_column":
            searcher = TableMultiColJoinableSearcher(index_table=INDEX_TABLE, augmentation_types=augmentation_types, mode="with_header", con_data=con_data)
        else:
            raise ValueError(f"Unknown search_type: {search_type!r}. Expected one of: keyword/single_column/unionable/multi_column.")

        # Mirror BaseTableSearcher.search_table() but keep score-like detail.
        mode, payload = searcher.check_query_get_tid(table_path)
        if mode == "internal":
            tokens = searcher._tokens_from_index(payload["filename"])
        else:
            tokens = searcher._tokens_from_new(table_path)

        results_detail = searcher.search_table_detail(query_tokens=tokens, query_group=payload["group"], k=k)
        ranked_results_with_scores = searcher.rerank(results_detail, k=k)

        out: List[Dict[str, Any]] = []
        for i, item in enumerate(ranked_results_with_scores[:k], start=1):
            out.append({"filename": str(item[0]), "score": float(item[1]), "rank": i})

    if output_json:
        parent = os.path.dirname(os.path.abspath(output_json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    return out

def main() -> None:
    """
    Minimal CLI for local testing (kept consistent with build_index.md usage).
    """
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Table to Table Search using Blend_internal (in-process wrapper)")
    parser.add_argument("--search_type", choices=["single_column", "multi_column", "keyword", "unionable"], required=True)
    parser.add_argument("--query", required=True, help="Input table path/name. keyword/single_column extract normalized tokens (JSON array sent to Blend); multi_column/unionable pass table path directly.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output_json", default="", help="Optional output json path to write Blend_internal results.")
    parser.add_argument("--resources", nargs="+", default=["hugging"], choices=["hugging", "github", "arxiv"], help="Table resource filter on tab2tab results.")
    parser.add_argument("--augmentation_types", default="ori", help="Table augmentation types. Comma-separated or space-separated; e.g. 'ori' or 'ori,tr,str'.") # no space separator
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
    augmentation_types = [x.strip().lower() for x in args.augmentation_types.split(",") if x.strip()]
    allowed = {"ori", "tr", "str"}
    bad = [x for x in augmentation_types if x not in allowed]
    if bad:
        raise ValueError(f"Invalid augmentation_types={bad!r}. Allowed: {sorted(allowed)}")
    import duckdb
    con_data = duckdb.connect(db_path, read_only=True)
    results = search_table2table(search_type=args.search_type, query=args.query, k=args.k, output_json=args.output_json, con_data=con_data, augmentation_types=augmentation_types)
    con_data.close()
    if args.output_json:
        print(f"Saved json: {args.output_json}")
    else:
        print(f"Found {len(results)} tables: {results}")
    print(f"Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()

