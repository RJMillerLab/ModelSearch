"""
Table to Table Search (Blend_internal) via in-process import.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from contextlib import contextmanager
from typing import Any, List, Optional

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
    augmentation_types: Optional[List[str]] = None,
    output_json: Optional[str] = None,
    parse_payload: bool = True,
) -> List[str]:
    """
    Run Blend_internal table_searcher in-process and return CSV **basenames** only.

    The query table basename is filtered out from results if present.
    """
    if not isinstance(query, str):
        raise ValueError("query must be a table name/path string")
    table_path = resolve_table_path(query) or (query if os.path.exists(query) else None)
    if not table_path:
        raise ValueError(f"No existing table found for query: {query!r}")

    _ensure_blend_exists()
    db_path = db_path or MODELLAKE_DB
    aug_types = augmentation_types or ["ori", "tr", "str"]

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
            searcher = TableKeywordSearcher(
                index_table=INDEX_TABLE,
                augmentation_types=aug_types,
                db_path=db_path,
            )
        elif search_type_l == "single_column":
            # Match original semantics: "cell values only" (no header tokens).
            searcher = TableSingleColJoinableSearcher(
                index_table=INDEX_TABLE,
                augmentation_types=aug_types,
                mode="without_header",
                db_path=db_path,
            )
        elif search_type_l == "unionable":
            # Match original semantics: per-column overlap over "cell values".
            searcher = TableUnionableSearcher(
                index_table=INDEX_TABLE,
                augmentation_types=aug_types,
                mode="without_header",
                db_path=db_path,
            )
        elif search_type_l == "multi_column":
            # Match original semantics: MultiColumnOverlap uses cell values only.
            searcher = TableMultiColJoinableSearcher(
                index_table=INDEX_TABLE,
                augmentation_types=aug_types,
                mode="without_header",
                db_path=db_path,
            )
        else:
            raise ValueError(
                f"Unknown search_type: {search_type!r}. Expected one of: keyword/single_column/unionable/multi_column."
            )

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

