#!/usr/bin/env python3
"""
Standalone Card2Tab2Card by-type pipeline (no changes to src/search/card2tab2card.py).

Flow:
  1. Get all tables for the query model (from relationship parquet).
  2. Filter out s2orc/llm and generic tables.
  3. For each query table: infer type (via tab2tab_by_type) and search only same-type tables.
  4. Merge per-table results (filter self/s2orc/generic, top-k per table, round-robin merge).
  5. Map retrieved table IDs → filenames (DuckDB) → model IDs (relationship parquet).
  6. Write JSON in the same shape as card2tab2card (so demo can load it as by_type).

Usage:
  python scripts/run_card2tab2card_by_type.py \
    --model_id tdro-llm/s2-tdro-Qwen1.5-1.8B-top70 \
    --search_type keyword \
    --k 5 \
    --db_path data/modellake.db \
    --relationship_parquet data_citationlake/processed/modelcard_step3_dedup.parquet \
    --classification_json data/table_classifications.json \
    --output_json data/jobs/xxx/card2tab2card_by_type.json

Safe to revert: this script is additive; existing card2tab2card and tab2tab_by_type are unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd

# Project root for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.modelsearch.compare_baselines import (
    get_modelids_for_basenames_duckdb,
    get_tables_for_model_duckdb,
)
from src.search.tab2tab_by_type import search_table2table_by_type
from src.utils.table_loader import resolve_table_path

# ----- Helpers (no import from card2tab2card) -----

DEFAULT_RELATIONSHIP_PARQUET = "data_citationlake/processed/modelcard_step3_dedup.parquet"
GENERIC_TABLE_PATTERNS = ["1910.09700_table", "204823751_table"]


def _classify_table_source_by_basename(basename: str) -> str:
    """Infer source from filename (github / huggingface / html / llm / unknown)."""
    b = str(basename).replace("_s.csv", ".csv").replace("_t.csv", ".csv")
    if re.fullmatch(r"[0-9a-f]{32}_table_\d+\.csv", b):
        return "github"
    if re.fullmatch(r"\d+\.\d+(?:v\d+)?_table\d+\.csv", b):
        return "html"
    if re.fullmatch(r"[0-9a-f]{10}_table\d+\.csv", b):
        return "huggingface"
    if re.fullmatch(r"\d+_table\d+\.csv", b):
        return "llm"
    return "unknown"


def _is_generic_table(basename_or_path: str) -> bool:
    base = os.path.basename(str(basename_or_path))
    return any(p in base for p in GENERIC_TABLE_PATTERNS)


def _filter_s2orc_tables(tables: List[str]) -> List[str]:
    out = [t for t in tables if _classify_table_source_by_basename(os.path.basename(str(t))) != "llm"]
    if len(out) < len(tables):
        print(f"   Filtered out {len(tables) - len(out)} s2orc/llm tables (remain: {len(out)})")
    return out


def _resolve_csv_path(table_path: str) -> Optional[str]:
    p = str(table_path)
    if os.path.exists(p):
        return p
    return resolve_table_path(os.path.basename(p))


def _get_csv_headers(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path, nrows=0)
    return [str(c).lower().strip() for c in df.columns if str(c).strip()]


def _get_table_query(csv_path: str, table_path: Any, search_type: str) -> Optional[List[str]]:
    headers = _get_csv_headers(csv_path)
    if search_type == "single_column":
        df = pd.read_csv(csv_path, nrows=100)
        if len(df) > 0 and len(df.columns) > 0:
            return df[df.columns[0]].dropna().astype(str).tolist()
    return headers or [os.path.basename(str(table_path))]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Card2Tab2Card by type (standalone): query model tables → classify → search same-type only → model cards",
    )
    parser.add_argument("--model_id", required=True, help="Hugging Face model ID (seed model)")
    parser.add_argument("--relationship_parquet", default=DEFAULT_RELATIONSHIP_PARQUET, help="Model–table relationship parquet")
    parser.add_argument("--db_path", default="data/modellake.db", help="modellake.db path")
    parser.add_argument("--classification_json", default="data/table_classifications.json", help="Precomputed table classifications JSON")
    parser.add_argument("--search_type", choices=["keyword", "single_column", "multi_column", "unionable"], default="keyword")
    parser.add_argument("--k", type=int, default=5, help="Table search top-k per query table (then merge)")
    parser.add_argument("--modelcard_k", type=int, default=0, help="Max model cards to return (0 = no limit)")
    parser.add_argument("--max_query_tables", type=int, default=20, help="Max number of query tables to search (default 20)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for per-table search (1 = sequential, safe default)")
    parser.add_argument("--output_json", default="data/card2tab2card_by_type_standalone.json", help="Output JSON path")
    parser.add_argument("--no_citationlake", action="store_true", help="Ignored; kept for CLI compatibility")
    args = parser.parse_args()

    model_id = args.model_id
    relationship_parquet = args.relationship_parquet
    db_path = args.db_path
    classification_json = args.classification_json
    search_type = args.search_type
    table_search_k = args.k
    modelcard_k = args.modelcard_k
    max_query_tables = args.max_query_tables
    workers = max(1, args.workers)
    output_json = args.output_json

    if not os.path.exists(relationship_parquet):
        print(f"Error: relationship_parquet not found: {relationship_parquet}", file=sys.stderr)
        return 1
    if not os.path.exists(db_path):
        print(f"Error: db_path not found: {db_path}", file=sys.stderr)
        return 1
    if not os.path.exists(classification_json):
        print(f"Error: classification_json not found: {classification_json}", file=sys.stderr)
        return 1

    print(f"\n{'='*60}")
    print("Card2Tab2Card by type (standalone)")
    print(f"{'='*60}")
    print(f"  model_id={model_id}")
    print(f"  search_type={search_type}  table_search_k={table_search_k}  modelcard_k={modelcard_k}")
    print(f"  classification_json={classification_json}")
    print(f"{'='*60}\n")

    # Step 1: Query model → tables
    query_tables_raw = get_tables_for_model_duckdb(relationship_parquet, model_id)
    query_tables = _filter_s2orc_tables(query_tables_raw)
    query_tables = [t for t in query_tables if not _is_generic_table(t)]
    query_tables = query_tables[:max_query_tables]
    if not query_tables:
        print("No query tables after filtering. Exiting.")
        out = {
            "query_model": model_id,
            "query_tables": [],
            "searched_tables": [],
            "model_ids": [],
            "intermediate": {"retrieved_table_ids": [], "retrieved_table_filenames": [], "table_id_to_filename": {}, "table_to_models": {}},
        }
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return 0

    print(f"Step 1: {len(query_tables)} query tables (after filter, max {max_query_tables})")

    k_request = max(table_search_k + 4, table_search_k * 2)

    def search_one(ti: int, table_path: str) -> Optional[Tuple[List[int], str]]:
        csv_path = _resolve_csv_path(table_path)
        if not csv_path:
            return None
        tquery = _get_table_query(csv_path, table_path, search_type)
        if not tquery:
            return None
        t0 = time.time()
        ids = search_table2table_by_type(
            tquery,
            search_type,
            k_request,
            db_path=db_path,
            classification_json=classification_json,
        )
        elapsed = time.time() - t0
        bn = os.path.basename(str(table_path))
        print(f"   [Table {ti+1}] {bn}  k_request={k_request}  -> {len(ids or [])} ids  {elapsed:.1f}s")
        return (ids or [], bn)

    # Step 2: Per-table search (same type inside tab2tab_by_type)
    print(f"\nStep 2: Per-table search (by type), k_per_table={table_search_k}, k_request={k_request}, workers={workers}")
    results_by_ti: Dict[int, Tuple[List[int], str]] = {}
    if workers <= 1:
        for ti, tp in enumerate(query_tables):
            res = search_one(ti, tp)
            if res:
                results_by_ti[ti] = res
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(query_tables))) as ex:
            futures = {ex.submit(search_one, ti, tp): ti for ti, tp in enumerate(query_tables)}
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    ti = futures[fut]
                    results_by_ti[ti] = res
    per_table_results = [results_by_ti[ti] for ti in sorted(results_by_ti.keys())]

    if not per_table_results:
        print("No per-table results. Exiting.")
        out = {
            "query_model": model_id,
            "query_tables": query_tables,
            "searched_tables": [],
            "model_ids": [],
            "intermediate": {"retrieved_table_ids": [], "retrieved_table_filenames": [], "table_id_to_filename": {}, "table_to_models": {}},
        }
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return 0

    all_ids = [tid for ids, _ in per_table_results for tid in ids]
    tableid_to_filename: Dict[int, str] = {}
    if all_ids:
        with duckdb.connect(db_path, read_only=True) as con:
            table_ids_str = ",".join(str(tid) for tid in all_ids)
            rows = con.execute(f"""
                SELECT DISTINCT tableid, filename FROM modellake_index
                WHERE tableid IN ({table_ids_str}) AND rowid = -1
            """).fetchall()
            tableid_to_filename = {tid: filename for tid, filename in rows}

    n_self = n_s2orc = n_generic = 0
    per_table_filtered: List[List[Tuple[int, str]]] = []
    for ids, src_basename in per_table_results:
        kept: List[Tuple[int, str]] = []
        for tid in ids:
            if tid not in tableid_to_filename:
                continue
            fbase = os.path.basename(str(tableid_to_filename[tid]))
            if fbase == src_basename:
                n_self += 1
                continue
            if _classify_table_source_by_basename(fbase) == "llm":
                n_s2orc += 1
                continue
            if _is_generic_table(fbase):
                n_generic += 1
                continue
            kept.append((tid, src_basename))
        per_table_filtered.append(kept[:table_search_k])
    if n_self or n_s2orc or n_generic:
        print(f"   Filter: self={n_self} s2orc={n_s2orc} generic={n_generic}  top-k={table_search_k}/table")

    seen: set = set()
    similar_table_data: List[Tuple[int, str]] = []
    max_len = max(len(r) for r in per_table_filtered) if per_table_filtered else 0
    for rank in range(max_len):
        for row in per_table_filtered:
            if rank < len(row) and row[rank][0] not in seen:
                seen.add(row[rank][0])
                similar_table_data.append(row[rank])
    similar_table_ids = [tid for tid, _ in similar_table_data]
    print(f"   Merged: {len(similar_table_ids)} unique tables")

    # Re-fetch filenames for final list (in case we filtered)
    tableid_to_filename_final = {}
    if similar_table_ids:
        with duckdb.connect(db_path, read_only=True) as con:
            table_ids_str = ",".join(str(tid) for tid in similar_table_ids)
            rows = con.execute(f"""
                SELECT DISTINCT tableid, filename FROM modellake_index
                WHERE tableid IN ({table_ids_str}) AND rowid = -1
            """).fetchall()
            tableid_to_filename_final = {tid: filename for tid, filename in rows}

    seen_filenames = set()
    retrieved_filenames: List[str] = []
    for tid in similar_table_ids:
        if tid in tableid_to_filename_final:
            fname = tableid_to_filename_final[tid]
            if fname not in seen_filenames:
                seen_filenames.add(fname)
                retrieved_filenames.append(fname)
    retrieved_filenames = [f for f in retrieved_filenames if not _is_generic_table(f)]

    # Step 3: Tables → model cards
    print(f"\nStep 3: Map {len(retrieved_filenames)} tables to model cards")
    table_basenames = [os.path.basename(f) for f in retrieved_filenames]
    basename_to_models = get_modelids_for_basenames_duckdb(relationship_parquet, table_basenames)
    similar_model_ids = set()
    table_to_models: Dict[str, List[str]] = {}
    for filename in retrieved_filenames:
        basename = os.path.basename(filename)
        matched = basename_to_models.get(basename, [])
        if matched:
            similar_model_ids.update(matched)
            table_to_models[filename] = matched
    similar_model_ids = [mid for mid in similar_model_ids if mid != model_id]
    for fname in table_to_models:
        table_to_models[fname] = [mid for mid in table_to_models[fname] if mid != model_id]
    if modelcard_k and modelcard_k > 0:
        final_model_ids = similar_model_ids[:modelcard_k]
    else:
        final_model_ids = similar_model_ids

    print(f"   Matched {len(final_model_ids)} model cards (excluding query model)")

    table_id_to_filename_str = {str(tid): fname for tid, fname in tableid_to_filename_final.items()}
    result = {
        "query_model": model_id,
        "query_tables": query_tables,
        "searched_tables": retrieved_filenames,
        "model_ids": final_model_ids,
        "intermediate": {
            "retrieved_table_ids": similar_table_ids,
            "retrieved_table_filenames": retrieved_filenames,
            "table_id_to_filename": table_id_to_filename_str,
            "table_to_models": table_to_models,
        },
    }
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
