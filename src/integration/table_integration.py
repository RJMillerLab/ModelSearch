"""
Table Integration Implementation

Integrates multiple tables using various methods:
- Union: Combine all rows from all tables
- Intersection: Find common rows across all tables
- ALITE: FD-based integration using dialite_internal (requires dialite_internal repository)
- Outer Join: Merge all tables using outer join

Test: python -m tests.test_integration
"""

import os
import sys
import time
import re
import io
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
import json
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import Counter

from src.utils import resolve_table_path, load_table, _get_models_to_tables_batch_sql

def _reorder_columns_deterministic(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministically reorder columns for readability/comparability.

    Rules (table-only; no external reference):
    1) Columns that are not entirely null/empty first.
    2) Within those, higher non-null rate first.
    3) Columns that are entirely null/empty always go to the end.
    4) Stable tie-breaker: original column order.
    """
    if df is None or df.empty or len(df.columns) == 0:
        return df
    cols = list(df.columns)
    mask = df.notna() & (df != "")
    rate = mask.mean().values
    is_all_null = (rate == 0).astype(int)
    order = sorted(range(len(cols)), key=lambda i: (is_all_null[i], -rate[i], i))
    ordered_cols = [cols[i] for i in order]
    if ordered_cols != cols:
        print(
            "[reorder] columns changed (deterministic)\n"
            f"  before: {cols}\n"
            f"  after:  {ordered_cols}"
        )
        return df[ordered_cols]
    return df

def _integrate_tables_union(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Integrate multiple tables using Union operation.
    Returns the full integrated result (no row limit).
    
    Args:
        tables: List of DataFrames to integrate
        
    Returns:
        Integrated DataFrame or None if integration fails
    """
    if not tables or len(tables) == 0:
        return None
    
    if len(tables) == 1:
        return tables[0].copy()
    
    # Use pandas concat for union (combine all rows)
    all_columns: List[str] = []
    for df in tables:
        for col in df.columns:
            if col not in all_columns:
                all_columns.append(col)
    aligned_tables = [df.reindex(columns=all_columns) for df in tables]
    integrated = pd.concat(aligned_tables, axis=0, ignore_index=True)
    integrated = integrated.drop_duplicates()
    return integrated

def _integrate_tables_intersection(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Integrate multiple tables using Intersection operation (find common rows).
    Returns the full result (no row limit).
    
    Args:
        tables: List of DataFrames to integrate
        
    Returns:
        Integrated DataFrame with common rows or None if no common rows
    """
    if not tables or len(tables) == 0:
        return None
    
    if len(tables) == 1:
        return tables[0].copy()

    # Find common columns
    common_columns = set(tables[0].columns)
    for df in tables[1:]:
        common_columns = common_columns.intersection(set(df.columns))

    if not common_columns:
        print("⚠️  No common columns found for intersection")
        return pd.DataFrame()

    # Convert to string for comparison
    common_columns = list(common_columns)

    # Find intersection (rows that appear in all tables)
    result = tables[0][common_columns].copy()
    result['_temp_key'] = result.apply(lambda x: '|'.join(x.astype(str)), axis=1)

    for df in tables[1:]:
        df_subset = df[common_columns].copy()
        df_subset['_temp_key'] = df_subset.apply(lambda x: '|'.join(x.astype(str)), axis=1)
        result = result[result['_temp_key'].isin(df_subset['_temp_key'])]

    result = result[common_columns]
    return result

def _integrate_tables_alite(tables: List[pd.DataFrame], table_paths: List[str]) -> Optional[pd.DataFrame]:
    """
    Integrate tables using ALITE FD-based algorithm.
    Returns the full result (no row limit).
    
    Args:
        tables: List of DataFrames (not used directly, but kept for consistency)
        table_paths: List of paths to CSV files (required for ALITE)
        
    Returns:
        Integrated DataFrame or None if integration fails
    """
    from src.config import DIALITE_INTERNAL_REPO
    dialite_repo = DIALITE_INTERNAL_REPO
    if dialite_repo not in sys.path:
        sys.path.insert(0, dialite_repo)

    import alite.alite_fd as alite_module

    if not table_paths:
        print("⚠️  ALITE requires file paths, not DataFrames")
        return None

    alite_verbose = os.environ.get("ALITE_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}
    if alite_verbose:
        result_FD, stats_df, debug_dict = alite_module.FDAlgorithm(table_paths.copy())
    else:
        # ALITE emits many internal progress prints; suppress them by default.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result_FD, stats_df, debug_dict = alite_module.FDAlgorithm(table_paths.copy())
    if result_FD is not None and len(result_FD) > 0:
        return result_FD
    return result_FD


def _integrate_tables_outer_join(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Integrate tables using outer join (merge all tables on index).
    Returns the full result (no row limit).
    
    Args:
        tables: List of DataFrames to integrate
        
    Returns:
        Integrated DataFrame or None if integration fails
    """
    if not tables or len(tables) == 0:
        return None
    
    if len(tables) == 1:
        return tables[0].copy()
    
    # Start with first table; merge others with outer join on index
    result = tables[0].copy()
    for df in tables[1:]:
        result_reset = result.reset_index(drop=True)
        df_reset = df.reset_index(drop=True)
        result = pd.concat([result_reset, df_reset], axis=1, join='outer')
    return result


def _table_resources_for_integration(search_results: Dict[str, Any]) -> Optional[List[str]]:
    """
    Parquet column scope for modelId -> csv basenames (align with search pipeline --resources).

    - If ``table_resources`` is **absent** from JSON (legacy jobs): return ``None`` → all list columns
      in parquet (previous integration behavior).
    - If **present** (new backend always writes it): use that list; empty/invalid → config default.
    """
    from src.config import TABLE_RESOURCE_ALLOWLIST

    if "table_resources" not in search_results:
        print("ℹ️  Legacy search_results (no table_resources key): all parquet table-list columns")
        return None

    tr = search_results.get("table_resources")
    if isinstance(tr, list) and tr:
        out = [
            str(x).strip().lower()
            for x in tr
            if str(x).strip() and str(x).strip().lower() in ("hugging", "github", "arxiv", "llm")
        ]
        if out:
            return out
    fallback = [r for r in TABLE_RESOURCE_ALLOWLIST if r in ("hugging", "github", "arxiv", "llm")]
    print(
        "ℹ️  table_resources empty or invalid in search_results.json; "
        f"using TABLE_RESOURCE_ALLOWLIST={fallback!r} for parquet table lists"
    )
    return fallback


def _resolve_table_paths_for_model_ids(
    model_ids: List[str],
    *,
    resources: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Resolve unique local table paths for model IDs (only columns selected by ``resources``)."""
    model_to_tables = _get_models_to_tables_batch_sql(model_ids, resources=resources)
    seen_paths: Set[str] = set()
    table_paths: List[str] = []
    model_to_table_paths: Dict[str, List[str]] = {mid: [] for mid in model_ids}

    for mid in model_ids:
        for basename in model_to_tables.get(mid, []):
            resolved_path = resolve_table_path(basename)
            if not resolved_path:
                continue
            if resolved_path not in seen_paths:
                seen_paths.add(resolved_path)
                table_paths.append(resolved_path)
            model_to_table_paths[mid].append(resolved_path)

    return table_paths, model_to_table_paths


def integrate_tables(table_paths: List[str], integration_type: str = "union", k: int = 10, filename_to_tableid: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Integrate multiple tables from modellake.db (by tableid) or from CSV paths.
        
    Returns:
        Dictionary with integration results
    """
    print(f"\n{'='*60}")
    print(f"🔗 Table Integration")
    print(f"{'='*60}")
    print(f"Integration type: {integration_type}")
    print(f"Number of tables: {len(table_paths)}")
    print(f"Top K: {k}")
    
    tables = []
    loaded_paths = []
    
    for table_path in table_paths:
        basename = os.path.basename(table_path)
        # Safety: only load actual table CSVs.
        # If an upstream payload accidentally contains command/log strings, we should not treat them as CSV inputs.
        # Real table filenames are typically like `..._table1.csv` (no underscore after `table`)
        # but we also accept legacy `..._table_1.csv`.
        if not re.search(r"_table_?\d+\.csv$", basename, flags=re.IGNORECASE):
            print(f"⚠️  Skipping non-table input for integration: {basename}")
            continue
        tid = filename_to_tableid.get(basename) if filename_to_tableid else None
        resolved_table_path = resolve_table_path(table_path) or table_path
        # `table_paths` coming from card2tab2card outputs are often basenames.
        # Always load from the resolved absolute path when available.
        df = load_table(resolved_table_path)
        if df is not None:
            tables.append(df)
            loaded_paths.append(resolved_table_path)
            print(f"✅ Loaded: {os.path.basename(table_path)} ({len(df)} rows, {len(df.columns)} columns)")
        else:
            print(f"⚠️  Failed to load: {os.path.basename(table_path)}")
    
    if not tables:
        return {"success": False, "error": "No tables could be loaded", "integrated_table": None, "stats": {}}
    
    # Integrate tables (k = number of tables only; we never truncate output rows)
    if integration_type == "union":
        integrated_df = _integrate_tables_union(tables)
    elif integration_type == "intersection":
        integrated_df = _integrate_tables_intersection(tables)
    elif integration_type == "alite":
        # ALITE requires file paths, not DataFrames
        integrated_df = _integrate_tables_alite(tables, loaded_paths)
    elif integration_type == "outer_join":
        integrated_df = _integrate_tables_outer_join(tables)
    else:
        return {"success": False, "error": f"Unknown integration type: {integration_type}. Supported types: union, intersection, alite, outer_join", "integrated_table": None, "stats": {}}
    
    # if empty
    if integrated_df is None or (isinstance(integrated_df, pd.DataFrame) and len(integrated_df) == 0 and len(integrated_df.columns) == 0):
        if integration_type == "intersection":
            if tables:
                common_cols = set(tables[0].columns)
                for df in tables[1:]:
                    common_cols = common_cols.intersection(set(df.columns))
                empty_df = pd.DataFrame(columns=list(common_cols) if common_cols else [])
                return {"success": True, "integrated_table": empty_df, "stats": {}, "table_paths": loaded_paths}
        return {"success": False, "error": "Integration failed", "integrated_table": None, "stats": {}}

    # Calculate statistics
    integrated_df = _reorder_columns_deterministic(integrated_df)
    stats = {"input_tables": len(tables), "input_rows": sum(len(df) for df in tables), "output_rows": len(integrated_df), "output_columns": len(integrated_df.columns), "integration_type": integration_type}
    
    print(f"\n✅ Integration successful!")
    print(f"   Input: {stats['input_tables']} tables, {stats['input_rows']} total rows")
    print(f"   Output: {stats['output_rows']} rows, {stats['output_columns']} columns")
    print(f"{'='*60}\n")    
    return {"success": True, "integrated_table": integrated_df, "stats": stats, "table_paths": loaded_paths}

def integrate_tables_from_card2tab2card(
    search_results_json: str,
    search_type: str = "single_column",
    integration_type: str = "union",
    k: int = 10,
    tables_source: str = "intermediate",
    table_resources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Integrate tables from backend search_results.json Card2Tab2Card payloads."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"🔗 Table Integration from Search Results")
    print(f"{'='*60}")
    
    # Load search results
    if not os.path.exists(search_results_json):
        return {"success": False, "error": f"Search results file not found: {search_results_json}", "integrated_table": None}
    
    with open(search_results_json, 'r', encoding='utf-8') as f:
        search_results = json.load(f)

    parquet_resources: Optional[List[str]] = (
        table_resources
        if isinstance(table_resources, list) and table_resources
        else _table_resources_for_integration(search_results)
    )
    _scope = "ALL_COLUMNS" if parquet_resources is None else repr(parquet_resources)
    print(f"ℹ️  Parquet table-list columns scoped to resources={_scope} (hugging/github/arxiv/llm)")

    # Only support the current backend schema:
    # search_results["card2tab2card_results"][search_type]
    card2tab2card_results = search_results.get("card2tab2card_results")
    if not isinstance(card2tab2card_results, dict):
        return {"success": False, "error": "search_results.json must contain card2tab2card_results", "integrated_table": None}

    search_payload = card2tab2card_results.get(search_type)
    if not isinstance(search_payload, dict):
        available_types = list(card2tab2card_results.keys())
        return {"success": False, "error": f"Search type '{search_type}' not found in card2tab2card_results. Available types: {', '.join(available_types)}", "integrated_table": None}

    intermediate = search_payload.get("intermediate")
    if not isinstance(intermediate, dict):
        return {"success": False, "error": f"Search type '{search_type}' has no intermediate payload", "integrated_table": None}

    # Current card2tab2card payload stores the final table list in
    # `searched_tables` and duplicates it in `intermediate.retrieved_table_filenames`.
    table_to_models = intermediate.get("table_to_models", {})
    table_id_to_filename = intermediate.get("table_id_to_filename", {})

    retrieved_filenames = search_payload.get("searched_tables", []) if isinstance(search_payload, dict) else []
    if retrieved_filenames:
        print(f"ℹ️  Using searched_tables: {len(retrieved_filenames)} tables")
    else:
        retrieved_filenames = intermediate.get("retrieved_table_filenames", [])
        if retrieved_filenames:
            print(f"ℹ️  Using intermediate.retrieved_table_filenames: {len(retrieved_filenames)} tables")
    if not retrieved_filenames:
        return {"success": False, "error": "No retrieved tables found in card2tab2card payload (expected searched_tables or intermediate.retrieved_table_filenames)", "integrated_table": None}
    
    models_with_tables_list = []
    if tables_source == "all_from_modelcards":
        # Get model_ids from table_to_models, then DuckDB batch query
        model_ids = set()
        table_to_models = intermediate.get("table_to_models", {})
        for table_path, model_list in table_to_models.items():
            for m in (model_list if isinstance(model_list, list) else []):
                mid = m.get("model_id") or m.get("modelId") if isinstance(m, dict) else str(m)
                if mid:
                    model_ids.add(mid)
        # No model cap: use ALL models from table_to_models (was: model_ids[:50])
        model_ids = list(model_ids)
        if not model_ids:
            return {"success": False, "error": "No model IDs in intermediate for all_from_modelcards", "integrated_table": None}
        print(f"✅ tables_source=all_from_modelcards: DuckDB batch query for {len(model_ids)} models")
        table_paths, model_to_table_paths_ts = _resolve_table_paths_for_model_ids(model_ids, resources=parquet_resources)
        # No table top-k cap: use ALL tables
        print(f"   Resolved {len(table_paths)} tables for integration (no table top-k cap; from {len(retrieved_filenames)} retrieved tables)")
        models_with_tables_list = model_ids
    else:
        # intermediate: use retrieved tables (current behavior)
        # No table top-k cap: use ALL retrieved tables (per-table k already applied in card2tab2card)
        print(f"✅ tables_source=intermediate: {len(retrieved_filenames)} retrieved tables from search (no table top-k cap)")
        table_paths = retrieved_filenames
        table_to_models = intermediate.get("table_to_models", {})
        # Align with card2tab2card model_ids (50) when available - same set as retrieval results
        c2t2c_model_ids = list(search_payload["model_ids"]) if isinstance(search_payload.get("model_ids"), (list, tuple)) else []
        if c2t2c_model_ids:
            models_with_tables_list = c2t2c_model_ids
            print(f"   Using card2tab2card model_ids ({len(c2t2c_model_ids)}) for alignment with retrieval results")
        else:
            basename_to_key = {os.path.basename(key): key for key in table_to_models}
            model_ids_set = set()
            for tp in table_paths:
                model_list = table_to_models.get(tp) or table_to_models.get(basename_to_key.get(os.path.basename(tp)))
                for m in (model_list or []):
                    mid = m.get("model_id") or m.get("modelId") if isinstance(m, dict) else str(m)
                    if mid:
                        model_ids_set.add(mid)
            models_with_tables_list = list(model_ids_set)
    
    if not table_paths:
        return {"success": False, "error": "No tables to integrate", "integrated_table": None}
    
    # Build model_id -> table_paths for UI debug (intermediate path: reverse of table_to_models)
    if tables_source != "all_from_modelcards":
        model_to_table_paths_ts = {}
        basename_to_key = {os.path.basename(key): key for key in table_to_models}
        for tp in table_paths:
            key = tp if tp in table_to_models else basename_to_key.get(os.path.basename(tp))
            model_list = table_to_models.get(key, []) if key else []
            for m in (model_list or []):
                mid = m.get("model_id") or m.get("modelId") if isinstance(m, dict) else str(m)
                if mid:
                    model_to_table_paths_ts.setdefault(mid, []).append(tp)
    # Build filename -> tableid
    filename_to_tableid: Dict[str, int] = {}
    if table_id_to_filename:
        for tid, fname in table_id_to_filename.items():
            bn = os.path.basename(str(fname))
            if bn:
                filename_to_tableid[bn] = int(tid) if isinstance(tid, (int, float)) else tid
    print(f"📊 Integration (Table Search): #tables={len(table_paths)}, #models={len(models_with_tables_list)}")
    result = integrate_tables(table_paths, integration_type, k, filename_to_tableid=filename_to_tableid or None)
    result["models_with_tables"] = models_with_tables_list
    result["model_to_table_paths"] = model_to_table_paths_ts
    elapsed = time.time() - t0
    # Attach timing + source info to stats for debugging
    if not isinstance(result.get("stats"), dict):
        result["stats"] = result.get("stats") or {}
    result["stats"]["elapsed_seconds"] = elapsed
    result["stats"]["tables_source"] = tables_source
    result["stats"]["parquet_table_resources"] = parquet_resources
    result["stats"]["total_unique_tables"] = len(table_paths)
    print(f"⏱️  Table Search integration elapsed: {elapsed:.2f}s (tables_source={tables_source})")
    return result

def integrate_tables_from_card2card(
    search_results_json: str,
    integration_type: str = "union",
    k: int = 10,
    max_models: int = 10,
    card2card_retrieval_mode: Optional[str] = None,
    table_resources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Integrate tables from backend search_results.json Card2Card payloads."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"🔗 Table Integration from Model Search Results")
    print(f"{'='*60}")
    
    # Load search results
    if not os.path.exists(search_results_json):
        return {"success": False, "error": f"Search results file not found: {search_results_json}", "integrated_table": None}
    
    with open(search_results_json, 'r', encoding='utf-8') as f:
        search_results = json.load(f)

    parquet_resources: Optional[List[str]] = (
        table_resources
        if isinstance(table_resources, list) and table_resources
        else _table_resources_for_integration(search_results)
    )
    _scope_c2c = "ALL_COLUMNS" if parquet_resources is None else repr(parquet_resources)
    print(f"ℹ️  Parquet table-list columns scoped to resources={_scope_c2c} (hugging/github/arxiv/llm)")
    
    # Only support the current backend schema:
    # search_results["card2card_all_modes"][mode] or search_results["card2card_results"]
    if not isinstance(search_results, dict):
        return {"success": False, "error": "search_results.json must be a backend Card2Card payload object", "integrated_table": None}

    if card2card_retrieval_mode:
        card2card_all_modes = search_results.get("card2card_all_modes")
        if not isinstance(card2card_all_modes, dict):
            return {"success": False, "error": "search_results.json must contain card2card_all_modes when card2card_retrieval_mode is specified", "integrated_table": None}
        card2card_results = card2card_all_modes.get(card2card_retrieval_mode)
        if isinstance(card2card_results, dict) and card2card_results.get("error"):
            return {"success": False, "error": f"Card2Card mode '{card2card_retrieval_mode}' failed: {card2card_results.get('error')}", "integrated_table": None}
        if not isinstance(card2card_results, list):
            return {"success": False, "error": f"Card2Card mode '{card2card_retrieval_mode}' is missing or not a list", "integrated_table": None}
    else:
        card2card_results = search_results.get("card2card_results")
        if not isinstance(card2card_results, list):
            return {"success": False, "error": "search_results.json must contain card2card_results as a list", "integrated_table": None}

    model_ids = []
    for item in card2card_results:
        if isinstance(item, str):
            model_ids.append(item)
        elif isinstance(item, dict) and "modelId" in item:
            model_ids.append(item["modelId"])
        elif isinstance(item, dict) and "model_id" in item:
            model_ids.append(item["model_id"])
    if not model_ids:
        return {"success": False, "error": "No model IDs found in Card2Card results", "integrated_table": None}
    
    print(f"✅ Found {len(model_ids)} models in Card2Card results")
    print(f"   Processing {len(model_ids)} models from Card2Card")

    # Collect all table paths from all models; also build model_id -> table_paths for UI debug
    # Fast path: DuckDB batch query (single SQL scan, no full parquet load)
    t_duck = time.time()
    print(f"✅ DuckDB batch query on parquet (fast, no full load)")
    all_table_paths, model_to_table_paths = _resolve_table_paths_for_model_ids(model_ids, resources=parquet_resources)
    models_with_tables = [mid for mid, paths in model_to_table_paths.items() if paths]
    models_without_tables = [mid for mid, paths in model_to_table_paths.items() if not paths]
    for model_id in models_with_tables:
        print(f"  ✅ Model {model_id}: {len(model_to_table_paths[model_id])} tables")
    for model_id in models_without_tables:
        print(f"  ⚠️  Model {model_id}: No tables found")
    print(f"   DuckDB query elapsed: {time.time() - t_duck:.2f}s")
    
    if not all_table_paths:
        return {
            "success": False,
            "error": f"No tables found for any of the {len(model_ids)} models",
            "integrated_table": None,
            "stats": {
                "models_processed": len(model_ids),
                "models_with_tables": len(models_with_tables),
                "models_without_tables": len(models_without_tables),
                "parquet_table_resources": parquet_resources,
            },
        }
    
    print(f"\n✅ Collected {len(all_table_paths)} unique tables from {len(models_with_tables)} models")

    # No table top-k cap: use ALL tables
    table_paths = all_table_paths
    print(f"📊 Integration (Model Search): #tables={len(table_paths)}, #models={len(models_with_tables)} (no table top-k cap)")
    # Integrate tables
    result = integrate_tables(table_paths, integration_type, k)
    
    # Add model search specific stats
    if result.get("success"):
        if not isinstance(result.get("stats"), dict):
            result["stats"] = result.get("stats") or {}
        result["stats"]["models_processed"] = len(model_ids)
        result["stats"]["models_with_tables"] = len(models_with_tables)
        result["stats"]["models_without_tables"] = len(models_without_tables)
        result["stats"]["total_unique_tables"] = len(all_table_paths)
        result["model_ids"] = model_ids
        result["models_with_tables"] = models_with_tables
        result["models_without_tables"] = models_without_tables
        # model_id -> list of table paths for UI debug display
        result["model_to_table_paths"] = model_to_table_paths
    
    elapsed = time.time() - t0
    # Attach timing + DuckDB info to stats for debugging
    if isinstance(result.get("stats"), dict):
        result["stats"]["elapsed_seconds"] = elapsed
        result["stats"]["parquet_table_resources"] = parquet_resources
    print(f"⏱️  Model Search integration elapsed: {elapsed:.2f}s")
    return result
