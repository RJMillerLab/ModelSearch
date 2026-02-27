"""
Table Integration Implementation

Integrates multiple tables using various methods:
- Union: Combine all rows from all tables
- Intersection: Find common rows across all tables
- ALITE: FD-based integration using dialite_internal (requires dialite_internal repository)
- Outer Join: Merge all tables using outer join

Works with pre-searched table results to avoid re-searching.
"""

import os
import sys
import time
import pandas as pd
import json
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import Counter

from utils.table_loader import load_table, resolve_table_path


def _normalize_val_to_items(val: Any) -> List[Any]:
    """Unify iteration over parquet list cells: handle list, tuple, np.ndarray, scalar."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, (list, tuple)):
        return val
    if hasattr(val, "__iter__") and not isinstance(val, str):
        return list(val)
    return [val]


def _get_tables_for_models_duckdb(
    parquet_path: str, model_ids: List[str]
) -> Dict[str, List[str]]:
    """
    Fast batch query: get all (modelId, csv_basename) for given models using DuckDB.
    Single SQL scan instead of per-model iteration.
    """
    import duckdb
    if not model_ids or not os.path.exists(parquet_path):
        return {}
    path_abs = os.path.abspath(parquet_path).replace("\\", "/")
    conn = duckdb.connect(":memory:")
    ids_sql = ",".join(repr(m) for m in model_ids)
    df = conn.execute(f"""
        SELECT modelId FROM read_parquet('{path_abs}')
        WHERE modelId IN ({ids_sql})
    """).fetchdf()
    if df.empty:
        conn.close()
        return {}
    cols = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{path_abs}')").fetchall()
    list_cols = [
        c[0] for c in cols
        if c[0] != "modelId" and ("csv" in c[0].lower() or "table_list" in c[0].lower())
    ][:4]
    if not list_cols:
        conn.close()
        return {}
    select_cols = "modelId, " + ", ".join(f'"{c}"' for c in list_cols)
    full_df = conn.execute(f"""
        SELECT {select_cols} FROM read_parquet('{path_abs}')
        WHERE modelId IN ({ids_sql})
    """).fetchdf()
    conn.close()
    model_to_tables: Dict[str, List[str]] = {m: [] for m in model_ids}
    for _, row in full_df.iterrows():
        mid = str(row["modelId"])
        for col in list_cols:
            val = row.get(col)
            items = _normalize_val_to_items(val)
            for v in items:
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    base = os.path.basename(s)
                    if base and base not in model_to_tables[mid]:
                        model_to_tables[mid].append(base)
    return model_to_tables

# Add Blend_internal to path
blend_path = os.path.join(os.path.dirname(__file__), '..', 'Blend_internal')
blend_path_abs = os.path.abspath(blend_path)
if blend_path_abs and os.path.exists(blend_path_abs):
    if blend_path_abs in sys.path:
        sys.path.remove(blend_path_abs)
    sys.path.insert(0, blend_path_abs)

# Blend_internal imports are optional and will be lazy-loaded if needed
# We don't import them at module level to avoid DB initialization issues
BLEND_AVAILABLE = bool(blend_path_abs and os.path.exists(blend_path_abs))


def integrate_tables_union(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
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


def integrate_tables_intersection(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
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


def _find_dialite_internal() -> Optional[str]:
    """Find dialite_internal repository directory - checks local others/ first, then external repos"""
    # Priority 1: Check local others/dialite directory (bundled with this repo)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    local_dialite = os.path.join(project_root, 'others', 'dialite')
    if os.path.exists(os.path.join(local_dialite, 'alite', 'alite_fd.py')):
        return local_dialite
    
    # Priority 2: Check environment variable
    if 'DIALITE_INTERNAL_REPO' in os.environ:
        repo_dir = os.environ['DIALITE_INTERNAL_REPO']
        if os.path.exists(os.path.join(repo_dir, 'alite', 'alite_fd.py')):
            return repo_dir
    
    # Priority 3: Check common external locations
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '../../..', 'dialite_internal'),
        os.path.join(os.path.dirname(__file__), '../..', 'dialite_internal'),
        '/Users/doradong/Repo/dialite_internal',
        os.path.join(os.path.expanduser('~'), 'Repo', 'dialite_internal'),
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(os.path.join(abs_path, 'alite', 'alite_fd.py')):
            return abs_path
    
    return None


def integrate_tables_alite(
    tables: List[pd.DataFrame],
    table_paths: List[str],
) -> Optional[pd.DataFrame]:
    """
    Integrate tables using ALITE FD-based algorithm.
    Returns the full result (no row limit).
    
    Args:
        tables: List of DataFrames (not used directly, but kept for consistency)
        table_paths: List of paths to CSV files (required for ALITE)
        
    Returns:
        Integrated DataFrame or None if integration fails
    """
    dialite_repo = _find_dialite_internal()
    if dialite_repo is None:
        print("⚠️  dialite_internal not found. ALITE integration requires dialite_internal repository.")
        print("   Set DIALITE_INTERNAL_REPO environment variable or ensure dialite_internal is in a standard location.")
        return None
    
    # Add dialite_internal to path
    if dialite_repo not in sys.path:
        sys.path.insert(0, dialite_repo)

    import alite.alite_fd as alite_module

    if not table_paths:
        print("⚠️  ALITE requires file paths, not DataFrames")
        return None

    result_FD, stats_df, debug_dict = alite_module.FDAlgorithm(table_paths.copy())
    if result_FD is not None and len(result_FD) > 0:
        return result_FD
    return result_FD


def integrate_tables_outer_join(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
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


def integrate_tables(
    table_paths: List[str],
    integration_type: str = "union",
    k: int = 10,
    db_path: Optional[str] = None,
    filename_to_tableid: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Integrate multiple tables from modellake.db (by tableid) or from CSV paths.
    
    Args:
        table_paths: List of paths or filenames (basename) to tables
        integration_type: "union", "intersection", "alite", or "outer_join"
        k: Number of tables to integrate (top k); the integrated result is not truncated by row count
        db_path: Optional path to modellake.db; when set with filename_to_tableid, load from DB first (no CSV)
        filename_to_tableid: Optional map basename -> tableid for loading from modellake_index
        
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
    use_db = bool(db_path and filename_to_tableid)
    
    for table_path in table_paths:
        basename = os.path.basename(table_path)
        tid = filename_to_tableid.get(basename) if filename_to_tableid else None
        df = load_table(table_path, db_path=db_path, tableid=tid)
        if df is not None:
            tables.append(df)
            loaded_paths.append(table_path)
            print(f"✅ Loaded: {os.path.basename(table_path)} ({len(df)} rows, {len(df.columns)} columns)")
        else:
            print(f"⚠️  Failed to load: {os.path.basename(table_path)}")
    
    if not tables:
        return {
            "success": False,
            "error": "No tables could be loaded",
            "integrated_table": None,
            "stats": {}
        }
    
    # Integrate tables (k = number of tables only; we never truncate output rows)
    if integration_type == "union":
        integrated_df = integrate_tables_union(tables)
    elif integration_type == "intersection":
        integrated_df = integrate_tables_intersection(tables)
    elif integration_type == "alite":
        # ALITE requires file paths, not DataFrames
        integrated_df = integrate_tables_alite(tables, loaded_paths)
    elif integration_type == "outer_join":
        integrated_df = integrate_tables_outer_join(tables)
    else:
        return {
            "success": False,
            "error": f"Unknown integration type: {integration_type}. Supported types: union, intersection, alite, outer_join",
            "integrated_table": None,
            "stats": {}
        }
    
    # Handle None or empty DataFrame results
    if integrated_df is None or (isinstance(integrated_df, pd.DataFrame) and len(integrated_df) == 0 and len(integrated_df.columns) == 0):
        # Check if it's intersection with no common rows/columns (this is valid, return empty result)
        if integration_type == "intersection":
            # Create empty DataFrame
            if tables:
                # Try to get common columns
                common_cols = set(tables[0].columns)
                for df in tables[1:]:
                    common_cols = common_cols.intersection(set(df.columns))
                
                # Return empty DataFrame (with or without common columns)
                empty_df = pd.DataFrame(columns=list(common_cols) if common_cols else [])
                return {
                    "success": True,
                    "integrated_table": empty_df,
                    "stats": {
                        "input_tables": len(tables),
                        "input_rows": sum(len(df) for df in tables),
                        "output_rows": 0,
                        "output_columns": len(common_cols) if common_cols else 0,
                        "integration_type": integration_type
                    },
                    "table_paths": loaded_paths
                }
        
        # Fallback: chosen method failed (ALITE/dialite missing, outer_join error, etc).
        # Retry with union so user gets a result instead of "Integration failed".
        if integration_type != "union":
            fallback_df = integrate_tables_union(tables)
            if fallback_df is not None and (len(fallback_df) > 0 or len(fallback_df.columns) > 0):
                stats = {
                    "input_tables": len(tables),
                    "input_rows": sum(len(df) for df in tables),
                    "output_rows": len(fallback_df),
                    "output_columns": len(fallback_df.columns),
                    "integration_type": "union"
                }
                print(f"⚠️  {integration_type} returned no result; used union as fallback")
                return {
                    "success": True,
                    "integrated_table": fallback_df,
                    "stats": stats,
                    "table_paths": loaded_paths,
                    "fallback": True,
                    "fallback_reason": f"{integration_type} did not produce a result (tables may not meet its requirements); used union instead"
                }
        
        return {
            "success": False,
            "error": "Integration failed",
            "integrated_table": None,
            "stats": {}
        }
    
    # Handle empty DataFrame with columns (valid result for intersection with no common rows)
    if isinstance(integrated_df, pd.DataFrame) and len(integrated_df) == 0 and len(integrated_df.columns) > 0:
        # This is a valid empty result (e.g., intersection with common columns but no common rows)
        stats = {
            "input_tables": len(tables),
            "input_rows": sum(len(df) for df in tables),
            "output_rows": 0,
            "output_columns": len(integrated_df.columns),
            "integration_type": integration_type
        }
        return {
            "success": True,
            "integrated_table": integrated_df,
            "stats": stats,
            "table_paths": loaded_paths
        }
    
    # Calculate statistics
    stats = {
        "input_tables": len(tables),
        "input_rows": sum(len(df) for df in tables),
        "output_rows": len(integrated_df),
        "output_columns": len(integrated_df.columns),
        "integration_type": integration_type
    }
    
    print(f"\n✅ Integration successful!")
    print(f"   Input: {stats['input_tables']} tables, {stats['input_rows']} total rows")
    print(f"   Output: {stats['output_rows']} rows, {stats['output_columns']} columns")
    print(f"{'='*60}\n")
    
    return {
        "success": True,
        "integrated_table": integrated_df,
        "stats": stats,
        "table_paths": loaded_paths
    }


def integrate_tables_from_search_results(
    search_results_json: str,
    search_type: str = "single_column",
    integration_type: str = "union",
    k: int = 10,
    db_path: Optional[str] = None,
    tables_source: str = "intermediate",
    relationship_parquet: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Integrate tables from Card2Tab2Card search results.
    
    Args:
        tables_source: "intermediate" = use retrieved tables from search (fast).
                       "all_from_modelcards" = get ALL tables for found models from parquet (DuckDB).
        relationship_parquet: Required when tables_source="all_from_modelcards".
        
    Returns:
        Dictionary with integration results
    """
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"🔗 Table Integration from Search Results")
    print(f"{'='*60}")
    
    # Load search results
    if not os.path.exists(search_results_json):
        return {
            "success": False,
            "error": f"Search results file not found: {search_results_json}",
            "integrated_table": None
        }
    
    with open(search_results_json, 'r', encoding='utf-8') as f:
        search_results = json.load(f)
    
    # Extract table filenames from intermediate results
    # Handle both old format (list) and new format (dict with model_ids and intermediate)
    if isinstance(search_results, dict):
        if "intermediate" in search_results:
            # New format with intermediate data
            intermediate = search_results["intermediate"]
        elif "card2tab2card_results" in search_results:
            # Results from backend API
            card2tab2card_results = search_results["card2tab2card_results"]
            if isinstance(card2tab2card_results, dict):
                if search_type in card2tab2card_results:
                    if isinstance(card2tab2card_results[search_type], dict) and "intermediate" in card2tab2card_results[search_type]:
                        intermediate = card2tab2card_results[search_type]["intermediate"]
                    else:
                        return {
                            "success": False,
                            "error": f"Search type '{search_type}' found but no intermediate data available",
                            "integrated_table": None
                        }
                else:
                    available_types = list(card2tab2card_results.keys())
                    return {
                        "success": False,
                        "error": f"Search type '{search_type}' not found in results. Available types: {', '.join(available_types)}",
                        "integrated_table": None
                    }
            else:
                return {
                    "success": False,
                    "error": "Invalid search results format",
                    "integrated_table": None
                }
        else:
            return {
                "success": False,
                "error": "No intermediate results found in search results",
                "integrated_table": None
            }
    else:
        return {
            "success": False,
            "error": "Invalid search results format",
            "integrated_table": None
        }
    
    # Get retrieved table filenames (prefer Tab2Tab relevance order)
    # Priority: retrieved_table_ids + table_id_to_filename > retrieved_table_filenames > table_to_models keys
    table_to_models = intermediate.get("table_to_models", {})
    retrieved_table_ids = intermediate.get("retrieved_table_ids", [])
    table_id_to_filename = intermediate.get("table_id_to_filename", {})
    
    retrieved_filenames = []
    if retrieved_table_ids and table_id_to_filename:
        # Rebuild ordered list from Tab2Tab relevance (retrieved_table_ids order), dedupe by filename
        seen_files = set()
        for tid in retrieved_table_ids:
            if tid in table_id_to_filename:
                f = table_id_to_filename[tid]
                if f not in seen_files:
                    seen_files.add(f)
                    retrieved_filenames.append(f)
        if retrieved_filenames:
            print(f"ℹ️  Using retrieved_table_ids order: {len(retrieved_filenames)} tables (Tab2Tab relevance)")
    
    if not retrieved_filenames:
        retrieved_filenames = intermediate.get("retrieved_table_filenames", [])
    
    if not retrieved_filenames:
        if table_to_models:
            retrieved_filenames = list(table_to_models.keys())
            print(f"ℹ️  Using table paths from table_to_models: {len(retrieved_filenames)} tables")
        else:
            return {
                "success": False,
                "error": "No retrieved tables found in search results (no retrieved_table_filenames or table_to_models)",
                "integrated_table": None
            }
    
    if not retrieved_filenames:
        return {
            "success": False,
            "error": "No retrieved tables found in search results",
            "integrated_table": None
        }
    
    models_with_tables_list = []
    if tables_source == "all_from_modelcards" and relationship_parquet and os.path.exists(relationship_parquet):
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
        model_to_tables = _get_tables_for_models_duckdb(relationship_parquet, model_ids)
        seen = set()
        table_paths = []
        model_to_table_paths_ts = {mid: [] for mid in model_ids}
        for mid in model_ids:
            for bn in model_to_tables.get(mid, []):
                if bn in seen:
                    continue
                seen.add(bn)
                p = resolve_table_path(bn)
                if p:
                    table_paths.append(p)
                    model_to_table_paths_ts[mid].append(p)
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
        c2t2c_model_ids = []
        if "card2tab2card_results" in search_results and search_type in search_results.get("card2tab2card_results", {}):
            stub = search_results["card2tab2card_results"][search_type]
            if isinstance(stub, dict) and "model_ids" in stub:
                c2t2c_model_ids = list(stub["model_ids"]) if isinstance(stub["model_ids"], (list, tuple)) else []
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
    # all_from_modelcards: model_to_table_paths_ts already built above
    
    # Build filename -> tableid so we can try loading from modellake.db first (no CSV dirs needed)
    filename_to_tableid: Dict[str, int] = {}
    if table_id_to_filename:
        for tid, fname in table_id_to_filename.items():
            bn = os.path.basename(str(fname))
            if bn:
                filename_to_tableid[bn] = int(tid) if isinstance(tid, (int, float)) else tid
    print(f"📊 Integration (Table Search): #tables={len(table_paths)}, #models={len(models_with_tables_list)}")
    result = integrate_tables(table_paths, integration_type, k, db_path=db_path, filename_to_tableid=filename_to_tableid or None)
    result["models_with_tables"] = models_with_tables_list
    result["model_to_table_paths"] = model_to_table_paths_ts
    elapsed = time.time() - t0
    # Attach timing + source info to stats for debugging
    if not isinstance(result.get("stats"), dict):
        result["stats"] = result.get("stats") or {}
    result["stats"]["elapsed_seconds"] = elapsed
    result["stats"]["tables_source"] = tables_source
    result["stats"]["total_unique_tables"] = len(table_paths)
    print(f"⏱️  Table Search integration elapsed: {elapsed:.2f}s (tables_source={tables_source})")
    return result


def integrate_tables_from_model_search_results(
    search_results_json: str,
    integration_type: str = "union",
    k: int = 10,
    max_models: int = 10,
    db_path: Optional[str] = None,
    relationship_parquet: Optional[str] = None,
    schema_log_path: str = "data_citationlake/logs/parquet_schema.log",
    use_citationlake: bool = True,
    card2card_retrieval_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Integrate tables from Card2Card (model search) results.
    
    Args:
        search_results_json: Path to JSON file with search results
        integration_type: "union", "intersection", "alite", or "outer_join"
        k: Maximum number of rows in result
        max_models: Maximum number of models to process (default: 10)
        db_path: Optional path to modellake.db (for Blend_internal integration)
        relationship_parquet: Optional path to relationship parquet file
        schema_log_path: Path to parquet_schema.log (for get_from-style approach, e.g., ModelTables)
        use_citationlake: Whether to use get_from-style mapping (default: True)
        card2card_retrieval_mode: Optional retrieval mode: "dense", "sparse", or "hybrid"
        
    Returns:
        Dictionary with integration results
    """
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"🔗 Table Integration from Model Search Results")
    print(f"{'='*60}")
    
    # Load search results
    if not os.path.exists(search_results_json):
        return {
            "success": False,
            "error": f"Search results file not found: {search_results_json}",
            "integrated_table": None
        }
    
    with open(search_results_json, 'r', encoding='utf-8') as f:
        search_results = json.load(f)
    
    # Extract Card2Card model IDs (optionally from a specific retrieval mode: dense/sparse/hybrid)
    card2card_results = []
    if isinstance(search_results, dict):
        if card2card_retrieval_mode and search_results.get("card2card_all_modes"):
            mode_data = search_results["card2card_all_modes"].get(card2card_retrieval_mode)
            if isinstance(mode_data, list):
                card2card_results = mode_data
            elif isinstance(mode_data, dict) and "error" not in mode_data:
                card2card_results = []
            # else keep []
        if not card2card_results and "card2card_results" in search_results:
            card2card_results = search_results["card2card_results"]
        elif not card2card_results and "results" in search_results and "card2card_results" in search_results["results"]:
            card2card_results = search_results["results"]["card2card_results"]
    elif isinstance(search_results, list):
        card2card_results = search_results
    
    # Handle both string and object formats
    model_ids = []
    for item in card2card_results:
        if isinstance(item, str):
            model_ids.append(item)
        elif isinstance(item, dict) and "model_id" in item:
            model_ids.append(item["model_id"])
        elif isinstance(item, dict) and "modelId" in item:
            model_ids.append(item["modelId"])
    
    if not model_ids:
        return {
            "success": False,
            "error": "No model IDs found in Card2Card results",
            "integrated_table": None
        }
    
    print(f"✅ Found {len(model_ids)} models in Card2Card results")
    print(f"   Processing first {min(max_models, len(model_ids))} models (max_models cap COMMENTED OUT - using all models)")

    # Limit to max_models -- COMMENTED OUT: use all models for now (user request)
    # model_ids = model_ids[:max_models]
    
    # Import get_tables_for_model and load_relationship_parquet
    # Import directly from card2tab2card to avoid triggering card2card imports that require torch
    # Add parent directory to path if needed
    import sys
    parent_dir = os.path.join(os.path.dirname(__file__), '../..')
    parent_dir_abs = os.path.abspath(parent_dir)
    if parent_dir_abs not in sys.path:
        sys.path.insert(0, parent_dir_abs)
    
    from src.search.card2tab2card import get_tables_for_model, load_relationship_parquet
    
    # Load relationship data only when DuckDB path is unavailable and we truly need a full DataFrame
    relationship_df = None
    if not use_citationlake:
        # If we don't have a usable parquet path for DuckDB, fall back to loading via pandas
        parquet_for_df = None
        if relationship_parquet and os.path.exists(relationship_parquet):
            # DuckDB branch below will handle this path directly; avoid full load here
            parquet_for_df = None
        else:
            default_parquet = "data_citationlake/processed/modelcard_step3_dedup.parquet"
            if os.path.exists(default_parquet):
                parquet_for_df = default_parquet
        if parquet_for_df:
            relationship_df = load_relationship_parquet(parquet_for_df)
            print(f"✅ Loaded relationship parquet for fallback DataFrame path: {parquet_for_df}")

    # Table–model mapping uses relationship_parquet only (no optional get_from)
    citationlake_available = False
    
    # Fallback: Try to extract table information from search_results JSON
    # This is useful for template/fake data when real data sources are unavailable
    fallback_table_paths = []
    fallback_models_with_tables = {}
    
    # Priority 1: Check for dedicated card2card_integration_tables field (separate from table search)
    if isinstance(search_results, dict) and "card2card_integration_tables" in search_results:
        card2card_integration = search_results["card2card_integration_tables"]
        print(f"✅ Found dedicated card2card_integration_tables field in search results")
        
        # Get table paths
        if "table_paths" in card2card_integration:
            fallback_table_paths = card2card_integration["table_paths"]
            print(f"   Found {len(fallback_table_paths)} table paths")
        
        # Get model_to_tables mapping
        if "model_to_tables" in card2card_integration:
            model_to_tables = card2card_integration["model_to_tables"]
            # Map our model_ids to tables
            for model_id in model_ids:
                if model_id in model_to_tables:
                    tables = model_to_tables[model_id]
                    if isinstance(tables, list):
                        fallback_models_with_tables[model_id] = tables
                        # Add to fallback_table_paths if not already there
                        for table_path in tables:
                            if table_path not in fallback_table_paths:
                                fallback_table_paths.append(table_path)
                    print(f"   Model {model_id}: {len(tables)} tables")
    
    # Priority 2: card2tab2card_results intermediate (tables from related-search)
    elif isinstance(search_results, dict) and "card2tab2card_results" in search_results:
        print(f"   Using card2tab2card_results intermediate (table_to_models)")
        card2tab2card_results = search_results["card2tab2card_results"]
        # Extract table_to_models mapping from any search type
        for search_type, type_results in card2tab2card_results.items():
            if isinstance(type_results, dict) and "intermediate" in type_results:
                intermediate = type_results["intermediate"]
                table_to_models = intermediate.get("table_to_models", {})
                # Build reverse mapping: model_id -> list of tables
                for table_path, model_list in table_to_models.items():
                    # Normalize model list
                    normalized_models = []
                    for m in (model_list if isinstance(model_list, list) else []):
                        if isinstance(m, str):
                            normalized_models.append(m)
                        elif isinstance(m, dict):
                            normalized_models.append(m.get("model_id") or m.get("modelId") or str(m))
                    
                    # Check if any of our Card2Card models are in this list
                    for model_id in model_ids:
                        if model_id in normalized_models:
                            if model_id not in fallback_models_with_tables:
                                fallback_models_with_tables[model_id] = []
                            if table_path not in fallback_models_with_tables[model_id]:
                                fallback_models_with_tables[model_id].append(table_path)
                            if table_path not in fallback_table_paths:
                                fallback_table_paths.append(table_path)
    
    # Model Search: always get tables from parquet (DuckDB batch when available). No "intermediate" - that's Table Search only.
    use_fallback = False
    if use_citationlake and not citationlake_available and relationship_df is None:
        if fallback_table_paths:
            use_fallback = True
        else:
            return {
                "success": False,
                "error": "Cannot get tables: get_from-style mapping not available, relationship parquet failed to load, and no fallback data found in search results",
                "integrated_table": None
            }
    else:
        # Use relationship_df approach if get_from-style mapping is not available or if explicitly requested
        if not citationlake_available or (relationship_df is not None and not use_citationlake):
            if relationship_df is None:
                # Try fallback before giving up
                if fallback_table_paths:
                    use_fallback = True
                else:
                    return {
                        "success": False,
                        "error": "relationship_df is required when get_from-style mapping is not available, and no fallback data found",
                        "integrated_table": None
                    }
            else:
                use_citationlake = False
    
    # Collect all table paths from all models; also build model_id -> table_paths for UI debug
    all_table_paths = []
    models_with_tables = []
    models_without_tables = []
    model_to_table_paths: Dict[str, List[str]] = {}
    
    # Use fallback only when parquet/get_from-style mapping is unavailable (e.g. from card2tab2card_results)
    if use_fallback:
        print(f"⚠️  Using {len(fallback_table_paths)} fallback tables from search results JSON (parquet unavailable)")
        all_table_paths = fallback_table_paths  # No table top-k cap
        models_with_tables = list(fallback_models_with_tables.keys())
        models_without_tables = [m for m in model_ids if m not in models_with_tables]
        for mid in models_with_tables:
            model_to_table_paths[mid] = fallback_models_with_tables.get(mid, [])
    elif relationship_parquet and os.path.exists(relationship_parquet):
        # Fast path: DuckDB batch query (single SQL scan, no full parquet load)
        t_duck = time.time()
        print(f"✅ DuckDB batch query on parquet (fast, no full load)")
        model_to_tables = _get_tables_for_models_duckdb(relationship_parquet, model_ids)
        for model_id in model_ids:
            basenames = model_to_tables.get(model_id, [])
            model_to_table_paths[model_id] = []
            if basenames:
                # No per-model table cap: use ALL tables from each model
                for bn in basenames:
                    path = resolve_table_path(bn)
                    if path and path not in all_table_paths:
                        all_table_paths.append(path)
                    if path:
                        model_to_table_paths[model_id].append(path)
                models_with_tables.append(model_id)
                print(f"  ✅ Model {model_id}: {len(basenames)} tables")
            else:
                models_without_tables.append(model_id)
                print(f"  ⚠️  Model {model_id}: No tables found")
        print(f"   DuckDB query elapsed: {time.time() - t_duck:.2f}s")
    else:
        # Fallback: per-model get_tables_for_model (slower)
        for model_id in model_ids:
            model_to_table_paths[model_id] = []
            # Get tables for this model
            if use_citationlake and citationlake_available:
                model_tables = get_tables_for_model(
                    model_id=model_id,
                    schema_log_path=schema_log_path,
                    use_citationlake=True
                )
            else:
                if relationship_df is None:
                    raise ValueError("relationship_df is required when use_citationlake=False")
                model_tables = get_tables_for_model(
                    model_id=model_id,
                    relationship_df=relationship_df,
                    use_citationlake=False
                )

            if model_tables:
                # Resolve basenames to path (utils.resolve_table_path) or keep basename for load_table later
                for table_basename in model_tables:
                    table_path = resolve_table_path(table_basename) or table_basename
                    if table_path not in all_table_paths:
                        all_table_paths.append(table_path)
                    model_to_table_paths[model_id].append(table_path)

                models_with_tables.append(model_id)
                print(f"  ✅ Model {model_id}: {len(model_tables)} tables")
            else:
                models_without_tables.append(model_id)
                print(f"  ⚠️  Model {model_id}: No tables found")
    
    if not all_table_paths:
        return {
            "success": False,
            "error": f"No tables found for any of the {len(model_ids)} models",
            "integrated_table": None,
            "stats": {
                "models_processed": len(model_ids),
                "models_with_tables": len(models_with_tables),
                "models_without_tables": len(models_without_tables)
            }
        }
    
    print(f"\n✅ Collected {len(all_table_paths)} unique tables from {len(models_with_tables)} models")
    # No table top-k cap: use ALL tables
    table_paths = all_table_paths
    print(f"📊 Integration (Model Search): #tables={len(table_paths)}, #models={len(models_with_tables)} (no table top-k cap)")
    # Integrate tables
    result = integrate_tables(table_paths, integration_type, k, db_path)
    
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
    duckdb_used = relationship_parquet and os.path.exists(relationship_parquet) and not use_fallback
    # Attach timing + DuckDB info to stats for debugging
    if isinstance(result.get("stats"), dict):
        result["stats"]["elapsed_seconds"] = elapsed
        result["stats"]["duckdb_used"] = bool(duckdb_used)
    print(f"⏱️  Model Search integration elapsed: {elapsed:.2f}s (DuckDB={'yes' if duckdb_used else 'no'})")
    return result


if __name__ == "__main__":
    # Quick test: integration from template search results
    import os as _os
    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../.."))
    _path = _os.path.join(_root, "config", "demo_template", "search_results.json")
    if not _os.path.exists(_path):
        print("Test skip: config/demo_template/search_results.json not found")
    else:
        r = integrate_tables_from_model_search_results(_path, integration_type="union", k=10, max_models=5, schema_log_path=_os.path.join(_root, "data_citationlake", "logs", "parquet_schema.log"), use_citationlake=True)
        print("Test integration: success", r.get("success"), "stats", r.get("stats"))

