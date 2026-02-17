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
import pandas as pd
import json
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import Counter

# Base dirs for table lookup (in order of priority)
TABLE_BASE_DIRS = [
    "data_citationlake/processed/deduped_hugging_csvs",
    "data_citationlake/processed/deduped_github_csvs",
    "data_citationlake/processed/tables_output",
]

_CACHED_BASENAME_TO_PATH: Optional[Dict[str, str]] = None


def _build_basename_index() -> Dict[str, str]:
    """Build basename->fullpath index for fast table lookup. Uses DuckDB/SQL for speed."""
    global _CACHED_BASENAME_TO_PATH
    if _CACHED_BASENAME_TO_PATH is not None:
        return _CACHED_BASENAME_TO_PATH
    index: Dict[str, str] = {}
    cwd = os.getcwd()
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


def _resolve_table_path(basename: str) -> Optional[str]:
    """Resolve CSV basename to full path using cached index or fallback search."""
    base = os.path.basename(basename)
    idx = _build_basename_index()
    if base in idx:
        return idx[base]
    for base_dir in TABLE_BASE_DIRS:
        p = os.path.join(base_dir, base)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


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
    try:
        df = conn.execute(f"""
            SELECT modelId FROM read_parquet(?)
            WHERE modelId IN ({ids_sql})
        """, [path_abs]).fetchdf()
    except Exception:
        try:
            df = conn.execute(f"""
                SELECT modelId FROM read_parquet('{path_abs}')
                WHERE modelId IN ({ids_sql})
            """).fetchdf()
        except Exception:
            conn.close()
            return {}
    if df.empty:
        conn.close()
        return {}
    try:
        cols = conn.execute("DESCRIBE SELECT * FROM read_parquet(?)", [path_abs]).fetchall()
    except Exception:
        cols = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{path_abs}')").fetchall()
    list_cols = [
        c[0] for c in cols
        if c[0] != "modelId" and ("csv" in c[0].lower() or "table_list" in c[0].lower())
    ][:4]
    if not list_cols:
        conn.close()
        return {}
    select_cols = "modelId, " + ", ".join(f'"{c}"' for c in list_cols)
    try:
        full_df = conn.execute(f"""
            SELECT {select_cols} FROM read_parquet(?)
            WHERE modelId IN ({ids_sql})
        """, [path_abs]).fetchdf()
    except Exception:
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
BLEND_AVAILABLE = False
try:
    # Check if Blend_internal exists
    if blend_path_abs and os.path.exists(blend_path_abs):
        BLEND_AVAILABLE = True
except:
    pass


def load_table_from_file(table_path: str, use_index: bool = True) -> Optional[pd.DataFrame]:
    """
    Load a table from CSV file.
    
    Supports multiple path locations including data_citationlake and CitationLake paths.
    When use_index=True, uses cached basename->path index for fast lookup.
    
    Args:
        table_path: Path to CSV file (can be basename or full path)
        use_index: If True, use pre-built index for faster resolution (default True)
        
    Returns:
        DataFrame or None if file not found
    """
    basename = os.path.basename(table_path)
    
    # Build comprehensive list of possible base directories
    possible_base_dirs = [
        # data_citationlake paths (current structure)
        "data_citationlake/processed/deduped_hugging_csvs",
        "data_citationlake/processed/deduped_github_csvs",
        "data_citationlake/processed/tables_output",
        # CitationLake paths (if CitationLake is in parent directory)
        "../CitationLake/data/processed/deduped_hugging_csvs",
        "../CitationLake/data/processed/deduped_github_csvs",
        "../CitationLake/data/processed/tables_output",
        # Alternative CitationLake paths
        "../../CitationLake/data/processed/deduped_hugging_csvs",
        "../../CitationLake/data/processed/deduped_github_csvs",
        "../../CitationLake/data/processed/tables_output",
    ]
    
    # Strategy 1: Try the provided path first (if it's already a full path)
    possible_paths = [table_path]
    
    # Strategy 2: Try to infer directory from table_path if it contains path hints
    path_lower = table_path.lower()
    if "hugging" in path_lower:
        for base_dir in [d for d in possible_base_dirs if "hugging" in d.lower()]:
            if base_dir.startswith('../'):
                abs_base_dir = os.path.abspath(os.path.join(os.getcwd(), base_dir))
            else:
                abs_base_dir = os.path.abspath(base_dir)
            possible_paths.append(os.path.join(abs_base_dir, basename))
    elif "github" in path_lower:
        for base_dir in [d for d in possible_base_dirs if "github" in d.lower()]:
            if base_dir.startswith('../'):
                abs_base_dir = os.path.abspath(os.path.join(os.getcwd(), base_dir))
            else:
                abs_base_dir = os.path.abspath(base_dir)
            possible_paths.append(os.path.join(abs_base_dir, basename))
    
    # Strategy 3: Add all possible base directories
    for base_dir in possible_base_dirs:
        if base_dir.startswith('../'):
            abs_base_dir = os.path.abspath(os.path.join(os.getcwd(), base_dir))
        else:
            abs_base_dir = os.path.abspath(base_dir)
        possible_paths.append(os.path.join(abs_base_dir, basename))
    
    # Fast path: use cached index if available
    if use_index:
        resolved = _resolve_table_path(table_path)
        if resolved and os.path.exists(resolved):
            try:
                return pd.read_csv(resolved)
            except Exception as e:
                print(f"⚠️  Error loading {resolved}: {e}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in possible_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    
    # Try each path
    for path in unique_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df
            except Exception as e:
                print(f"⚠️  Error loading {path}: {e}")
                continue
    
    print(f"⚠️  Table not found: {table_path} (basename: {basename})")
    print(f"   Searched in {len(unique_paths)} possible locations")
    return None


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
    
    try:
        # Use pandas concat for union (combine all rows)
        # Align columns first
        all_columns = set()
        for df in tables:
            all_columns.update(df.columns)
        
        # Align all tables to have the same columns
        aligned_tables = []
        for df in tables:
            aligned_df = df.copy()
            for col in all_columns:
                if col not in aligned_df.columns:
                    aligned_df[col] = None
            aligned_tables.append(aligned_df[list(all_columns)])
        
        # Union (concatenate) all tables - return full result, no row cap
        integrated = pd.concat(aligned_tables, axis=0, ignore_index=True)
        
        # Remove duplicates if needed
        integrated = integrated.drop_duplicates()
        
        return integrated
    
    except Exception as e:
        print(f"❌ Error in union integration: {e}")
        return None


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
    
    try:
        # Find common columns
        common_columns = set(tables[0].columns)
        for df in tables[1:]:
            common_columns = common_columns.intersection(set(df.columns))
        
        if not common_columns:
            print("⚠️  No common columns found for intersection")
            # Return empty DataFrame with no columns instead of None
            # This allows the caller to handle it gracefully
            return pd.DataFrame()
        
        # Convert to string for comparison
        common_columns = list(common_columns)
        
        # Find intersection (rows that appear in all tables)
        # Start with first table
        result = tables[0][common_columns].copy()
        result['_temp_key'] = result.apply(lambda x: '|'.join(x.astype(str)), axis=1)
        
        # Find rows that exist in all other tables
        for df in tables[1:]:
            df_subset = df[common_columns].copy()
            df_subset['_temp_key'] = df_subset.apply(lambda x: '|'.join(x.astype(str)), axis=1)
            
            # Keep only rows that exist in current table
            result = result[result['_temp_key'].isin(df_subset['_temp_key'])]
        
        # Remove temp key
        result = result[common_columns]
        
        return result
    
    except Exception as e:
        print(f"❌ Error in intersection integration: {e}")
        return None


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
    
    try:
        # Add dialite_internal to path
        if dialite_repo not in sys.path:
            sys.path.insert(0, dialite_repo)
        
        # Import alite module
        import alite.alite_fd as alite_module
        
        # ALITE requires file paths, not DataFrames
        # Use the provided table_paths
        if not table_paths:
            print("⚠️  ALITE requires file paths, not DataFrames")
            return None
        
        # Run ALITE algorithm - return full result, no row cap
        result_FD, stats_df, debug_dict = alite_module.FDAlgorithm(table_paths.copy())
        
        if result_FD is not None and len(result_FD) > 0:
            return result_FD
        return result_FD
    
    except Exception as e:
        print(f"❌ Error in ALITE integration: {e}")
        import traceback
        traceback.print_exc()
        return None


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
    
    try:
        # Start with first table
        result = tables[0].copy()
        
        # Merge all other tables using outer join on index
        # This combines all columns from all tables
        for df in tables[1:]:
            # Reset index for both to ensure proper merging
            result_reset = result.reset_index(drop=True)
            df_reset = df.reset_index(drop=True)
            
            # Outer join: keep all rows from both tables
            # Use index as join key (implicit)
            result = pd.concat([result_reset, df_reset], axis=1, join='outer')
        
        return result
    
    except Exception as e:
        print(f"❌ Error in outer join integration: {e}")
        import traceback
        traceback.print_exc()
        return None


def integrate_tables(
    table_paths: List[str],
    integration_type: str = "union",
    k: int = 10,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Integrate multiple tables from file paths.
    
    Args:
        table_paths: List of paths to CSV files (caller typically passes top-k tables)
        integration_type: "union", "intersection", "alite", or "outer_join"
        k: Number of tables to integrate (top k); the integrated result is not truncated by row count
        db_path: Optional path to modellake.db (for Blend_internal integration)
        
    Returns:
        Dictionary with integration results
    """
    print(f"\n{'='*60}")
    print(f"🔗 Table Integration")
    print(f"{'='*60}")
    print(f"Integration type: {integration_type}")
    print(f"Number of tables: {len(table_paths)}")
    print(f"Top K: {k}")
    
    # Load all tables
    tables = []
    loaded_paths = []
    
    for table_path in table_paths:
        df = load_table_from_file(table_path)
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
    
    # Get retrieved table filenames
    # Try retrieved_table_filenames first, then fall back to table_to_models keys
    retrieved_filenames = intermediate.get("retrieved_table_filenames", [])
    
    # If no retrieved_table_filenames, try to extract from table_to_models
    if not retrieved_filenames:
        table_to_models = intermediate.get("table_to_models", {})
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
    
    if tables_source == "all_from_modelcards" and relationship_parquet and os.path.exists(relationship_parquet):
        # Get model_ids from table_to_models, then DuckDB batch query
        model_ids = set()
        table_to_models = intermediate.get("table_to_models", {})
        for table_path, model_list in table_to_models.items():
            for m in (model_list if isinstance(model_list, list) else []):
                mid = m.get("model_id") or m.get("modelId") if isinstance(m, dict) else str(m)
                if mid:
                    model_ids.add(mid)
        model_ids = list(model_ids)[:50]
        if not model_ids:
            return {"success": False, "error": "No model IDs in intermediate for all_from_modelcards", "integrated_table": None}
        print(f"✅ tables_source=all_from_modelcards: DuckDB batch query for {len(model_ids)} models")
        model_to_tables = _get_tables_for_models_duckdb(relationship_parquet, model_ids)
        all_basenames = []
        for mid in model_ids:
            all_basenames.extend(model_to_tables.get(mid, [])[:20])
        seen = set()
        table_paths = []
        for bn in all_basenames:
            if bn in seen:
                continue
            seen.add(bn)
            p = _resolve_table_path(bn)
            if p:
                table_paths.append(p)
            if len(table_paths) >= k:
                break
        print(f"   Resolved {len(table_paths)} tables for integration")
    else:
        # intermediate: use retrieved tables (current behavior)
        print(f"✅ tables_source=intermediate: {len(retrieved_filenames)} tables from search")
        table_paths = retrieved_filenames[:k]
    
    if not table_paths:
        return {"success": False, "error": "No tables to integrate", "integrated_table": None}
    
    return integrate_tables(table_paths, integration_type, k, db_path)


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
        schema_log_path: Path to parquet_schema.log (for CitationLake)
        use_citationlake: Whether to use CitationLake get_from (default: True)
        card2card_retrieval_mode: Optional retrieval mode: "dense", "sparse", or "hybrid"
        
    Returns:
        Dictionary with integration results
    """
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
    print(f"   Processing first {min(max_models, len(model_ids))} models")
    
    # Limit to max_models
    model_ids = model_ids[:max_models]
    
    # Import get_tables_for_model and load_relationship_parquet
    # Import directly from card2tab2card to avoid triggering card2card imports that require torch
    # Add parent directory to path if needed
    import sys
    parent_dir = os.path.join(os.path.dirname(__file__), '../..')
    parent_dir_abs = os.path.abspath(parent_dir)
    if parent_dir_abs not in sys.path:
        sys.path.insert(0, parent_dir_abs)
    
    # Import directly from card2tab2card module (avoiding __init__.py which imports card2card)
    try:
        # Try direct import first (avoids triggering card2card imports)
        import importlib.util
        card2tab2card_path = os.path.join(os.path.dirname(__file__), '..', 'search', 'card2tab2card.py')
        if os.path.exists(card2tab2card_path):
            spec = importlib.util.spec_from_file_location("card2tab2card", card2tab2card_path)
            card2tab2card_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(card2tab2card_module)
            get_tables_for_model = card2tab2card_module.get_tables_for_model
            load_relationship_parquet = card2tab2card_module.load_relationship_parquet
        else:
            # Fallback to regular import
            from src.search.card2tab2card import get_tables_for_model, load_relationship_parquet
    except Exception as e:
        # Last resort: try importing from search module (may trigger torch requirement)
        try:
            from src.search.card2tab2card import get_tables_for_model, load_relationship_parquet
        except ImportError:
            raise ImportError(f"Could not import get_tables_for_model and load_relationship_parquet: {e}")
    
    # Load relationship data if needed
    relationship_df = None
    if not use_citationlake:
        # Only load relationship parquet if CitationLake is not being used
        if relationship_parquet and os.path.exists(relationship_parquet):
            try:
                relationship_df = load_relationship_parquet(relationship_parquet)
                print(f"✅ Loaded relationship parquet: {relationship_parquet}")
            except Exception as e:
                print(f"⚠️  Failed to load relationship parquet {relationship_parquet}: {e}")
                print(f"   Will try to use CitationLake approach instead...")
                use_citationlake = True  # Fallback to CitationLake
        else:
            # Try default path
            default_parquet = "data_citationlake/processed/modelcard_step3_dedup.parquet"
            if os.path.exists(default_parquet):
                try:
                    relationship_df = load_relationship_parquet(default_parquet)
                    print(f"✅ Loaded default relationship parquet: {default_parquet}")
                except Exception as e:
                    print(f"⚠️  Failed to load default relationship parquet: {e}")
                    print(f"   Will try to use CitationLake approach instead...")
                    use_citationlake = True  # Fallback to CitationLake
    
    # Check if CitationLake is actually available
    try:
        # Try to check if CitationLake get_from is available
        card2tab2card_path = os.path.join(os.path.dirname(__file__), '..', 'search', 'card2tab2card.py')
        if os.path.exists(card2tab2card_path):
            spec = importlib.util.spec_from_file_location("card2tab2card_check", card2tab2card_path)
            card2tab2card_check_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(card2tab2card_check_module)
            citationlake_available = getattr(card2tab2card_check_module, 'USE_CITATIONLAKE_GET_FROM', False)
        else:
            citationlake_available = False
    except:
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
                "error": "Cannot get tables: CitationLake not available, relationship parquet failed to load, and no fallback data found in search results",
                "integrated_table": None
            }
    else:
        # Use relationship_df approach if CitationLake is not available or if explicitly requested
        if not citationlake_available or (relationship_df is not None and not use_citationlake):
            if relationship_df is None:
                # Try fallback before giving up
                if fallback_table_paths:
                    use_fallback = True
                else:
                    return {
                        "success": False,
                        "error": "relationship_df is required when CitationLake is not available, and no fallback data found",
                        "integrated_table": None
                    }
            else:
                use_citationlake = False
    
    # Collect all table paths from all models
    all_table_paths = []
    models_with_tables = []
    models_without_tables = []
    
    # Use fallback only when parquet/CitationLake unavailable (e.g. from card2tab2card_results)
    if use_fallback:
        print(f"⚠️  Using {len(fallback_table_paths)} fallback tables from search results JSON (parquet unavailable)")
        all_table_paths = fallback_table_paths[:k]
        models_with_tables = list(fallback_models_with_tables.keys())
        models_without_tables = [m for m in model_ids if m not in models_with_tables]
    elif relationship_parquet and os.path.exists(relationship_parquet):
        # Fast path: DuckDB batch query (single SQL scan)
        print(f"✅ DuckDB batch query on parquet (fast)")
        model_to_tables = _get_tables_for_models_duckdb(relationship_parquet, model_ids)
        for model_id in model_ids:
            basenames = model_to_tables.get(model_id, [])
            if basenames:
                for bn in basenames[:50]:
                    path = _resolve_table_path(bn)
                    if path and path not in all_table_paths:
                        all_table_paths.append(path)
                models_with_tables.append(model_id)
                print(f"  ✅ Model {model_id}: {len(basenames)} tables")
            else:
                models_without_tables.append(model_id)
                print(f"  ⚠️  Model {model_id}: No tables found")
    else:
        # Fallback: per-model get_tables_for_model (slower)
        for model_id in model_ids:
            try:
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
                    # Convert basenames to full paths
                    for table_basename in model_tables:
                        # Try to find the full path
                        table_path = None
                        
                        # Try common base directories
                        possible_base_dirs = [
                            "data_citationlake/processed/deduped_hugging_csvs",
                            "data_citationlake/processed/deduped_github_csvs",
                            "data_citationlake/processed/tables_output",
                        ]
                        
                        for base_dir in possible_base_dirs:
                            full_path = os.path.join(base_dir, table_basename)
                            if os.path.exists(full_path):
                                table_path = full_path
                                break
                        
                        # If not found, try using load_table_from_file to find it
                        if not table_path:
                            test_df = load_table_from_file(table_basename)
                            if test_df is not None:
                                # Find which path worked
                                for base_dir in possible_base_dirs:
                                    full_path = os.path.join(base_dir, table_basename)
                                    if os.path.exists(full_path):
                                        table_path = full_path
                                        break
                        
                        if table_path and table_path not in all_table_paths:
                            all_table_paths.append(table_path)
                    
                    models_with_tables.append(model_id)
                    print(f"  ✅ Model {model_id}: {len(model_tables)} tables")
                else:
                    models_without_tables.append(model_id)
                    print(f"  ⚠️  Model {model_id}: No tables found")
            except Exception as e:
                print(f"  ❌ Error getting tables for model {model_id}: {str(e)}")
                models_without_tables.append(model_id)
    
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
    print(f"   Using first {min(k, len(all_table_paths))} tables for integration")
    
    # Limit to k tables
    table_paths = all_table_paths[:k]
    
    # Integrate tables
    result = integrate_tables(table_paths, integration_type, k, db_path)
    
    # Add model search specific stats
    if result.get("success"):
        result["stats"]["models_processed"] = len(model_ids)
        result["stats"]["models_with_tables"] = len(models_with_tables)
        result["stats"]["models_without_tables"] = len(models_without_tables)
        result["stats"]["total_unique_tables"] = len(all_table_paths)
        result["model_ids"] = model_ids
        result["models_with_tables"] = models_with_tables
        result["models_without_tables"] = models_without_tables
    
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

