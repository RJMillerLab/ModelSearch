"""
Card to Tab to Card Search

This module provides a two-stage search:
1. Given a model card, find its associated tables
2. Search for similar tables using tab2tab
3. Map similar tables back to model cards using relationship parquet

Uses a get_from.py-style approach (via ModelTables / data_citationlake) for robust parquet schema handling when available.
"""

import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set, Dict, Optional, Any, Tuple
import argparse
import pandas as pd

# Add parent directory to path for imports (so "from src.*" works when run as script or from demo)
_reporoot = os.path.join(os.path.dirname(__file__), '../..')
if _reporoot not in sys.path:
    sys.path.insert(0, _reporoot)
from src.utils.table_loader import resolve_table_path
from src.modelsearch.compare_baselines import (
    get_tables_for_model_duckdb,
    get_modelids_for_basenames_duckdb,
    _read_relationships,
)


def _resolve_csv_path(table_path) -> Optional[str]:
    """Resolve table_path to CSV path: use as-is if exists, else resolve_table_path(basename)."""
    p = str(table_path)
    if os.path.exists(p):
        return p
    return resolve_table_path(os.path.basename(p))


def _get_csv_headers(csv_path: str) -> List[str]:
    """Read CSV header row and return normalized column names (lower, stripped, non-empty)."""
    df = pd.read_csv(csv_path, nrows=0)
    return [str(c).lower().strip() for c in df.columns if str(c).strip()]


def _get_table_query(csv_path: str, table_path, search_type: str):
    """Build tquery for search: headers (keyword) or first column values (single_column)."""
    headers = _get_csv_headers(csv_path)
    if search_type == "single_column":
        df_read = pd.read_csv(csv_path, nrows=100)
        if len(df_read) > 0 and len(df_read.columns) > 0:
            return df_read[df_read.columns[0]].dropna().astype(str).tolist()
    return headers or [os.path.basename(str(table_path))]


# Lazy import tab2tab to avoid DBHandler initialization on import
# tab2tab will only be imported when search_card2tab2card is actually called
_tab2tab_search_table2table = None
_tab2tab_by_type_search_table2table_by_type = None

def _get_search_table2table():
    """Lazy import search_table2table from tab2tab."""
    global _tab2tab_search_table2table
    if _tab2tab_search_table2table is None:
        import sys
        print(f"   [DEBUG] Importing search_table2table from tab2tab...")
        sys.stdout.flush()
        from src.search.tab2tab import search_table2table
        print(f"   [DEBUG] Import successful")
        sys.stdout.flush()
        _tab2tab_search_table2table = search_table2table
    return _tab2tab_search_table2table

def _get_search_table2table_by_type():
    """Lazy import search_table2table_by_type from tab2tab_by_type."""
    global _tab2tab_by_type_search_table2table_by_type
    if _tab2tab_by_type_search_table2table_by_type is None:
        import sys
        print(f"   [DEBUG] Importing search_table2table_by_type from tab2tab_by_type...")
        sys.stdout.flush()
        from src.search.tab2tab_by_type import search_table2table_by_type
        print(f"   [DEBUG] Import successful")
        sys.stdout.flush()
        _tab2tab_by_type_search_table2table_by_type = search_table2table_by_type
    return _tab2tab_by_type_search_table2table_by_type

# Default path for model–table relationship (used when caller does not pass relationship_parquet)
DEFAULT_RELATIONSHIP_PARQUET = "data_citationlake/processed/modelcard_step3_dedup.parquet"


def get_modelids_from_table(
    table_path: str,
    relationship_parquet: Optional[str] = None,
    debug: bool = False
) -> List[str]:
    """
    Get model IDs that have a specific table (by CSV basename), using relationship parquet.
    Uses DuckDB when parquet exists, else loads parquet and does DataFrame lookup.
    """
    parquet = relationship_parquet or DEFAULT_RELATIONSHIP_PARQUET
    basename = os.path.basename(str(table_path))
    if os.path.exists(parquet):
        mapping = get_modelids_for_basenames_duckdb(parquet, [basename])
        return list(mapping.get(basename, []))
    df = _read_relationships(parquet)
    basename_col = next((c for c in ["csv_basename", "basename", "filename"] if c in df.columns), None)
    if basename_col is None:
        return []
    mids = df.loc[df[basename_col] == basename, "modelId"].dropna().unique().tolist()
    return [str(m) for m in mids]


def _classify_table_source_by_basename(basename: str) -> str:
    """
    Classify table source from filename (from ModelTables batch_process_tables / quick_retrieval).
    Returns: 'github', 'huggingface', 'html', 'llm', or 'unknown'
    """
    b = str(basename).replace("_s.csv", ".csv").replace("_t.csv", ".csv")
    # GitHub: 32 hex + _table_N.csv
    if re.fullmatch(r"[0-9a-f]{32}_table_\d+\.csv", b):
        return "github"
    # HTML/arXiv: D.D(vN)_tableN.csv
    if re.fullmatch(r"\d+\.\d+(?:v\d+)?_table\d+\.csv", b):
        return "html"
    # HuggingFace: 10 hex + _tableN.csv
    if re.fullmatch(r"[0-9a-f]{10}_table\d+\.csv", b):
        return "huggingface"
    # LLM/S2ORC: digits only before _table (e.g. 215768677_table2.csv, 237485280_table3.csv)
    if re.fullmatch(r"\d+_table\d+\.csv", b):
        return "llm"
    return "unknown"


# Generic tables that appear in countless model cards - invalid for search (from ModelTables paper)
# 1910.09700_table: Google Cloud carbon emission template (Lacoste et al.); 204823751_table: country code data
GENERIC_TABLE_PATTERNS = ["1910.09700_table", "204823751_table"]


def _is_generic_table(basename_or_path: str) -> bool:
    """True if table is a known generic set (appears in countless model cards, invalid for search)."""
    base = os.path.basename(str(basename_or_path))
    return any(p in base for p in GENERIC_TABLE_PATTERNS)


def _filter_s2orc_tables(tables: List[str]) -> List[str]:
    """Filter out s2orc/llm tables (unstable). Use ModelTables naming rules to infer source."""
    def is_s2orc_or_llm(p: str) -> bool:
        base = os.path.basename(str(p))
        src = _classify_table_source_by_basename(base)
        return src == "llm"
    out = [t for t in tables if not is_s2orc_or_llm(t)]
    if len(out) < len(tables):
        print(f"ℹ️  Filtered out {len(tables) - len(out)} s2orc/llm tables (remain: {len(out)})")
    return out


def load_relationship_parquet(parquet_path: str) -> pd.DataFrame:
    """
    Load relationship parquet file that maps modelId to CSV basenames.
    Returns DataFrame with columns: modelId, csv_basename
    """
    return _read_relationships(parquet_path)


def get_tables_for_model(
    model_id: str,
    relationship_df: Optional[pd.DataFrame] = None,
    relationship_parquet: Optional[str] = None,
) -> List[str]:
    """
    Get list of table CSV basenames for a given model ID.
    Uses DuckDB when relationship_parquet exists, else DataFrame (given or loaded from parquet).
    """
    parquet = relationship_parquet or DEFAULT_RELATIONSHIP_PARQUET
    if os.path.exists(parquet):
        return get_tables_for_model_duckdb(parquet, model_id)
    if relationship_df is None:
        relationship_df = load_relationship_parquet(parquet)
    csvs = relationship_df.loc[
        relationship_df["modelId"] == model_id,
        "csv_basename"
    ].dropna().unique().tolist()
    return csvs


def search_card2tab2card(
    model_id: str,
    relationship_parquet: Optional[str] = None,
    query: Optional[Any] = None,
    search_type: str = "single_column",
    k: int = 10,
    table_search_k: Optional[int] = None,
    modelcard_k: Optional[int] = None,
    schema_log_path: str = "data_citationlake/logs/parquet_schema.log",
    use_citationlake: bool = True,
    output_json: str = "data/card2tab2card_results.json",
    db_path: Optional[str] = None,
    global_table_topk: bool = True,
) -> List[str]:
    """
    Search for model cards via table search.
    
    Pipeline: Query -> ModelCard -> Tables -> Retrieved Tables -> Corresponding ModelCards
    
    Process:
    1. Get tables for the query model using a get_from-style function (or relationship parquet)
    2. Use tab2tab to search for similar tables
    3. Map retrieved tables back to model cards using a get_from-style function (or relationship)
    
    Args:
        model_id: Hugging Face model ID to search from
        relationship_parquet: Optional path to relationship parquet file (fallback if get_from-based approach not available)
        query: Optional query data for table search. If None, uses tables from the model
        search_type: Type of table search - "single_column", "multi_column", "keyword", or "unionable"
        k: Legacy parameter for backward compatibility. If table_search_k or modelcard_k are None, uses k for both.
        table_search_k: Number of table results to retrieve (defaults to k if not provided)
        modelcard_k: Number of final model card results to return (defaults to k if not provided)
        schema_log_path: Path to parquet_schema.log (for get_from-style approach)
        use_citationlake: Whether to use get_from-style mapping (default: True)
        output_json: Optional path to save results as JSON
        db_path: Path to modellake.db (default: data_citationlake/modellake.db)
        global_table_topk: If True (default), when model has multiple tables: search each table as equivalent query,
            merge results round-robin, take global top-k. Ensures table_search_k limits total tables, not per-table.
    
    Returns:
        List of model IDs that have similar tables
    """
    # Handle backward compatibility: if table_search_k or modelcard_k are None, use k
    if table_search_k is None:
        table_search_k = k
    if modelcard_k is None:
        modelcard_k = k
    # Print pipeline overview
    import sys
    sys.stdout.flush()  # Ensure output is flushed
    print(f"\n{'='*60}")
    print(f"🔍 Card2Tab2Card Search Pipeline")
    print(f"{'='*60}")
    print(f"Pipeline: Query -> ModelCard -> Tables -> Retrieved Tables -> Corresponding ModelCards")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    # Resolve relationship parquet path (default or first existing)
    if relationship_parquet is None:
        for p in (DEFAULT_RELATIONSHIP_PARQUET, "data_citationlake/processed/modelcard_step3.parquet"):
            if os.path.exists(p):
                relationship_parquet = p
                break
        if relationship_parquet is None:
            relationship_parquet = DEFAULT_RELATIONSHIP_PARQUET
    if not os.path.exists(relationship_parquet):
        raise FileNotFoundError(
            f"relationship_parquet not found: {relationship_parquet}\n"
            "Use --relationship_parquet or place modelcard_step3_dedup.parquet in data_citationlake/processed/"
        )
    # Get tables for the query model (DuckDB or full parquet load)
    query_tables = get_tables_for_model(model_id=model_id, relationship_parquet=relationship_parquet)
    query_tables = _filter_s2orc_tables(query_tables)
    if not query_tables:
        print(f"⚠️  Warning: No tables found for model {model_id}")
        # Still save intermediate results even if no query tables
        if output_json:
            result = {
                "query_model": model_id,
                "query_tables": [],
                "model_ids": [],
                "intermediate": {
                    "retrieved_table_ids": [],
                    "retrieved_table_filenames": [],
                    "table_id_to_filename": {},
                    "table_to_models": {}
                }
            }
            os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
            with open(output_json, 'w', encoding='utf-8') as f:
                import json
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ Results saved to {output_json}")
        return []
    
    print(f"\n{'='*60}")
    print(f"[Card2Tab2Card] INPUT: model_id={model_id} search_type={search_type} k={table_search_k} | OUTPUT: {output_json}")
    print(f"{'='*60}")
    # Step 1: Query -> ModelCard -> Tables (Part of Pipeline)
    print(f"[STEP 1] INPUT: model_id={model_id} | OUTPUT: query_tables (for Step 2)")
    print(f"✅ Query Model ID: {model_id}")
    print(f"✅ Found {len(query_tables)} tables for model {model_id}")
    print(f"📝 Sample tables (showing first 2):")
    for i, table in enumerate(query_tables[:2], 1):
        print(f"   {i}. {os.path.basename(str(table))}")
    if len(query_tables) > 2:
        print(f"   ... and {len(query_tables) - 2} more tables")
    
    # If query is provided, use it; otherwise search based on query_tables
    use_per_table_search = (
        query is None
        and global_table_topk
        and len(query_tables) > 1
        and search_type in ("keyword", "single_column")  # These support per-table query easily
    )
    if query is None:
        # Use the query model's tables as the search query
        # For keyword search, we need to load the actual CSV files to get headers
        # (Blend_internal uses rowid=-1 which represents headers in the index)
        print(f"ℹ️  No query provided, using model's tables as query")
        if use_per_table_search:
            print(f"ℹ️  global_table_topk=True: each of {len(query_tables)} tables as equivalent query, merge → global top-{table_search_k}")
        if search_type == "keyword":
            all_headers = []
            for table_path in query_tables[:10]:
                csv_path = _resolve_csv_path(table_path)
                if csv_path:
                    all_headers.extend(_get_csv_headers(csv_path))
            query = list(set(all_headers))
            if not query:
                query = [os.path.basename(str(t)) for t in query_tables[:10]]
        else:
            search_type = "keyword"
            all_headers = []
            for table_path in query_tables[:10]:
                csv_path = _resolve_csv_path(table_path)
                if csv_path:
                    all_headers.extend(_get_csv_headers(csv_path))
            query = list(set(all_headers))
            if not query:
                query = [os.path.basename(str(t)) for t in query_tables[:10]]
    else:
        print(f"ℹ️  Using provided query (not model's tables)")
    
    # Step 2: Tables -> Retrieved Tables
    _query_safe = query if query is not None else []
    print(f"\n{'='*60}")
    print(f"🔍 Step 2: Tables -> Retrieved Tables")
    print(f"{'='*60}")
    print(f"✅ Search type: {search_type}")
    if search_type == "keyword":
        print(f"✅ Query keywords: {_query_safe[:5]}{'...' if len(_query_safe) > 5 else ''}")
        print(f"   (Total {len(_query_safe)} keywords)")
    elif search_type == "single_column":
        print(f"✅ Query values: {_query_safe[:5]}{'...' if len(_query_safe) > 5 else ''}")
        print(f"   (Total {len(_query_safe)} values)")
    elif search_type in ["multi_column", "unionable", "complex", "correlation", "imputation", "augmentation", "dependent_data", "feature_for_ml", "multi_column_collinearity", "negative_example"]:
        print(f"✅ Query: DataFrame with {len(query)} rows and {len(query.columns)} columns")
        if search_type == "multi_column":
            print(f"   Multi-column search: finds tables with overlapping values across multiple columns")
        elif search_type == "complex":
            print(f"   Complex search combines: Union + Join + Correlation sub-pipelines")
        elif search_type == "correlation":
            print(f"   Correlation search: finds tables with correlated categorical and numerical columns")
        elif search_type == "imputation":
            print(f"   Imputation search: finds tables that can fill missing values based on examples")
        elif search_type == "augmentation":
            print(f"   Augmentation search: finds tables that can augment data based on examples")
        elif search_type == "dependent_data":
            print(f"   Dependent data search: finds tables with dependent column pairs")
        elif search_type == "feature_for_ml":
            print(f"   Feature for ML search: finds columns correlated with target but not with feature")
        elif search_type == "multi_column_collinearity":
            print(f"   Multi-column collinearity search: finds tables with correlated columns and multi-column overlap")
        elif search_type == "negative_example":
            print(f"   Negative example search: finds tables with exclusive but not inclusive examples")
    print(f"✅ Table Search Top K: {table_search_k}")
    print(f"✅ ModelCard Top K: {modelcard_k}")
    
    # Search for similar tables using tab2tab (lazy import)
    # Default db_path to data_citationlake/modellake.db if not provided
    if db_path is None:
        db_path = "data_citationlake/modellake.db"
    print(f"✅ Database: {db_path}")
    
    print(f"🔎 Getting search_table2table function...")
    sys.stdout.flush()
    search_table2table = _get_search_table2table()
    print(f"✅ Got search_table2table function")
    sys.stdout.flush()
    if use_per_table_search:
        # table_search_k = per-table k (user input, e.g. 1)
        # k_request = k_per_table + 4 (over-fetch for self/s2orc/generic filter)
        num_seed_tables = len(query_tables[:20])
        k_per_table = max(1, table_search_k)
        k_request = k_per_table + 4  # Over-fetch: top1 -> request 5, so after filter we can get top-k
        print(f"📊 Per-table k={k_per_table} | request={k_request} (over-fetch k+4) | flow: filter per table → top-{k_per_table} → merge (no post-filter)")
        print(f"🔎 [STEP 2a] Parallel per-table search (each table independent read-only), then filter→topk→merge")
        sys.stdout.flush()

        def _search_one_table(args: Tuple[int, str, str, str]) -> Optional[Tuple[List[int], str]]:
            """Run one table search. Returns (ids, query_basename) or None. Used for parallel. Fails fast on error."""
            ti, table_path, st, db = args
            csv_path = _resolve_csv_path(table_path)
            if not csv_path:
                return None
            df_required_types = {
                "multi_column", "unionable", "complex", "correlation", "imputation",
                "augmentation", "dependent_data", "feature_for_ml", "multi_column_collinearity", "negative_example",
            }
            if st in df_required_types:
                tquery = pd.read_csv(csv_path)
                if tquery is None or tquery.empty:
                    return None
            else:
                tquery = _get_table_query(csv_path, table_path, st)
                if not tquery:
                    return None
            t0 = time.time()
            ids = search_table2table(tquery, st, k_request, db_path=db)
            elapsed = time.time() - t0
            bn = os.path.basename(str(table_path))
            ts = time.strftime("%H:%M:%S")
            # For DataFrame queries, report (rows, cols); otherwise use len(query)
            if isinstance(tquery, pd.DataFrame):
                q_rows, q_cols = tquery.shape
                q_info = f"{q_rows}x{q_cols}"
            else:
                q_info = str(len(tquery))
            print(f"   [Table {ti+1}] @{ts} INPUT={bn} | query_len={q_info} k_request={k_request} | OUTPUT={len(ids) if ids else 0} ids | {elapsed:.1f}s")
            sys.stdout.flush()
            return (ids, bn) if ids else None

        tables_to_search = [(ti, tp, search_type, db_path) for ti, tp in enumerate(query_tables[:20])]
        max_workers = min(8, len(tables_to_search))  # Parallel table search
        results_by_ti: Dict[int, Tuple[List[int], str]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_search_one_table, a): a[0] for a in tables_to_search}
            for fut in as_completed(futures):
                ti = futures[fut]
                res = fut.result()
                if res:
                    results_by_ti[ti] = res
        per_table_results = [results_by_ti[ti] for ti in sorted(results_by_ti.keys())]
        print(f"   [STEP 2a] Done: {len(per_table_results)} tables searched in parallel")

        # Flow: 1) Filter per table (self/s2orc/generic), 2) top-k per table, 3) merge (no post-filter)
        all_ids = [tid for ids, _ in per_table_results for tid in ids]
        tableid_to_filename: Dict[int, str] = {}
        if all_ids:
            import duckdb
            with duckdb.connect(db_path, read_only=True) as con_filter:
                table_ids_str = ','.join(str(tid) for tid in all_ids)
                filename_query = f"""
                    SELECT DISTINCT tableid, filename
                    FROM modellake_index
                    WHERE tableid IN ({table_ids_str}) AND rowid = -1
                """
                filename_results = con_filter.execute(filename_query).fetchall()
                tableid_to_filename = {tid: filename for tid, filename in filename_results}

        per_table_filtered_topk: List[List[Tuple[int, str]]] = []
        n_self, n_s2orc, n_generic = 0, 0, 0
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
            per_table_filtered_topk.append(kept[:k_per_table])  # top-k per table
        if n_self or n_s2orc or n_generic:
            print(f"   [STEP 2b] Filter per table: self={n_self}, s2orc={n_s2orc}, generic={n_generic} | top-k={k_per_table}/table")
        print(f"   [STEP 2c] Merge (round-robin of per-table top-{k_per_table})")
        seen: Set[int] = set()
        similar_table_data: List[Tuple[int, str]] = []
        max_len = max(len(r) for r in per_table_filtered_topk) if per_table_filtered_topk else 0
        for rank in range(max_len):
            for row in per_table_filtered_topk:
                if rank < len(row) and row[rank][0] not in seen:
                    seen.add(row[rank][0])
                    similar_table_data.append(row[rank])
        similar_table_ids = [tid for tid, _ in similar_table_data]
        print(f"✅ Found {len(similar_table_ids)} tables (from {len(per_table_results)} query tables, per-table filter→top{k_per_table}→merge, no post-filter)")
    else:
        k_overfetch = table_search_k * 2  # Over-fetch, then filter, then topk
        print(f"🔎 [STEP 2] Single-query search: INPUT=query | k_request={k_overfetch} | OUTPUT=merged")
        print(f"   Query type: {type(query)}, Search type: {search_type}, table_search_k: {table_search_k}")
        sys.stdout.flush()
        # Handle correlation search specially - need to extract source and target columns
        if search_type == "correlation" and isinstance(query, pd.DataFrame):
            source_col = query[query.columns[0]].astype(str).tolist()
            numeric_cols = query.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                target_col = query[numeric_cols[0]].tolist()
                print(f"   Correlation: source='{query.columns[0]}', target='{numeric_cols[0]}'")
                similar_table_ids = search_table2table(
                    query, search_type, k_overfetch, db_path=db_path,
                    source_column=source_col, target_column=target_col
                )
            else:
                print(f"⚠️  No numeric column found for correlation search, skipping...")
                similar_table_ids = []
        else:
            similar_table_ids = search_table2table(query, search_type, k_overfetch, db_path=db_path)
        print(f"✅ Found {len(similar_table_ids)} retrieved tables (table IDs)")
    sys.stdout.flush()
    
    if not similar_table_ids:
        print(f"⚠️  No tables retrieved, cannot proceed to Step 3")
        # Still save intermediate results even if no tables retrieved
        if output_json:
            result = {
                "query_model": model_id,
                "query_tables": query_tables,
                "model_ids": [],
                "intermediate": {
                    "retrieved_table_ids": [],
                    "retrieved_table_filenames": [],
                    "table_id_to_filename": {},
                    "table_to_models": {}
                }
            }
            os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
            with open(output_json, 'w', encoding='utf-8') as f:
                import json
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ Results saved to {output_json}")
        return []
    
    # Get filenames for retrieved tables from database
    import duckdb
    with duckdb.connect(db_path, read_only=True) as con:
        table_ids_str = ','.join(str(tid) for tid in similar_table_ids)
        filename_query = f"""
            SELECT DISTINCT tableid, filename 
            FROM modellake_index 
            WHERE tableid IN ({table_ids_str}) AND rowid = -1
        """
        filename_results = con.execute(filename_query).fetchall()
        retrieved_table_files = {tid: filename for tid, filename in filename_results}
        print(f"📝 Sample retrieved tables (showing first 2):")
        for i, (tid, filename) in enumerate(list(retrieved_table_files.items())[:2], 1):
            print(f"   {i}. Table ID {tid}: {filename}")
        if len(retrieved_table_files) > 2:
            print(f"   ... and {len(retrieved_table_files) - 2} more retrieved tables")

    # Step 3: Retrieved Tables -> Corresponding ModelCards
    print(f"\n{'='*60}")
    print(f"[STEP 3] INPUT: {len(similar_table_ids)} retrieved table IDs | OUTPUT: model_ids (from relationship)")
    print(f"{'='*60}")
    print(f"✅ Mapping {len(similar_table_ids)} retrieved tables to model cards...")
    
    # Get filenames for retrieved tables from database
    import duckdb
    with duckdb.connect(db_path, read_only=True) as con:
        if not similar_table_ids:
            print(f"⚠️  No table IDs to map")
            # Still save intermediate results even if no tables found
            if output_json:
                result = {
                    "query_model": model_id,
                    "query_tables": query_tables,
                    "model_ids": [],
                    "intermediate": {
                        "retrieved_table_ids": [],
                        "retrieved_table_filenames": [],
                        "table_id_to_filename": {},
                        "table_to_models": {}
                    }
                }
                os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
                with open(output_json, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"✅ Results saved to {output_json}")
            return []
        
        table_ids_str = ','.join(str(tid) for tid in similar_table_ids)
        filename_query = f"""
            SELECT DISTINCT tableid, filename 
            FROM modellake_index 
            WHERE tableid IN ({table_ids_str}) AND rowid = -1
        """
        filename_results = con.execute(filename_query).fetchall()
        tableid_to_filename = {tid: filename for tid, filename in filename_results}
        # Per-table path: already filtered+topk+merge in Step 2, no post-filter
        if not (use_per_table_search and per_table_results):
            seed_basenames = {os.path.basename(str(t)) for t in query_tables}
            filtered = []
            for tid in similar_table_ids:
                if tid not in tableid_to_filename:
                    continue
                fbase = os.path.basename(str(tableid_to_filename[tid]))
                if fbase in seed_basenames or _classify_table_source_by_basename(fbase) == "llm" or _is_generic_table(fbase):
                    continue
                filtered.append(tid)
            similar_table_ids = filtered[:table_search_k]
        # Preserve Tab2Tab relevance order: iterate, dedupe by filename
        tables_before_dedup = len(similar_table_ids)
        seen_filenames = set()
        retrieved_filenames = []
        for tid in similar_table_ids:
            if tid in tableid_to_filename:
                fname = tableid_to_filename[tid]
                if fname not in seen_filenames:
                    seen_filenames.add(fname)
                    retrieved_filenames.append(fname)
        tables_after_dedup = len(retrieved_filenames)
        print(f"📊 Table set dedup: {tables_before_dedup} table_ids → {tables_after_dedup} unique filenames")
        # Filter generic tables (carbon emission template, country code - appear in countless model cards)
        retrieved_filenames = [f for f in retrieved_filenames if not _is_generic_table(f)]
        n_generic = tables_after_dedup - len(retrieved_filenames)
        if n_generic:
            print(f"ℹ️  Filtered {n_generic} generic tables (1910.09700_table, 204823751_table) - invalid for search")
        print(f"✅ Retrieved {len(retrieved_filenames)} unique filenames (ordered by Tab2Tab relevance)")
        # Debug: print query_tables and searched_tables (retrieved filenames used for model mapping)
        print(f"📋 query_tables ({len(query_tables)}): {[os.path.basename(str(t)) for t in query_tables[:5]]}{'...' if len(query_tables) > 5 else ''}")
        print(f"📋 searched_tables ({len(retrieved_filenames)}): {[os.path.basename(str(f)) for f in retrieved_filenames[:5]]}{'...' if len(retrieved_filenames) > 5 else ''}")
        
        if not retrieved_filenames:
            print(f"⚠️  No filenames found for retrieved table IDs")
            # Still save intermediate results even if no filenames found
            if output_json:
                # Build table_id to filename mapping (even if empty)
                table_id_to_filename_dict = {}
                for tid, filename in tableid_to_filename.items():
                    table_id_to_filename_dict[tid] = filename
                
                result = {
                    "query_model": model_id,
                    "query_tables": query_tables,
                    "model_ids": [],
                    "intermediate": {
                        "retrieved_table_ids": similar_table_ids if 'similar_table_ids' in locals() else [],
                        "retrieved_table_filenames": [],
                        "table_id_to_filename": table_id_to_filename_dict,
                        "table_to_models": {}
                    }
                }
                os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
                with open(output_json, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"✅ Results saved to {output_json}")
            return []

    # Map similar tables back to model cards
    similar_model_ids = set()
    table_to_models = {}  # Map table filename to list of model IDs
    models_raw_count = 0  # before dedup

    # Map table basenames to model IDs via relationship parquet (DuckDB or full load)
    table_basenames = [os.path.basename(fname) for fname in retrieved_filenames]
    print(f"📋 Using relationship_parquet to map tables to model cards...")
    print(f"📝 Matching {len(table_basenames)} table basenames...")
    print(f"   Sample basenames: {table_basenames[:3]}{'...' if len(table_basenames) > 3 else ''}")
    basename_to_models = get_modelids_for_basenames_duckdb(relationship_parquet, table_basenames)
    for filename in retrieved_filenames:
        basename = os.path.basename(filename)
        matched_models = basename_to_models.get(basename, [])
        if matched_models:
            models_raw_count += len(matched_models)
            similar_model_ids.update(matched_models)
            table_to_models[filename] = matched_models
            print(f"   ✅ Matched {basename} -> {len(matched_models)} models")
        else:
            print(f"   ⚠️  No match for {basename}")
    print(f"✅ Matched {len(similar_model_ids)} model cards from relationship data")
    
    # Per-table model count (value_counts): which tables span how many models (user cares about "countless" tables)
    if table_to_models:
        sorted_tables = sorted(table_to_models.keys(), key=lambda x: len(table_to_models[x]), reverse=True)
        print(f"📊 Per-table model counts (table -> #models, sorted desc):")
        for fname in sorted_tables:
            bn = os.path.basename(fname)
            n = len(table_to_models[fname])
            print(f"   {bn}: {n} models")
    
    # Remove the query model itself
    print(f"📊 Model set dedup: {models_raw_count} raw (sum over tables) → {len(similar_model_ids)} unique models (before excl. query)")
    similar_model_ids = [mid for mid in similar_model_ids if mid != model_id]
    
    # Also remove query model from table_to_models
    for filename in table_to_models:
        table_to_models[filename] = [mid for mid in table_to_models[filename] if mid != model_id]
    
    print(f"✅ Found {len(similar_model_ids)} unique model cards (excluding query model)")
    
    # Limit to top modelcard_k only if modelcard_k > 0 (0 or None = no limit: all models for these tables)
    if modelcard_k and modelcard_k > 0:
        final_results = list(similar_model_ids)[:modelcard_k]
        print(f"   (capped at top {modelcard_k} models)")
    else:
        final_results = list(similar_model_ids)
        print(f"   (no limit: returning all {len(final_results)} models that contain the retrieved tables)")
    
    # One-line TopK decision: seed_tables | searched_tables | model_cards
    print(f"📊 TopK decision: seed_tables={len(query_tables)} | searched_tables={len(similar_table_ids)} | model_cards={len(final_results)}")
    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 Final Results Summary")
    print(f"{'='*60}")
    print(f"✅ Query Model: {model_id}")
    print(f"✅ Found {len(final_results)} similar model cards" + (f" (top {modelcard_k})" if modelcard_k and modelcard_k > 0 else ""))
    if final_results:
        print(f"📝 Sample results (showing first 2):")
        for i, model_id_result in enumerate(final_results[:2], 1):
            print(f"   {i}. {model_id_result}")
        if len(final_results) > 2:
            print(f"   ... and {len(final_results) - 2} more model cards")
    print(f"{'='*60}\n")
    
    # Save if requested
    if output_json:
        # Build table_id to filename mapping
        table_id_to_filename = {}
        for tid, filename in tableid_to_filename.items():
            table_id_to_filename[tid] = filename
        
        result = {
            "query_model": model_id,
            "query_tables": query_tables,
            "searched_tables": retrieved_filenames,  # tables used for model mapping (after self-filter)
            "model_ids": final_results,
            "intermediate": {
                "retrieved_table_ids": similar_table_ids,
                "retrieved_table_filenames": retrieved_filenames,
                "table_id_to_filename": table_id_to_filename,
                "table_to_models": table_to_models
            }
        }
        os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            import json
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved to {output_json}")
    
    return final_results


def search_card2tab2card_from_tables(
    model_id: str,
    table_search_results: Dict[str, List[str]],
    relationship_parquet: Optional[str] = None,
    schema_log_path: str = "data_citationlake/logs/parquet_schema.log",
    use_citationlake: bool = True,
    k: int = 10,
    modelcard_k: Optional[int] = None
) -> List[str]:
    """
    Search for model cards using pre-computed table search results.
    
    This is useful when you have already run table search and have the results.
    
    Args:
        model_id: Hugging Face model ID to search from
        table_search_results: Dictionary mapping CSV basename/path to list of similar CSV basenames/paths
        relationship_parquet: Optional path to relationship parquet file (fallback)
        schema_log_path: Path to parquet_schema.log (for get_from-style approach)
        use_citationlake: Whether to use get_from-style mapping (default: True)
        k: Legacy parameter for backward compatibility. If modelcard_k is None, uses k.
        modelcard_k: Maximum number of model card results to return (defaults to k if not provided)
    
    Returns:
        List of model IDs that have similar tables
    """
    # Handle backward compatibility
    if modelcard_k is None:
        modelcard_k = k
    parquet = relationship_parquet or DEFAULT_RELATIONSHIP_PARQUET
    if not os.path.exists(parquet):
        raise FileNotFoundError(f"relationship_parquet not found: {parquet}")
    query_tables = get_tables_for_model(model_id=model_id, relationship_parquet=parquet)
    query_tables = _filter_s2orc_tables(query_tables)
    if not query_tables:
        print(f"Warning: No tables found for model {model_id}")
        return []
    
    # Collect all retrieved tables
    retrieved_tables: Set[str] = set()
    for query_table in query_tables:
        # Try both full path and basename
        table_key = query_table
        table_basename = os.path.basename(str(query_table))
        
        if table_key in table_search_results:
            retrieved_tables.update(table_search_results[table_key])
        if table_basename in table_search_results:
            retrieved_tables.update(table_search_results[table_basename])
    
    if not retrieved_tables:
        return []
    
    # Map retrieved tables to model IDs via relationship parquet
    similar_model_ids = set()
    table_basenames = [os.path.basename(str(t)) for t in retrieved_tables]
    basename_to_models = get_modelids_for_basenames_duckdb(parquet, table_basenames)
    for bn in table_basenames:
        similar_model_ids.update(basename_to_models.get(bn, []))

    # Remove the query model itself
    similar_model_ids = [mid for mid in similar_model_ids if mid != model_id]
    
    # Limit to top modelcard_k only if modelcard_k > 0 (0 = no limit)
    if modelcard_k and modelcard_k > 0:
        final_results = list(similar_model_ids)[:modelcard_k]
    else:
        final_results = list(similar_model_ids)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 Final Results Summary")
    print(f"{'='*60}")
    print(f"✅ Query Model: {model_id}")
    print(f"✅ Found {len(final_results)} similar model cards" + (f" (top {modelcard_k})" if modelcard_k and modelcard_k > 0 else ""))
    if final_results:
        print(f"📝 Sample results (showing first 2):")
        for i, model_id_result in enumerate(final_results[:2], 1):
            print(f"   {i}. {model_id_result}")
        if len(final_results) > 2:
            print(f"   ... and {len(final_results) - 2} more model cards")
    print(f"{'='*60}\n")
    
    return final_results


def search_card2tab2card_by_type(
    model_id: str,
    relationship_parquet: Optional[str] = None,
    query: Optional[Any] = None,
    search_type: str = "single_column",
    k: int = 10,
    table_search_k: Optional[int] = None,
    modelcard_k: Optional[int] = None,
    schema_log_path: str = "data_citationlake/logs/parquet_schema.log",
    use_citationlake: bool = True,
    output_json: str = "data/card2tab2card_by_type_results.json",
    db_path: Optional[str] = None,
    classification_json: Optional[str] = None,
    classifications: Optional[Dict[int, str]] = None,
    global_table_topk: bool = True,
) -> List[str]:
    """
    Search for model cards via table search with classification filtering.
    
    This is similar to search_card2tab2card but uses tab2tab_by_type which filters
    search results to only include tables with the same classification as the query table.
    
    Pipeline: Query -> ModelCard -> Tables -> Classify -> Retrieved Tables (filtered by type) -> Corresponding ModelCards
    
    Process:
    1. Get tables for the query model using a get_from-style function (or relationship parquet)
    2. Classify the query table
    3. Use tab2tab_by_type to search for similar tables (filtered by classification)
    4. Map retrieved tables back to model cards using a get_from-style function (or relationship)
    
    Args:
        model_id: Hugging Face model ID to search from
        relationship_parquet: Optional path to relationship parquet file (fallback if get_from-style approach not available)
        query: Optional query data for table search. If None, uses tables from the model
        search_type: Type of table search - "single_column", "multi_column", "keyword", or "unionable"
        k: Legacy parameter for backward compatibility. If table_search_k or modelcard_k are None, uses k for both.
        table_search_k: Number of table results to retrieve (defaults to k if not provided)
        modelcard_k: Number of final model card results to return (defaults to k if not provided)
        schema_log_path: Path to parquet_schema.log (for get_from-style approach)
        use_citationlake: Whether to use get_from-style mapping (default: True)
        output_json: Optional path to save results as JSON
        db_path: Path to modellake.db (default: data_citationlake/modellake.db)
        classification_json: Path to JSON file with pre-computed classifications
        classifications: Optional pre-loaded classifications dictionary (tableid -> label)
    
    Returns:
        List of model IDs that have similar tables (filtered by classification)
    """
    # Handle backward compatibility: if table_search_k or modelcard_k are None, use k
    if table_search_k is None:
        table_search_k = k
    if modelcard_k is None:
        modelcard_k = k
    
    # Print pipeline overview
    import sys
    sys.stdout.flush()
    print(f"\n{'='*60}")
    print(f"🔍 Card2Tab2Card Search Pipeline (by Type)")
    print(f"{'='*60}")
    print(f"Pipeline: Query -> ModelCard -> Tables -> Classify -> Retrieved Tables (filtered) -> Corresponding ModelCards")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    
    # Resolve relationship parquet and get tables for the query model
    if relationship_parquet is None:
        for p in (DEFAULT_RELATIONSHIP_PARQUET, "data_citationlake/processed/modelcard_step3.parquet"):
            if os.path.exists(p):
                relationship_parquet = p
                break
        if relationship_parquet is None:
            relationship_parquet = DEFAULT_RELATIONSHIP_PARQUET
    if not os.path.exists(relationship_parquet):
        raise FileNotFoundError(f"relationship_parquet not found: {relationship_parquet}")
    query_tables = get_tables_for_model(model_id=model_id, relationship_parquet=relationship_parquet)
    query_tables = _filter_s2orc_tables(query_tables)
    if not query_tables:
        print(f"⚠️  Warning: No tables found for model {model_id}")
        if output_json:
            result = {
                "query_model": model_id,
                "query_tables": [],
                "model_ids": [],
                "intermediate": {
                    "retrieved_table_ids": [],
                    "retrieved_table_filenames": [],
                    "table_id_to_filename": {},
                    "table_to_models": {}
                }
            }
            os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
            with open(output_json, 'w', encoding='utf-8') as f:
                import json
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ Results saved to {output_json}")
        return []
    
    # Step 1: Query -> ModelCard -> Tables
    print(f"\n{'='*60}")
    print(f"📊 Step 1: Query -> ModelCard -> Tables")
    print(f"{'='*60}")
    print(f"✅ Query Model ID: {model_id}")
    print(f"✅ Found {len(query_tables)} tables for model {model_id}")
    
    # If query is provided, use it; otherwise search based on query_tables
    use_per_table_search = (
        query is None
        and global_table_topk
        and len(query_tables) > 1
        and search_type in ("keyword", "single_column")
    )
    if query is None:
        # Use the query model's tables as the search query
        if use_per_table_search:
            print(f"ℹ️  global_table_topk=True: each of {len(query_tables)} tables as equivalent query, merge → global top-{table_search_k}")
        if search_type == "keyword":
            all_headers = []
            for table_path in query_tables[:10]:
                csv_path = _resolve_csv_path(table_path)
                if csv_path:
                    all_headers.extend(_get_csv_headers(csv_path))
            query = list(set(all_headers))
            if not query:
                query = [os.path.basename(str(t)) for t in query_tables[:10]]
        else:
            query = None
            search_type = "keyword"
    
    # Step 2: Use tab2tab_by_type to search for similar tables (with classification filtering)
    print(f"\n{'='*60}")
    print(f"🔍 Step 2: Tables -> Retrieved Tables (with classification filtering)")
    print(f"{'='*60}")
    print(f"✅ Search type: {search_type}")
    print(f"✅ Table Search Top K: {table_search_k}")
    print(f"✅ ModelCard Top K: {modelcard_k}")
    
    # Default db_path
    if db_path is None:
        db_path = "data_citationlake/modellake.db"
    print(f"✅ Database: {db_path}")
    
    # Default classification_json
    if classification_json is None:
        classification_json = "data/table_classifications.json"
    
    # Initialize variables
    similar_table_ids = []
    tableid_to_filename = {}
    retrieved_filenames = []
    
    print(f"🔎 Getting search_table2table_by_type function...")
    sys.stdout.flush()
    search_table2table_by_type = _get_search_table2table_by_type()
    print(f"✅ Got search_table2table_by_type function")
    sys.stdout.flush()
    if use_per_table_search:
        num_seed_tables = len(query_tables[:20])
        k_per_table = max(1, table_search_k)
        k_request = k_per_table + 4  # Over-fetch: top1 -> request 5
        print(f"📊 Per-table k={k_per_table} | request={k_request} (k+4) | flow: filter per table → top-{k_per_table} → merge (no post-filter)")
        print(f"🔎 [STEP 2a] Parallel per-table search (by_type)...")
        sys.stdout.flush()

        def _search_one_by_type(args):
            ti, table_path, st, db, cj, cl = args
            csv_path = _resolve_csv_path(table_path)
            if not csv_path:
                return None
            tquery = _get_table_query(csv_path, table_path, st)
            if not tquery:
                return None
            t0 = time.time()
            ids = search_table2table_by_type(tquery, st, k_request, db_path=db,
                classification_json=cj, classifications=cl)
            elapsed = time.time() - t0
            bn = os.path.basename(str(table_path))
            ts = time.strftime("%H:%M:%S")
            print(f"   [Table {ti+1}] @{ts} INPUT={bn} k_request={k_request} | OUTPUT={len(ids) if ids else 0} ids | {elapsed:.1f}s")
            sys.stdout.flush()
            return (ids, bn) if ids else None

        tables_to_search = [(ti, tp, search_type, db_path, classification_json, classifications) for ti, tp in enumerate(query_tables[:20])]
        max_workers = min(8, len(tables_to_search))
        results_by_ti = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_search_one_by_type, a): a[0] for a in tables_to_search}
            for fut in as_completed(futures):
                ti = futures[fut]
                res = fut.result()
                if res:
                    results_by_ti[ti] = res
        per_table_results = [results_by_ti[ti] for ti in sorted(results_by_ti.keys())]
        # Flow: filter per table → top-k per table → merge (no post-filter)
        import duckdb
        all_ids = [tid for ids, _ in per_table_results for tid in ids]
        tableid_to_filename: Dict[int, str] = {}
        if all_ids:
            with duckdb.connect(db_path, read_only=True) as con_filter:
                table_ids_str = ','.join(str(tid) for tid in all_ids)
                filename_query = f"""
                    SELECT DISTINCT tableid, filename
                    FROM modellake_index
                    WHERE tableid IN ({table_ids_str}) AND rowid = -1
                """
                filename_results = con_filter.execute(filename_query).fetchall()
                tableid_to_filename = {tid: filename for tid, filename in filename_results}
        per_table_filtered_topk: List[List[Tuple[int, str]]] = []
        n_self, n_s2orc, n_generic = 0, 0, 0
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
            per_table_filtered_topk.append(kept[:k_per_table])
        if n_self or n_s2orc or n_generic:
            print(f"   [STEP 2b] Filter per table: self={n_self}, s2orc={n_s2orc}, generic={n_generic} | top-k={k_per_table}/table")
        seen = set()
        similar_table_data = []
        max_len = max(len(r) for r in per_table_filtered_topk) if per_table_filtered_topk else 0
        for rank in range(max_len):
            for row in per_table_filtered_topk:
                if rank < len(row) and row[rank][0] not in seen:
                    seen.add(row[rank][0])
                    similar_table_data.append(row[rank])
        similar_table_ids = [tid for tid, _ in similar_table_data]
        print(f"   [STEP 2a] Done: {len(per_table_results)} tables | filter→top{k_per_table}→merge: {len(similar_table_data)}")
    else:
        print(f"🔎 Searching for similar tables (with classification filtering)...")
        sys.stdout.flush()
        # Handle correlation search specially
        if search_type == "correlation" and isinstance(query, pd.DataFrame):
            source_col = query[query.columns[0]].astype(str).tolist()
            numeric_cols = query.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                target_col = query[numeric_cols[0]].tolist()
                print(f"   Correlation: source='{query.columns[0]}', target='{numeric_cols[0]}'")
                similar_table_ids = search_table2table_by_type(
                    query, search_type, table_search_k, db_path=db_path,
                    classification_json=classification_json,
                    classifications=classifications,
                    source_column=source_col, target_column=target_col
                )
            else:
                print(f"⚠️  No numeric column found for correlation search, skipping...")
                similar_table_ids = []
        else:
            similar_table_ids = search_table2table_by_type(
                query, search_type, table_search_k, db_path=db_path,
                classification_json=classification_json,
                classifications=classifications
            )
        print(f"✅ Found {len(similar_table_ids)} retrieved tables (table IDs, filtered by classification)")
    sys.stdout.flush()
    
    if not similar_table_ids:
        print(f"⚠️  No tables retrieved, cannot proceed to Step 3")
        if output_json:
            result = {
                "query_model": model_id,
                "query_tables": query_tables,
                "model_ids": [],
                "intermediate": {
                    "retrieved_table_ids": [],
                    "retrieved_table_filenames": [],
                    "table_id_to_filename": {},
                    "table_to_models": {}
                }
            }
            os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
            with open(output_json, 'w', encoding='utf-8') as f:
                import json
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ Results saved to {output_json}")
        return []
    
    # Get filenames for retrieved tables from database
    import duckdb
    with duckdb.connect(db_path, read_only=True) as con:
        table_ids_str = ','.join(str(tid) for tid in similar_table_ids)
        filename_query = f"""
            SELECT DISTINCT tableid, filename 
            FROM modellake_index 
            WHERE tableid IN ({table_ids_str}) AND rowid = -1
        """
        filename_results = con.execute(filename_query).fetchall()
        tableid_to_filename = {tid: filename for tid, filename in filename_results}
        if not (use_per_table_search and per_table_results):
            seed_basenames = {os.path.basename(str(t)) for t in query_tables}
            filtered = [tid for tid in similar_table_ids if tid in tableid_to_filename
                and os.path.basename(str(tableid_to_filename[tid])) not in seed_basenames
                and _classify_table_source_by_basename(os.path.basename(str(tableid_to_filename[tid]))) != "llm"
                and not _is_generic_table(tableid_to_filename[tid])]
            similar_table_ids = filtered[:table_search_k]
        seen_filenames = set()
        retrieved_filenames = []
        for tid in similar_table_ids:
            if tid in tableid_to_filename:
                fname = tableid_to_filename[tid]
                if fname not in seen_filenames:
                    seen_filenames.add(fname)
                    retrieved_filenames.append(fname)
        n_before = len(retrieved_filenames)
        retrieved_filenames = [f for f in retrieved_filenames if not _is_generic_table(f)]
        if n_before > len(retrieved_filenames):
            print(f"ℹ️  Filtered {n_before - len(retrieved_filenames)} generic tables (1910.09700_table, 204823751_table)")
        print(f"✅ Retrieved {len(retrieved_filenames)} unique filenames from database (ordered by Tab2Tab relevance)")
        print(f"📋 query_tables ({len(query_tables)}): {[os.path.basename(str(t)) for t in query_tables[:5]]}{'...' if len(query_tables) > 5 else ''}")
        print(f"📋 searched_tables ({len(retrieved_filenames)}): {[os.path.basename(str(f)) for f in retrieved_filenames[:5]]}{'...' if len(retrieved_filenames) > 5 else ''}")

    # Step 3: Retrieved Tables -> Corresponding ModelCards (same as search_card2tab2card)
    print(f"\n{'='*60}")
    print(f"🔄 Step 3: Retrieved Tables -> Corresponding ModelCards")
    print(f"{'='*60}")
    print(f"✅ Mapping {len(similar_table_ids)} retrieved tables to model cards...")
    
    # Map similar tables back to model cards
    similar_model_ids = set()
    table_to_models = {}
    models_raw_count = 0

    table_basenames = [os.path.basename(fname) for fname in retrieved_filenames]
    print(f"📋 Using relationship_parquet to map tables to model cards...")
    basename_to_models = get_modelids_for_basenames_duckdb(relationship_parquet, table_basenames)
    for filename in retrieved_filenames:
        basename = os.path.basename(filename)
        matched_models = basename_to_models.get(basename, [])
        if matched_models:
            models_raw_count += len(matched_models)
            similar_model_ids.update(matched_models)
            table_to_models[filename] = matched_models
    print(f"✅ Matched {len(similar_model_ids)} model cards from relationship data")
    
    # Per-table model count (value_counts): which tables span how many models
    if table_to_models:
        sorted_tables = sorted(table_to_models.keys(), key=lambda x: len(table_to_models[x]), reverse=True)
        print(f"📊 Per-table model counts (table -> #models, sorted desc):")
        for fname in sorted_tables:
            bn = os.path.basename(fname)
            n = len(table_to_models[fname])
            print(f"   {bn}: {n} models")
    
    # Remove the query model itself
    print(f"📊 Model set dedup: {models_raw_count} raw (sum over tables) → {len(similar_model_ids)} unique models (before excl. query)")
    similar_model_ids = [mid for mid in similar_model_ids if mid != model_id]
    
    # Also remove query model from table_to_models
    for filename in table_to_models:
        table_to_models[filename] = [mid for mid in table_to_models[filename] if mid != model_id]
    
    print(f"✅ Found {len(similar_model_ids)} unique model cards (excluding query model)")
    
    # Limit to top modelcard_k only if modelcard_k > 0 (0 = no limit)
    if modelcard_k and modelcard_k > 0:
        final_results = list(similar_model_ids)[:modelcard_k]
    else:
        final_results = list(similar_model_ids)
    
    # One-line TopK decision: seed_tables | searched_tables | model_cards
    print(f"📊 TopK decision: seed_tables={len(query_tables)} | searched_tables={len(similar_table_ids)} | model_cards={len(final_results)}")
    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 Final Results Summary")
    print(f"{'='*60}")
    print(f"✅ Query Model: {model_id}")
    print(f"✅ Found {len(final_results)} similar model cards" + (f" (top {modelcard_k}, filtered by classification)" if modelcard_k and modelcard_k > 0 else " (filtered by classification)"))
    if final_results:
        print(f"📝 Sample results (showing first 2):")
        for i, model_id_result in enumerate(final_results[:2], 1):
            print(f"   {i}. {model_id_result}")
        if len(final_results) > 2:
            print(f"   ... and {len(final_results) - 2} more model cards")
    print(f"{'='*60}\n")
    
    # Save if requested
    if output_json:
        table_id_to_filename = {}
        for tid, filename in tableid_to_filename.items():
            table_id_to_filename[tid] = filename
        
        result = {
            "query_model": model_id,
            "query_tables": query_tables,
            "searched_tables": retrieved_filenames,  # tables used for model mapping (after self-filter)
            "model_ids": final_results,
            "intermediate": {
                "retrieved_table_ids": similar_table_ids,
                "retrieved_table_filenames": retrieved_filenames,
                "table_id_to_filename": table_id_to_filename,
                "table_to_models": table_to_models
            }
        }
        os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            import json
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved to {output_json}")
    
    return final_results


def main():
    """CLI entry point for card2tab2card search"""
    parser = argparse.ArgumentParser(description="Card to Tab to Card Search")
    parser.add_argument('--model_id', required=True,
                       help='Hugging Face model ID to search from')
    parser.add_argument('--relationship_parquet', default='data_citationlake/processed/modelcard_step3_dedup.parquet',
                       help='Path to relationship parquet file (default: data_citationlake/processed/modelcard_step3_dedup.parquet)')
    parser.add_argument('--schema_log', default='data_citationlake/logs/parquet_schema.log',
                       help='Path to parquet_schema.log (for get_from-style approach)')
    parser.add_argument('--query', default=None,
                       help='Query data for table search. For mode=single: comma-separated for single_column/keyword, CSV path for multi_column/unionable. For mode=all: CSV file path (required). If None, uses model tables.')
    parser.add_argument('--search_type', choices=['single_column', 'multi_column', 'keyword', 'unionable'],
                       default='keyword',
                       help='Type of table search')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of table results to retrieve (table_search_k)')
    parser.add_argument('--modelcard_k', type=int, default=0,
                       help='Max model cards to return (0 = no limit: all models that contain the retrieved tables)')
    parser.add_argument('--table_search_json', default=None,
                       help='Optional: Path to pre-computed table search results JSON')
    parser.add_argument('--use_citationlake', action='store_true', default=True,
                       help='Use get_from-style mapping approach (default: True)')
    parser.add_argument('--no_citationlake', dest='use_citationlake', action='store_false',
                       help='Disable get_from-style approach, use relationship_parquet instead')
    parser.add_argument('--output_json', default='data/card2tab2card_results.json',
                       help='Path to save results as JSON (default: data/card2tab2card_results.json)')
    parser.add_argument('--output_folder', default='data',
                       help='Output folder for results when mode=all (default: data)')
    parser.add_argument('--mode', choices=['single', 'all', 'by_type'], default='single',
                       help='Search mode: single (one search type), all (run all three: single_column, keyword, unionable), or by_type (with classification filtering)')
    parser.add_argument('--db_path', default='data_citationlake/modellake.db',
                       help='Path to modellake.db (default: data_citationlake/modellake.db)')
    parser.add_argument('--classification_json', default='data/table_classifications.json',
                       help='Path to JSON file with pre-computed classifications (required for by_type mode)')
    
    args = parser.parse_args()
    
    try:
        start_time = time.time()
        # If mode=all, run all three search types
        if args.mode == 'all':
            print(f"\n{'='*60}")
            print(f"🚀 Running ALL search modes")
            print(f"{'='*60}")
            print(f"Output folder: {args.output_folder}")
            print(f"Will run: single_column, keyword, unionable")
            print(f"{'='*60}\n")
            
            # For mode=all, query must be a CSV file path
            if not args.query:
                print(f"❌ Error: --query is required when --mode=all")
                print(f"   Please provide a CSV file path, e.g., --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv")
                sys.exit(1)
            
            if not os.path.exists(args.query):
                print(f"❌ Error: CSV file not found: {args.query}")
                sys.exit(1)
            
            print(f"✅ Using CSV file as query: {args.query}")
            
            # Load CSV once for all search types
            query_df = pd.read_csv(args.query)
            print(f"✅ Loaded CSV with {len(query_df)} rows and {len(query_df.columns)} columns")
            
            # Ensure output folder exists
            os.makedirs(args.output_folder, exist_ok=True)
            
            # Define search configurations
            search_configs = [
                {
                    'type': 'single_column',
                    'output': os.path.join(args.output_folder, 'card2tab2card_singlecol_results.json')
                },
                {
                    'type': 'keyword',
                    'output': os.path.join(args.output_folder, 'card2tab2card_keyword_results.json')
                },
                {
                    'type': 'unionable',
                    'output': os.path.join(args.output_folder, 'card2tab2card_unionable_results.json')
                }
            ]
            
            all_results = {}
            
            for config in search_configs:
                search_type = config['type']
                output_path = config['output']
                
                print(f"\n{'='*60}")
                print(f"🔄 Running {search_type} search...")
                print(f"{'='*60}")
                
                # Parse query based on search type (all from the same CSV)
                query = None
                if search_type == 'single_column':
                    # Use values from first column (consistent with Blend_internal README examples)
                    # From README: Seekers.SC(dataset[clm_name], k) - uses single column
                    # From ComplexSearch: Seekers.SC(examples[examples.columns[0]], k) - uses first column
                    first_col = query_df.columns[0]
                    query = query_df[first_col].dropna().astype(str).tolist()
                    print(f"✅ Using first column '{first_col}' with {len(query)} values for single_column search")
                    print(f"   Sample values: {query[:3]}{'...' if len(query) > 3 else ''}")
                elif search_type == 'keyword':
                    # Use headers (column names) - consistent with Blend_internal
                    # Blend_internal uses rowid=-1 which represents headers in the index
                    headers = [str(col).lower().strip() for col in query_df.columns]
                    headers = [h for h in headers if h]  # Filter empty headers
                    query = headers
                    print(f"✅ Using {len(headers)} headers for keyword search: {headers[:5]}{'...' if len(headers) > 5 else ''}")
                elif search_type == 'unionable':
                    # Use the entire DataFrame
                    query = query_df
                    print(f"✅ Using entire DataFrame ({len(query)} rows, {len(query.columns)} columns) for unionable search")
                
                try:
                    results = search_card2tab2card(
                        model_id=args.model_id,
                        relationship_parquet=args.relationship_parquet,
                        query=query,
                        search_type=search_type,
                        k=args.k,
                        schema_log_path=args.schema_log,
                        use_citationlake=args.use_citationlake,
                        output_json=output_path,
                        db_path=args.db_path
                    )
                    all_results[search_type] = {
                        'count': len(results),
                        'results': results,
                        'output': output_path
                    }
                    print(f"✅ {search_type} search completed: {len(results)} results saved to {output_path}")
                except Exception as e:
                    print(f"❌ Error in {search_type} search: {e}")
                    all_results[search_type] = {
                        'count': 0,
                        'results': [],
                        'output': output_path,
                        'error': str(e)
                    }
            
            # Summary
            print(f"\n{'='*60}")
            print(f"📊 Summary of ALL search modes")
            print(f"{'='*60}")
            for search_type, result_info in all_results.items():
                if 'error' in result_info:
                    print(f"❌ {search_type}: Error - {result_info['error']}")
                else:
                    print(f"✅ {search_type}: {result_info['count']} results -> {result_info['output']}")
            print(f"{'='*60}\n")
        
        elif args.mode == 'by_type':
            # By type mode - use classification filtering
            print(f"\n{'='*60}")
            print(f"🚀 Running Card2Tab2Card Search (by Type)")
            print(f"{'='*60}")
            print(f"Mode: Classification-filtered search")
            print(f"Classification JSON: {args.classification_json}")
            print(f"{'='*60}\n")
            
            # Parse query based on search type (if provided)
            query = None
            if args.query:
                if args.search_type == 'single_column':
                    query = [x.strip() for x in args.query.split(',')]
                elif args.search_type == 'multi_column':
                    query = pd.read_csv(args.query)
                elif args.search_type == 'unionable':
                    query = pd.read_csv(args.query)
                elif args.search_type == 'keyword':
                    query = [x.strip() for x in args.query.split(',')]
            
            results = search_card2tab2card_by_type(
                model_id=args.model_id,
                relationship_parquet=args.relationship_parquet,
                query=query,
                search_type=args.search_type,
                k=args.k,
                schema_log_path=args.schema_log,
                use_citationlake=args.use_citationlake,
                output_json=args.output_json or "data/card2tab2card_by_type_results.json",
                db_path=args.db_path,
                classification_json=args.classification_json
            )
            
            print(f"Found {len(results)} similar model cards for {args.model_id} (filtered by classification):")
            for i, model_id in enumerate(results, 1):
                print(f"  {i}. {model_id}")
            
        else:
            # Single mode - original behavior
            # If table_search_json is provided, use pre-computed results
            if args.table_search_json:
                import json
                with open(args.table_search_json, 'r') as f:
                    table_search_results = json.load(f)
                results = search_card2tab2card_from_tables(
                    model_id=args.model_id,
                    table_search_results=table_search_results,
                    relationship_parquet=args.relationship_parquet,
                    schema_log_path=args.schema_log,
                    use_citationlake=args.use_citationlake,
                    k=args.k
                )
            else:
                # Parse query based on search type (if provided)
                query = None
                if args.query:
                    if args.search_type == 'single_column':
                        query = [x.strip() for x in args.query.split(',')]
                    elif args.search_type == 'multi_column':
                        query = pd.read_csv(args.query)
                    elif args.search_type == 'unionable':
                        query = pd.read_csv(args.query)
                    elif args.search_type == 'keyword':
                        query = [x.strip() for x in args.query.split(',')]
                
                results = search_card2tab2card(
                    model_id=args.model_id,
                    relationship_parquet=args.relationship_parquet,
                    query=query,
                    search_type=args.search_type,
                    k=args.k,
                    table_search_k=args.k,
                    modelcard_k=args.modelcard_k,
                    schema_log_path=args.schema_log,
                    use_citationlake=args.use_citationlake,
                    output_json=args.output_json or "data/card2tab2card_results.json",
                    db_path=args.db_path
                )
            
            print(f"Found {len(results)} similar model cards for {args.model_id}:")
            for i, model_id in enumerate(results, 1):
                print(f"  {i}. {model_id}")

        elapsed = time.time() - start_time
        from src.utils import get_device
        print(f"\nTotal time: {elapsed:.2f}s (device: {get_device()})")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    import sys
    
    # If running as test (python src/search/card2tab2card.py test), run test cases
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Import directly from classification module to avoid __init__.py dependencies
        import importlib.util
        classification_path = os.path.join(os.path.dirname(__file__), 'classification.py')
        spec = importlib.util.spec_from_file_location("classification", classification_path)
        classification_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(classification_module)
        
        print("=" * 60)
        print("Running Card2Tab2Card by Type Test Cases")
        print("=" * 60)
        
        # Test 1: Test classification loading
        print("\n[Test 1] Test classification loading")
        print("-" * 60)
        import tempfile
        import json as json_module
        
        temp_class_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        mock_classifications = {
            "1": "numerical",
            "2": "numerical",
            "3": "categorical"
        }
        json_module.dump(mock_classifications, temp_class_file)
        temp_class_file.close()
        
        try:
            classifications = classification_module.load_classifications(temp_class_file.name)
            print(f"✅ Loaded {len(classifications)} classifications")
            assert len(classifications) == 3, f"Expected 3 classifications, got {len(classifications)}"
        finally:
            os.unlink(temp_class_file.name)
        
        # Test 2: Test table classification
        print("\n[Test 2] Test table classification")
        print("-" * 60)
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        # Use heuristic method for testing to avoid tab2know dependency
        classification = classification_module.classify_table(test_df, method="heuristic")
        print(f"✅ Test DataFrame classification: {classification}")
        assert classification == "numerical", f"Expected 'numerical', got '{classification}'"
        
        # Test 3: Test get_tables_by_classification
        print("\n[Test 3] Test get_tables_by_classification")
        print("-" * 60)
        test_classifications = {1: "numerical", 2: "numerical", 3: "categorical"}
        numerical_tables = classification_module.get_tables_by_classification("numerical", test_classifications)
        print(f"✅ Numerical tables: {numerical_tables}")
        assert set(numerical_tables) == {1, 2}, f"Expected [1, 2], got {numerical_tables}"
        
        print("\n" + "=" * 60)
        print("✅ All test cases passed!")
        print("Note: Full integration tests require:")
        print("  - modellake.db")
        print("  - Classification JSON file")
        print("  - Valid model_id and query data")
        print("=" * 60)
    else:
        main()

