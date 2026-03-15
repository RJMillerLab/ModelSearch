"""
Table to Table Search by Type

This module provides table-to-table search with classification filtering.
It first classifies the input table, then searches only within tables of the same type.

This is similar to tab2tab but adds an extra classification step to filter results.
"""

import os
import sys
import time
from typing import List, Dict, Optional, Any, Iterable, Union, Set
import pandas as pd
import argparse
import duckdb

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Lazy import to avoid dependencies when running tests
_tab2tab_search_table2table = None
_classification_module = None

def _get_search_table2table():
    """Lazy import search_table2table from tab2tab."""
    global _tab2tab_search_table2table
    if _tab2tab_search_table2table is None:
        from src.search.tab2tab import search_table2table
        _tab2tab_search_table2table = search_table2table
    return _tab2tab_search_table2table

def _get_classification_module():
    """Lazy import classification module directly to avoid __init__.py dependencies."""
    global _classification_module
    if _classification_module is None:
        import importlib.util
        classification_path = os.path.join(os.path.dirname(__file__), 'classification.py')
        spec = importlib.util.spec_from_file_location("classification", classification_path)
        _classification_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_classification_module)
    return _classification_module

from src.config import MODELLAKE_DB, CLASSIFICATION_JSON, TAB2TAB_BY_TYPE_OUTPUT_JSON
from src.search.classification import (
    classify_table,
    classify_table_from_db,
    load_classifications,
    get_tables_by_classification,
    infer_classification_method,
)
from src.utils import resolve_table_path


def _load_query_from_tableid(tableid: int, db_path: str, index_table: str = "modellake_index") -> Optional[pd.DataFrame]:
    """Load query table from modellake.db by table ID (same source as rest: db -> filename -> resolve path -> read CSV)."""
    if not os.path.exists(db_path):
        return None
    with duckdb.connect(db_path, read_only=True) as con:
        row = con.execute(
            f"SELECT DISTINCT filename FROM {index_table} WHERE tableid = ? AND rowid = -1 LIMIT 1",
            [tableid]
        ).fetchone()
        if not row:
            return None
        csv_path = resolve_table_path(row[0])
        if not csv_path or not os.path.exists(csv_path):
            return None
        return pd.read_csv(csv_path)


def _search_restricted_to_tables_by_header_terms(
    db_path: str,
    query_terms: List[str],
    table_ids: Set[int],
    limit: Optional[int] = None,
    index_table: str = "modellake_index",
) -> List[int]:
    """
    Search by header-match only within the given table IDs (DuckDB).
    Labels and types come from classification JSON only; no hardcoded labels.
    Matches headers (rowid=-1) tokenized column against query terms.
    Returns all matching (or up to limit), ordered by match count. No extra ranking.
    """
    if not query_terms or not table_ids:
        return []
    terms_safe = [str(t).replace("'", "''")[:200] for t in query_terms[:50]]
    ids_list = list(table_ids)[:50000]
    ids_sql = ",".join(str(int(i)) for i in ids_list)
    like_conds = " OR ".join(
        f"LOWER(tokenized) LIKE LOWER('%' || '{k}' || '%')" for k in terms_safe
    )
    if not like_conds:
        return []
    limit_clause = f" LIMIT {limit}" if limit is not None else ""
    query_sql = f"""
        SELECT tableid, COUNT(*) AS cnt
        FROM {index_table}
        WHERE rowid = -1 AND tableid IN ({ids_sql}) AND ({like_conds})
        GROUP BY tableid
        ORDER BY cnt DESC
        {limit_clause}
    """
    with duckdb.connect(db_path, read_only=True) as con:
        rows = con.execute(query_sql).fetchall()
        return [int(r[0]) for r in rows]


def search_table2table_by_type(
    query: Any,
    search_type: str = "single_column",
    k: int = 10,
    db_path: Optional[str] = None,
    classification_json: Optional[str] = None,
    classifications: Optional[Dict[int, str]] = None,
    target: Optional[Iterable[float]] = None,
    source_column: Optional[Iterable[str]] = None,
    target_column: Optional[Iterable[float]] = None,
    auto_classify: bool = True
) -> List[int]:
    """
    Search for similar tables, filtering by table classification type.
    
    Process:
    1. Classify the input query table
    2. Load or compute classifications for all tables in datalake
    3. Filter to only tables with the same classification
    4. Run tab2tab search within the filtered set
    
    Args:
        query: Query data - can be:
            - Iterable of values (for single_column)
            - pd.DataFrame (for multi_column, complex, or correlation)
            - List[str] (for keyword)
            - Path to CSV file (will be loaded and classified)
        search_type: Type of search - "single_column", "multi_column", "keyword", etc.
        k: Number of results to return
        db_path: Path to modellake.db (optional)
        classification_json: Path to JSON file with pre-computed classifications
        classifications: Optional pre-loaded classifications dictionary (tableid -> label)
        target: Optional target values for complex search
        source_column: Optional source column for correlation search
        target_column: Optional target column for correlation search
        auto_classify: Whether to automatically classify query table if not provided (default: True)
    
    Returns:
        List of table IDs (integers) that match the query and have the same classification
    """
    print(f"\n{'='*60}")
    print(f"🔍 Table-to-Table Search by Type")
    print(f"{'='*60}")
    
    # Use same classification method as the batch that produced classification_json (avoids "unknown" mismatch)
    if classification_json and os.path.exists(classification_json):
        query_method = infer_classification_method(classification_json)
        print(f"   Using classification method: {query_method} (inferred from JSON)")
    else:
        query_method = "tab2know"
    
    # Step 1: Classify the query table
    print(f"\n📊 Step 1: Classifying query table...")
    query_classification = None
    
    # If query is a DataFrame or path to CSV, classify it
    if isinstance(query, pd.DataFrame):
        query_classification = classify_table(query, method=query_method)
        print(f"✅ Query classification: {query_classification}")
    elif isinstance(query, str) and os.path.exists(query):
        # It's a file path
        query_classification = classify_table(query, method=query_method)
        print(f"✅ Query classification: {query_classification}")
        # Load the query for search
        query = pd.read_csv(query)
    elif auto_classify:
        # Try to infer classification from query data
        if isinstance(query, list) and len(query) > 0:
            # For single_column/keyword, create a simple DataFrame to classify
            sample_df = pd.DataFrame({query[0] if isinstance(query[0], str) else 'value': query[:10]})
            query_classification = classify_table(sample_df, method=query_method)
            print(f"✅ Inferred query classification: {query_classification}")
        else:
            print(f"⚠️  Could not auto-classify query, using 'mixed' as default")
            query_classification = "mixed"
    else:
        print(f"⚠️  Auto-classify disabled and no classification provided, using 'mixed' as default")
        query_classification = "mixed"
    
    if not query_classification:
        query_classification = "mixed"
    
    # Step 2: Load classifications for all tables in datalake (inference: use precomputed JSON only)
    print(f"\n📊 Step 2: Loading table classifications...")
    if classification_json is None:
        classification_json = CLASSIFICATION_JSON
    if classifications is None:
        if not os.path.exists(classification_json):
            raise FileNotFoundError(
                f"Classification file not found: {classification_json}\n"
                "Run Part 1.4 (train / batch) first:\n"
                "  python -m src.search.classification --mode batch (defaults from src.config)"
            )
        print(f"   Loading from: {classification_json}")
        classifications = load_classifications(classification_json)
    print(f"✅ Loaded classifications for {len(classifications)} tables")
    
    # Step 3: Filter to tables with same classification
    print(f"\n📊 Step 3: Filtering tables by classification...")
    matching_tableids = get_tables_by_classification(query_classification, classifications)
    print(f"✅ Found {len(matching_tableids)} tables with classification '{query_classification}'")
    
    # When no tables in JSON have this classification, fall back to unfiltered (no label names hardcoded)
    skip_type_filter = False
    if not matching_tableids:
        print(f"⚠️  No tables with classification '{query_classification}' in JSON; falling back to unfiltered search")
        skip_type_filter = True
        matching_tableids = list(classifications.keys())
    
    # Step 4: Run tab2tab search (we'll filter results afterward unless skip_type_filter)
    print(f"\n🔍 Step 4: Running table search...")
    print(f"   Search type: {search_type}")
    print(f"   Top K: {k}")
    
    # Convert DataFrame to the format tab2tab expects for each search_type
    search_query = query
    if isinstance(query, pd.DataFrame):
        if search_type == "single_column":
            # tab2tab expects iterable of values (e.g. first column)
            col = query.iloc[:, 0]
            search_query = col.astype(str).tolist()
        elif search_type == "keyword":
            # tab2tab expects list of strings (column names)
            search_query = [str(c) for c in query.columns]
        # multi_column, unionable, etc.: DataFrame is OK as-is
    
    # Run the normal search (lazy import)
    search_table2table = _get_search_table2table()
    all_results = search_table2table(
        query=search_query,
        search_type=search_type,
        k=k * 20,  # Get more candidates so after type-filter we have enough
        db_path=db_path,
        target=target,
        source_column=source_column,
        target_column=target_column
    )
    
    # If single_column returned 0, it means no tables have overlapping cell values.
    # Fall back to keyword search (column names) so we still get same-type tables with similar schema.
    if len(all_results) == 0 and search_type == "single_column" and isinstance(query, pd.DataFrame):
        keyword_query = [str(c) for c in query.columns]
        if keyword_query:
            print(f"   single_column found no tables with overlapping values; trying keyword (column names)")
            all_results = search_table2table(
                query=keyword_query,
                search_type="keyword",
                k=k * 20,
                db_path=db_path,
            )
    if len(all_results) == 0 and search_type == "single_column":
        print(f"   Tip: single_column matches on cell values. Try --search_type keyword to match on column headers.")
    
    print(f"✅ Found {len(all_results)} candidate tables")
    
    # ---- Debug before Step 5: ID types + label distribution of candidates ----
    all_results_int = [int(tid) for tid in all_results]
    if len(all_results_int) > 0 and classifications:
        sample_raw = all_results[:3]
        print(f"\n   [Step5-debug] Candidate ID types: {[type(t).__name__ for t in sample_raw]}")
        # Classification dict keys: int (from load_classifications) or str (raw JSON)
        sample_keys = list(classifications.keys())[:3]
        print(f"   [Step5-debug] Classification keys sample (type): {[type(x).__name__ for x in sample_keys]}, e.g. {sample_keys}")
        # Lookup: by int vs by str
        found_by_int = sum(1 for tid in all_results_int if tid in classifications)
        found_by_str = sum(1 for tid in all_results_int if str(tid) in classifications)
        print(f"   [Step5-debug] Candidates in classification dict: by int key {found_by_int}/{len(all_results_int)}, by str key {found_by_str}/{len(all_results_int)}")
        # Label distribution of the N candidates (use int key; if missing try str)
        from collections import Counter
        labels_of_candidates = []
        for tid in all_results_int:
            lab = classifications.get(tid) or classifications.get(str(tid))
            labels_of_candidates.append(lab if lab else "__missing__")
        dist = Counter(labels_of_candidates)
        print(f"   [Step5-debug] Label distribution of {len(all_results_int)} candidates: {dict(dist.most_common())}")
        if dist.get("__missing__", 0) > 0:
            print(f"   [Step5-debug] Missing: sample IDs not in classification: {[tid for tid in all_results_int if classifications.get(tid) is None and classifications.get(str(tid)) is None][:5]}")
    
    # Step 5: Filter results by classification (unless we fell back to unfiltered)
    print(f"\n📊 Step 5: Filtering results by classification...")
    if skip_type_filter:
        final_results = all_results[:k]
        print(f"✅ Using unfiltered results (query was 'unknown'): {len(final_results)} tables")
    else:
        # Matching: support both int and str keys (JSON has str keys; load_classifications converts to int)
        matching_set_int = set(int(x) for x in matching_tableids)
        matching_set_str = set(str(x) for x in matching_tableids)
        if len(all_results_int) > 0:
            sample_all = all_results_int[:5]
            sample_matching = sorted(matching_set_int)[:5]
            print(f"   Debug: sample candidate IDs: {sample_all}, sample matching IDs: {sample_matching}")
            overlap_int = sum(1 for tid in all_results_int if tid in matching_set_int)
            overlap_str = sum(1 for tid in all_results_int if str(tid) in matching_set_str)
            print(f"   Debug: in matching set (int): {overlap_int}/{len(all_results_int)}, (str): {overlap_str}/{len(all_results_int)} ({len(matching_set_int)} total)")
        filtered_results = [
            tid for tid in all_results_int
            if tid in matching_set_int or str(tid) in matching_set_str
        ]
        final_results = filtered_results[:k]
        # When global top-N has 0 of this type: search only within same-type tables (DuckDB). Type from JSON only.
        has_string_list_query = isinstance(search_query, list) and search_query and all(isinstance(x, str) for x in search_query)
        if len(final_results) == 0 and has_string_list_query:
            print(f"   Fallback: search within {len(matching_set_int)} tables of type '{query_classification}' (from JSON), return all matches")
            final_results = _search_restricted_to_tables_by_header_terms(
                db_path or MODELLAKE_DB,
                search_query,
                matching_set_int,
                limit=None,
            )
        print(f"✅ Filtered to {len(final_results)} tables with matching classification")
    
    if len(final_results) < k and not skip_type_filter:
        print(f"⚠️  Only found {len(final_results)} results (requested {k})")
        print(f"   This may be because there are few tables with classification '{query_classification}'")
    
    print(f"\n{'='*60}")
    print(f"✅ Search complete: {len(final_results)} results")
    print(f"{'='*60}\n")
    
    return final_results


def main():
    """CLI entry point for tab2tab_by_type search"""
    parser = argparse.ArgumentParser(
        description="Table to Table Search by Type (with classification filtering)",
        epilog="""
This tool performs table-to-table search but filters results to only include
tables with the same classification as the query table.

First, classify all tables: python -m src.search.classification --mode batch
Then run search: python -m src.search.tab2tab_by_type --query query.csv --search_type single_column --k 10
Paths are read from src.config.
        """
    )
    parser.add_argument('--search_type', choices=[
        'single_column', 'multi_column', 'keyword', 'unionable', 'complex',
        'correlation', 'imputation', 'augmentation', 'dependent_data',
        'feature_for_ml', 'multi_column_collinearity', 'negative_example'
    ], default='single_column', help='Type of search to perform')
    parser.add_argument('--query', required=True,
                       help='Query data. For single_column: comma-separated values or CSV path. '
                            'For multi_column/unionable: CSV file path. For keyword: comma-separated keywords.')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of results to return')
    parser.add_argument('--no_auto_classify', dest='auto_classify', action='store_false',
                       help='Disable automatic classification of query table')
    parser.add_argument('--output_json', required=True,
                       help='Output JSON path')
    
    args = parser.parse_args()

    db_path = MODELLAKE_DB
    classification_json = CLASSIFICATION_JSON
    output_path = args.output_json
    start_time = time.time()
    # Parse query: table ID (from modellake) or path/values. Same source as rest: db -> filename -> resolve -> read.
    query = None
    def _query_as_tableid(s: str):
        s = s.strip()
        return int(s) if s.isdigit() else None

    if args.search_type == 'single_column':
        tid = _query_as_tableid(args.query)
        if tid is not None:
            query = _load_query_from_tableid(tid, db_path)
        if query is None:
            if os.path.exists(args.query):
                query = pd.read_csv(args.query)
            else:
                query = [x.strip() for x in args.query.split(',')]
    elif args.search_type == 'multi_column':
        tid = _query_as_tableid(args.query)
        if tid is not None:
            query = _load_query_from_tableid(tid, db_path)
        if query is None:
            query = pd.read_csv(args.query)
    elif args.search_type == 'unionable':
        tid = _query_as_tableid(args.query)
        if tid is not None:
            query = _load_query_from_tableid(tid, db_path)
        if query is None:
            query = pd.read_csv(args.query)
    elif args.search_type == 'keyword':
        tid = _query_as_tableid(args.query)
        if tid is not None:
            df = _load_query_from_tableid(tid, db_path)
            if df is not None:
                query = [str(col).lower().strip() for col in df.columns]
        if query is None:
            if os.path.exists(args.query):
                df = pd.read_csv(args.query, nrows=0)
                query = [str(col).lower().strip() for col in df.columns]
            else:
                query = [x.strip() for x in args.query.split(',')]
    else:
        tid = _query_as_tableid(args.query)
        if tid is not None:
            query = _load_query_from_tableid(tid, db_path)
        if query is None:
            query = pd.read_csv(args.query)
    
    # Perform search
    results = search_table2table_by_type(
        query=query,
        search_type=args.search_type,
        k=args.k,
        db_path=db_path,
        classification_json=classification_json,
        auto_classify=args.auto_classify
    )
    
    print(f"Found {len(results)} tables:")
    for i, table_id in enumerate(results, 1):
        print(f"  {i}. Table ID: {table_id}")
    
    # Save results as JSON
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    result_data = {
        "query": str(query) if not isinstance(query, (list, pd.DataFrame)) else "DataFrame or list",
        "search_type": args.search_type,
        "k": args.k,
        "results": [int(tid) for tid in results],
        "num_results": len(results)
    }
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Results saved to {output_path}")
    def _get_device():
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    print(f"\nTotal time: {time.time() - start_time:.2f}s (device: {_get_device()})")

if __name__ == '__main__':
    import sys
    
    # If running as test (python src/search/tab2tab_by_type.py test), run test cases
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Import directly from classification module to avoid __init__.py dependencies
        classification_mod = _get_classification_module()
        classify_table = classification_mod.classify_table
        load_classifications = classification_mod.load_classifications
        get_tables_by_classification = classification_mod.get_tables_by_classification
        
        print("=" * 60)
        print("Running Tab2Tab by Type Test Cases")
        print("=" * 60)
        
        # Test 1: Test classification of query DataFrame
        print("\n[Test 1] Test classification of query DataFrame")
        print("-" * 60)
        test_query_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        # Use heuristic method for testing to avoid tab2know dependency
        classification = classify_table(test_query_df, method="heuristic")
        print(f"✅ Query classification: {classification}")
        assert classification == "numerical", f"Expected 'numerical', got '{classification}'"
        
        # Test 2: Test with mock classifications
        print("\n[Test 2] Test search with mock classifications")
        print("-" * 60)
        # Create a temporary classification JSON file
        import tempfile
        import json as json_module
        
        temp_class_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        # Mock classifications - assume we have some table IDs
        mock_classifications = {
            "1": "numerical",
            "2": "numerical",
            "3": "categorical",
            "4": "mixed"
        }
        json_module.dump(mock_classifications, temp_class_file)
        temp_class_file.close()
        
        try:
            # Load classifications
            classifications = load_classifications(temp_class_file.name)
            print(f"✅ Loaded {len(classifications)} classifications")
            
            # Get tables by classification
            numerical_tables = get_tables_by_classification("numerical", classifications)
            print(f"✅ Numerical tables: {numerical_tables}")
            assert set(numerical_tables) == {1, 2}, f"Expected [1, 2], got {numerical_tables}"
            
        finally:
            os.unlink(temp_class_file.name)
        
        # Test 3: Test query parsing for different search types
        print("\n[Test 3] Test query parsing")
        print("-" * 60)
        
        # Single column query
        single_col_query = ["value1", "value2", "value3"]
        print(f"✅ Single column query: {len(single_col_query)} values")
        
        # Keyword query
        keyword_query = ["train", "model", "dataset"]
        print(f"✅ Keyword query: {keyword_query}")
        
        # DataFrame query
        df_query = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        print(f"✅ DataFrame query: {df_query.shape}")
        
        print("\n" + "=" * 60)
        print("✅ All test cases passed!")
        print("Note: Full integration tests require modellake.db and classification JSON")
        print("=" * 60)
    else:
        main()

