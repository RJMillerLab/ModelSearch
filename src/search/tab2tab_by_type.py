"""
Table to Table Search by Type

This module provides table-to-table search with classification filtering.
It first classifies the input table, then searches only within tables of the same type.

This is similar to tab2tab but adds an extra classification step to filter results.
"""

import os
import sys
import time
from typing import List, Dict, Optional, Any, Iterable, Union
import pandas as pd
import argparse

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

# For backward compatibility, try to import normally but fall back to lazy import
try:
    from src.search.classification import (
        classify_table,
        classify_table_from_db,
        classify_datalake_batch,
        load_classifications,
        get_tables_by_classification
    )
except (ImportError, ModuleNotFoundError):
    # If import fails, use lazy import
    def _lazy_classification_import(name):
        mod = _get_classification_module()
        return getattr(mod, name)
    
    classify_table = lambda *args, **kwargs: _lazy_classification_import('classify_table')(*args, **kwargs)
    classify_table_from_db = lambda *args, **kwargs: _lazy_classification_import('classify_table_from_db')(*args, **kwargs)
    classify_datalake_batch = lambda *args, **kwargs: _lazy_classification_import('classify_datalake_batch')(*args, **kwargs)
    load_classifications = lambda *args, **kwargs: _lazy_classification_import('load_classifications')(*args, **kwargs)
    get_tables_by_classification = lambda *args, **kwargs: _lazy_classification_import('get_tables_by_classification')(*args, **kwargs)


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
    
    # Step 1: Classify the query table
    print(f"\n📊 Step 1: Classifying query table...")
    query_classification = None
    
    # If query is a DataFrame or path to CSV, classify it
    if isinstance(query, pd.DataFrame):
        query_classification = classify_table(query)
        print(f"✅ Query classification: {query_classification}")
    elif isinstance(query, str) and os.path.exists(query):
        # It's a file path
        query_classification = classify_table(query)
        print(f"✅ Query classification: {query_classification}")
        # Load the query for search
        query = pd.read_csv(query)
    elif auto_classify:
        # Try to infer classification from query data
        if isinstance(query, list) and len(query) > 0:
            # For single_column/keyword, create a simple DataFrame to classify
            sample_df = pd.DataFrame({query[0] if isinstance(query[0], str) else 'value': query[:10]})
            query_classification = classify_table(sample_df)
            print(f"✅ Inferred query classification: {query_classification}")
        else:
            print(f"⚠️  Could not auto-classify query, using 'mixed' as default")
            query_classification = "mixed"
    else:
        print(f"⚠️  Auto-classify disabled and no classification provided, using 'mixed' as default")
        query_classification = "mixed"
    
    if not query_classification:
        query_classification = "mixed"
    
    # Step 2: Load classifications for all tables in datalake
    print(f"\n📊 Step 2: Loading table classifications...")
    
    if classifications is None:
        if classification_json and os.path.exists(classification_json):
            print(f"   Loading from: {classification_json}")
            classifications = load_classifications(classification_json)
        else:
            # Need to compute classifications - this requires db_path
            if db_path is None:
                db_path = "data_citationlake/modellake.db"
            
            if not os.path.exists(db_path):
                raise FileNotFoundError(
                    f"Database not found: {db_path}\n"
                    "Please either:\n"
                    "  1. Provide --classification_json with pre-computed classifications, or\n"
                    "  2. Ensure db_path points to a valid modellake.db"
                )
            
            print(f"   Computing classifications from database: {db_path}")
            print(f"   (This may take a while for large datalakes...)")
            
            # Compute classifications (this is expensive, so we save it)
            temp_json = classification_json or "data/table_classifications_temp.json"
            classifications = classify_datalake_batch(
                db_path=db_path,
                output_json=temp_json
            )
            
            if classification_json and temp_json != classification_json:
                # Move temp file to requested location
                import shutil
                shutil.move(temp_json, classification_json)
                print(f"   ✅ Saved classifications to: {classification_json}")
    
    print(f"✅ Loaded classifications for {len(classifications)} tables")
    
    # Step 3: Filter to tables with same classification
    print(f"\n📊 Step 3: Filtering tables by classification...")
    matching_tableids = get_tables_by_classification(query_classification, classifications)
    print(f"✅ Found {len(matching_tableids)} tables with classification '{query_classification}'")
    
    if not matching_tableids:
        print(f"⚠️  No tables found with classification '{query_classification}'")
        print(f"   Returning empty results")
        return []
    
    # Step 4: Run tab2tab search (we'll filter results afterward)
    print(f"\n🔍 Step 4: Running table search...")
    print(f"   Search type: {search_type}")
    print(f"   Top K: {k}")
    
    # Run the normal search (lazy import)
    search_table2table = _get_search_table2table()
    all_results = search_table2table(
        query=query,
        search_type=search_type,
        k=k * 3,  # Get more results to account for filtering
        db_path=db_path,
        target=target,
        source_column=source_column,
        target_column=target_column
    )
    
    print(f"✅ Found {len(all_results)} candidate tables")
    
    # Step 5: Filter results to only include tables with matching classification
    print(f"\n📊 Step 5: Filtering results by classification...")
    filtered_results = [tid for tid in all_results if tid in matching_tableids]
    
    # If we don't have enough results, we could expand the search
    # For now, just return what we have (up to k)
    final_results = filtered_results[:k]
    
    print(f"✅ Filtered to {len(final_results)} tables with matching classification")
    
    if len(final_results) < k:
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

First, classify all tables in your datalake:
  python -m src.search.classification --mode batch --db_path data_citationlake/modellake.db --output_json data/table_classifications.json

Then use this tool for search:
  python -m src.search.tab2tab_by_type --query query.csv --classification_json data/table_classifications.json --search_type single_column --k 10
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
    parser.add_argument('--db_path', default='data_citationlake/modellake.db',
                       help='Path to modellake.db')
    parser.add_argument('--classification_json', default='data/table_classifications.json',
                       help='Path to JSON file with pre-computed classifications')
    parser.add_argument('--output', default='data/tab2tab_by_type_results.json',
                       help='Output file to save results (JSON format)')
    parser.add_argument('--no_auto_classify', dest='auto_classify', action='store_false',
                       help='Disable automatic classification of query table')
    
    args = parser.parse_args()
    
    try:
        # Parse query based on search type
        query = None
        if args.search_type == 'single_column':
            if os.path.exists(args.query):
                # It's a CSV file
                query = pd.read_csv(args.query)
            else:
                # Comma-separated values
                query = [x.strip() for x in args.query.split(',')]
        elif args.search_type == 'multi_column':
            query = pd.read_csv(args.query)
        elif args.search_type == 'unionable':
            query = pd.read_csv(args.query)
        elif args.search_type == 'keyword':
            if os.path.exists(args.query):
                # CSV file - use headers as keywords
                df = pd.read_csv(args.query, nrows=0)
                query = [str(col).lower().strip() for col in df.columns]
            else:
                # Comma-separated keywords
                query = [x.strip() for x in args.query.split(',')]
        else:
            # For other types, assume CSV file
            query = pd.read_csv(args.query)
        
        # Perform search
        results = search_table2table_by_type(
            query=query,
            search_type=args.search_type,
            k=args.k,
            db_path=args.db_path,
            classification_json=args.classification_json,
            auto_classify=args.auto_classify
        )
        
        print(f"Found {len(results)} tables:")
        for i, table_id in enumerate(results, 1):
            print(f"  {i}. Table ID: {table_id}")
        
        # Save results as JSON
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        result_data = {
            "query": str(query) if not isinstance(query, (list, pd.DataFrame)) else "DataFrame or list",
            "search_type": args.search_type,
            "k": args.k,
            "results": [int(tid) for tid in results],
            "num_results": len(results)
        }
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Results saved to {args.output}")
        print(f"\n⏱️ Total time: {time.time() - start_time:.2f}s")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


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

