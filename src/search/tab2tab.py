"""
Table to Table Search (Testing Tool)

This module provides functions for table-to-table search using Blend_internal.
Wraps and reuses functionality from Blend_internal/src/Tasks.
Supports reading tables from modellake.db for testing.
"""

import os
import sys
from typing import List, Dict, Optional, Iterable, Any
import pandas as pd
import argparse
import duckdb

# Add Blend_internal to path (now in src/)
# Try src/Blend_internal first (if cloned into ModelSearchDemo)
blend_path = os.path.join(os.path.dirname(__file__), '..', 'Blend_internal')
blend_path_abs = os.path.abspath(blend_path)
if not os.path.exists(blend_path_abs):
    # Fallback: try parent directory (if Blend_internal is sibling to ModelSearchDemo)
    blend_path_parent = os.path.join(os.path.dirname(__file__), '..', '..', 'Blend_internal')
    blend_path_parent_abs = os.path.abspath(blend_path_parent)
    if os.path.exists(blend_path_parent_abs):
        blend_path_abs = blend_path_parent_abs

# CRITICAL: Insert Blend_internal path at the BEGINNING of sys.path
# This ensures Blend_internal's src/utils.py is found before ModelSearchDemo's src/utils/
if blend_path_abs and os.path.exists(blend_path_abs):
    # Remove if already in path to avoid duplicates
    if blend_path_abs in sys.path:
        sys.path.remove(blend_path_abs)
    # Insert at position 0 to give highest priority
    sys.path.insert(0, blend_path_abs)

# Store Blend_internal path for later use
_BLEND_INTERNAL_PATH = blend_path_abs if os.path.exists(blend_path_abs) else blend_path_parent_abs if os.path.exists(blend_path_parent_abs) else None

# Lazy import - will be imported when needed, after config is set
_SingleColumnJoinSearch = None
_MultiColumnJoinSearch = None
_KeywordSearch = None

def _update_blend_config(db_path: str):
    """Update Blend_internal config.ini with the correct db_path before importing."""
    if _BLEND_INTERNAL_PATH is None:
        raise FileNotFoundError("Blend_internal not found. Please clone it first: git clone git@github.com:DoraDong-2023/Blend_internal.git src/Blend_internal")
    
    import configparser
    config_path = os.path.join(_BLEND_INTERNAL_PATH, 'config', 'config.ini')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Blend_internal config.ini not found at {config_path}")
    
    # Read and update config
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Convert relative path to absolute if needed
    if not os.path.isabs(db_path):
        # Make path relative to ModelSearchDemo root
        modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        db_path_abs = os.path.abspath(os.path.join(modelsearch_root, db_path))
    else:
        db_path_abs = os.path.abspath(db_path)
    
    # Update config
    if 'Database' not in config:
        config['Database'] = {}
    config['Database']['path'] = db_path_abs
    config['Database']['dbms'] = 'duckdb'
    config['Database']['index_table'] = 'modellake_index'
    
    # Write back
    with open(config_path, 'w') as f:
        config.write(f)
    
    return config_path

def _lazy_import_blend():
    """Lazy import Blend_internal functions after config is set."""
    global _SingleColumnJoinSearch, _MultiColumnJoinSearch, _KeywordSearch
    if _SingleColumnJoinSearch is None:
        # CRITICAL: Ensure Blend_internal path is at the front of sys.path
        # This must be done before any imports to ensure Blend_internal's src/utils.py
        # is found instead of ModelSearchDemo's src/utils/ package
        if _BLEND_INTERNAL_PATH and os.path.exists(_BLEND_INTERNAL_PATH):
            if _BLEND_INTERNAL_PATH in sys.path:
                sys.path.remove(_BLEND_INTERNAL_PATH)
            sys.path.insert(0, _BLEND_INTERNAL_PATH)
        
        # CRITICAL: Temporarily remove ModelSearchDemo root from sys.path if present
        # This prevents Python from finding ModelSearchDemo's src/utils/ package
        # instead of Blend_internal's src/utils.py module
        modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        modelsearch_removed = False
        if modelsearch_root in sys.path:
            sys.path.remove(modelsearch_root)
            modelsearch_removed = True
        
        try:
            # Clear any cached imports to ensure fresh import with updated config
            # IMPORTANT: Also clear src.utils to ensure Blend_internal's src/utils.py is used
            # instead of ModelSearchDemo's src/utils/ package
            modules_to_clear = [
                'src.Tasks.SingleColumnJoinSearch',
                'src.Tasks.MultiColumnJoinSearch', 
                'src.Tasks.KeywordSearch',
                'src.Plan',
                'src.Operators',
                'src.Operators.OperatorBase',
                'src.Operators.Seekers',
                'src.Operators.Seekers.MultiColumnOverlap',
                'src.DBHandler',
                'src.utils',  # CRITICAL: Clear this to use Blend_internal's src/utils.py
            ]
            for mod in modules_to_clear:
                if mod in sys.modules:
                    del sys.modules[mod]
            
            # Now import with updated config
            # The Blend_internal path is now at sys.path[0], so src.utils will resolve to
            # Blend_internal/src/utils.py instead of ModelSearchDemo/src/utils/
            from src.Tasks.SingleColumnJoinSearch import SingleColumnJoinSearch
            from src.Tasks.MultiColumnJoinSearch import MultiColumnJoinSearch
            from src.Tasks.KeywordSearch import KeywordSearch
            _SingleColumnJoinSearch = SingleColumnJoinSearch
            _MultiColumnJoinSearch = MultiColumnJoinSearch
            _KeywordSearch = KeywordSearch
        finally:
            # Restore ModelSearchDemo root to sys.path if we removed it
            if modelsearch_removed and modelsearch_root not in sys.path:
                sys.path.append(modelsearch_root)


def get_tables_from_modellake_db(
    db_path: str = "data/modellake.db",
    index_table: str = "modellake_index",
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get all tables from modellake.db for testing.
    
    Args:
        db_path: Path to modellake.db
        index_table: Name of the index table
        limit: Optional limit on number of tables to return
    
    Returns:
        List of table metadata dictionaries with keys: tableid, filename, table_group, table_type
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"modellake.db not found at {db_path}")
    
    con = duckdb.connect(db_path, read_only=True)
    try:
        query = f"""
        SELECT DISTINCT tableid, filename, table_group, table_type 
        FROM {index_table} 
        WHERE rowid = -1
        """
        if limit:
            query += f" LIMIT {limit}"
        
        results = con.execute(query).fetchall()
        tables = [
            {
                "tableid": row[0],
                "filename": row[1],
                "table_group": row[2],
                "table_type": row[3]
            }
            for row in results
        ]
        return tables
    finally:
        con.close()


def search_single_column(
    query_values: Iterable[Any],
    k: int = 10,
    db_path: Optional[str] = None
) -> List[int]:
    """
    Search for tables with overlapping values in a single column.
    
    Args:
        query_values: Iterable of values to search for
        k: Number of results to return
        db_path: Path to modellake.db (optional, will use config default if not provided)
    
    Returns:
        List of table IDs (integers)
    """
    # Always update config before importing (even if db_path is None, use default)
    if db_path:
        _update_blend_config(db_path)
    else:
        # Use default path
        default_path = "data/modellake.db"
        modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        default_path_abs = os.path.abspath(os.path.join(modelsearch_root, default_path))
        _update_blend_config(default_path_abs)
    _lazy_import_blend()
    plan = _SingleColumnJoinSearch(query_values, k)
    return plan.run()


def search_multi_column(
    query_dataset: pd.DataFrame,
    k: int = 10,
    db_path: Optional[str] = None
) -> List[int]:
    """
    Search for tables with overlapping values across multiple columns.
    
    Args:
        query_dataset: DataFrame with query data
        k: Number of results to return
        db_path: Path to modellake.db (optional, will use config default if not provided)
    
    Returns:
        List of table IDs (integers)
    """
    # Always update config before importing (even if db_path is None, use default)
    if db_path:
        _update_blend_config(db_path)
    else:
        # Use default path
        default_path = "data/modellake.db"
        modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        default_path_abs = os.path.abspath(os.path.join(modelsearch_root, default_path))
        _update_blend_config(default_path_abs)
    _lazy_import_blend()
    plan = _MultiColumnJoinSearch(query_dataset, k)
    return plan.run()


def search_keyword(
    query_values: List[str],
    k: int = 10,
    db_path: Optional[str] = None
) -> List[int]:
    """
    Search for tables using keyword matching.
    
    Args:
        query_values: List of keyword strings to search for
        k: Number of results to return
        db_path: Path to modellake.db (optional, will use config default if not provided)
    
    Returns:
        List of table IDs (integers)
    """
    # Always update config before importing (even if db_path is None, use default)
    if db_path:
        _update_blend_config(db_path)
    else:
        # Use default path
        default_path = "data/modellake.db"
        modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        default_path_abs = os.path.abspath(os.path.join(modelsearch_root, default_path))
        _update_blend_config(default_path_abs)
    _lazy_import_blend()
    plan = _KeywordSearch(query_values, k)
    return plan.run()


def search_table2table(
    query: Any,
    search_type: str = "single_column",
    k: int = 10,
    db_path: Optional[str] = None
) -> List[int]:
    """
    Unified interface for table-to-table search.
    
    Args:
        query: Query data - can be:
            - Iterable of values (for single_column)
            - pd.DataFrame (for multi_column)
            - List[str] (for keyword)
        search_type: Type of search - "single_column", "multi_column", or "keyword"
        k: Number of results to return
    
    Returns:
        List of table IDs (integers)
    """
    if search_type == "single_column":
        if not isinstance(query, (list, tuple, pd.Series)):
            raise ValueError("For single_column search, query must be an iterable of values")
        return search_single_column(query, k, db_path=db_path)
    elif search_type == "multi_column":
        if not isinstance(query, pd.DataFrame):
            raise ValueError("For multi_column search, query must be a pandas DataFrame")
        return search_multi_column(query, k, db_path=db_path)
    elif search_type == "keyword":
        if not isinstance(query, list) or not all(isinstance(x, str) for x in query):
            raise ValueError("For keyword search, query must be a list of strings")
        return search_keyword(query, k, db_path=db_path)
    else:
        raise ValueError(f"Unknown search_type: {search_type}. Must be 'single_column', 'multi_column', or 'keyword'")


def main():
    """CLI entry point for tab2tab search (testing tool)"""
    parser = argparse.ArgumentParser(
        description="Table to Table Search using Blend_internal (Testing Tool)",
        epilog="""
Note: modellake.db is a DuckDB database containing an indexed table (modellake_index) 
created from all CSV files. The index table has columns:
- tokenized: tokenized cell values
- tableid: table ID
- colid: column ID  
- rowid: row ID (-1 for headers)
- filename: CSV filename
- table_group: table group
- table_type: table type (ori, str, tr, etc.)

To create modellake.db, use:
  python -m src.Blend_internal.scripts.create_index_duckdb \\
    --db_path data/modellake.db \\
    --data_glob "path/to/csvs/*.csv" \\
    --table modellake_index
        """
    )
    parser.add_argument('--search_type', choices=['single_column', 'multi_column', 'keyword'],
                       default='single_column', help='Type of search to perform')
    parser.add_argument('--query', default=None,
                       help='Query data. For single_column: comma-separated values. '
                            'For multi_column: path to CSV file. For keyword: comma-separated keywords.')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of results to return')
    parser.add_argument('--db_path', default='data/modellake.db',
                       help='Path to modellake.db for testing')
    parser.add_argument('--list_tables', action='store_true',
                       help='List all tables from modellake.db and exit')
    parser.add_argument('--test_table_id', type=int, default=None,
                       help='Test with a specific table ID from modellake.db')
    parser.add_argument('--output', default='data/tab2tab_results.json',
                       help='Output file to save results (JSON format)')
    
    args = parser.parse_args()
    
    # Update Blend_internal config with db_path BEFORE any imports
    # This must happen before _lazy_import_blend() is called
    try:
        _update_blend_config(args.db_path)
        print(f"✅ Updated Blend_internal config to use db_path: {args.db_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not update config: {e}")
        print("   Continuing with default config...")
    
    # List tables if requested
    if args.list_tables:
        try:
            tables = get_tables_from_modellake_db(db_path=args.db_path)
            print(f"Found {len(tables)} tables in modellake.db:")
            for i, table in enumerate(tables[:50], 1):  # Show first 50
                print(f"  {i}. Table ID: {table['tableid']}, File: {table['filename']}, "
                      f"Group: {table['table_group']}, Type: {table['table_type']}")
            if len(tables) > 50:
                print(f"  ... and {len(tables) - 50} more tables")
        except Exception as e:
            print(f"❌ Error listing tables: {e}")
            print(f"\n💡 Tip: modellake.db needs to be created first. See --help for instructions.")
            import traceback
            traceback.print_exc()
        return
    
    # If no query provided, try to use test_table_id or require query
    if args.query is None and args.test_table_id is None:
        parser.error("Either --query or --test_table_id must be provided")
    
    # Parse query based on search type
    if args.query:
        if args.search_type == 'single_column':
            query = [x.strip() for x in args.query.split(',')]
        elif args.search_type == 'multi_column':
            query = pd.read_csv(args.query)
        elif args.search_type == 'keyword':
            query = [x.strip() for x in args.query.split(',')]
    else:
        # Use test_table_id - for now, just use the table ID as a keyword
        # This is a simplified test approach
        query = [str(args.test_table_id)]
        args.search_type = 'keyword'
        print(f"Testing with table ID {args.test_table_id} as keyword")
    
    # Perform search
    try:
        results = search_table2table(query, args.search_type, args.k, db_path=args.db_path)
        print(f"Found {len(results)} tables:")
        for i, table_id in enumerate(results, 1):
            print(f"  {i}. Table ID: {table_id}")
        
        # Save results as JSON
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        result_data = {
            "query": query if isinstance(query, list) else str(query),
            "search_type": args.search_type,
            "k": args.k,
            "results": [int(tid) for tid in results],
            "num_results": len(results)
        }
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved to {args.output}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

