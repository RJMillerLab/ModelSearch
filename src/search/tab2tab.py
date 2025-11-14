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
blend_path = os.path.join(os.path.dirname(__file__), '../Blend_internal')
if blend_path not in sys.path:
    sys.path.insert(0, blend_path)

# Import Blend_internal search functions
from src.Tasks.SingleColumnJoinSearch import SingleColumnJoinSearch
from src.Tasks.MultiColumnJoinSearch import MultiColumnJoinSearch
from src.Tasks.KeywordSearch import KeywordSearch


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
    k: int = 10
) -> List[int]:
    """
    Search for tables with overlapping values in a single column.
    
    Args:
        query_values: Iterable of values to search for
        k: Number of results to return
    
    Returns:
        List of table IDs (integers)
    """
    plan = SingleColumnJoinSearch(query_values, k)
    return plan.run()


def search_multi_column(
    query_dataset: pd.DataFrame,
    k: int = 10
) -> List[int]:
    """
    Search for tables with overlapping values across multiple columns.
    
    Args:
        query_dataset: DataFrame with query data
        k: Number of results to return
    
    Returns:
        List of table IDs (integers)
    """
    plan = MultiColumnJoinSearch(query_dataset, k)
    return plan.run()


def search_keyword(
    query_values: List[str],
    k: int = 10
) -> List[int]:
    """
    Search for tables using keyword matching.
    
    Args:
        query_values: List of keyword strings to search for
        k: Number of results to return
    
    Returns:
        List of table IDs (integers)
    """
    plan = KeywordSearch(query_values, k)
    return plan.run()


def search_table2table(
    query: Any,
    search_type: str = "single_column",
    k: int = 10
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
        return search_single_column(query, k)
    elif search_type == "multi_column":
        if not isinstance(query, pd.DataFrame):
            raise ValueError("For multi_column search, query must be a pandas DataFrame")
        return search_multi_column(query, k)
    elif search_type == "keyword":
        if not isinstance(query, list) or not all(isinstance(x, str) for x in query):
            raise ValueError("For keyword search, query must be a list of strings")
        return search_keyword(query, k)
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
        results = search_table2table(query, args.search_type, args.k)
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

