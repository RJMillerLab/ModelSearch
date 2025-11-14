"""
Card to Tab to Card Search

This module provides a two-stage search:
1. Given a model card, find its associated tables
2. Search for similar tables using tab2tab
3. Map similar tables back to model cards using relationship parquet

Uses CitationLake's get_from.py approach for robust parquet schema handling.
"""

import os
import sys
from typing import List, Set, Dict, Optional, Any
import argparse
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Add CitationLake to path for get_from functionality
citationlake_path = os.path.join(os.path.dirname(__file__), '../../CitationLake')
if os.path.exists(citationlake_path) and citationlake_path not in sys.path:
    sys.path.insert(0, citationlake_path)

# Lazy import tab2tab to avoid DBHandler initialization on import
# tab2tab will only be imported when search_card2tab2card is actually called
_tab2tab_search_table2table = None

def _get_search_table2table():
    """Lazy import search_table2table from tab2tab."""
    global _tab2tab_search_table2table
    if _tab2tab_search_table2table is None:
        from src.search.tab2tab import search_table2table
        _tab2tab_search_table2table = search_table2table
    return _tab2tab_search_table2table

# Try to import from CitationLake, fallback to local implementation
try:
    from src.data_analysis.get_from import generic_get_attr_from_attr
    USE_CITATIONLAKE_GET_FROM = True
except ImportError:
    # Fallback to local implementation
    from src.modelsearch.compare_baselines import _read_relationships
    USE_CITATIONLAKE_GET_FROM = False


def get_tables_for_model_from_citationlake(
    model_id: str,
    schema_log_path: str = "data_citationlake/logs/parquet_schema.log",
    table_attr: str = "hugging_table_list_dedup",
    debug: bool = False
) -> List[str]:
    """
    Get list of table CSV paths/basenames for a given model ID using CitationLake's get_from.
    
    This function uses the generic_get_attr_from_attr approach which automatically
    discovers the right parquet file from the schema log.
    
    Args:
        model_id: Hugging Face model ID
        schema_log_path: Path to parquet_schema.log (default: data_citationlake/logs/parquet_schema.log)
        table_attr: Which table attribute to get. Options:
            - "hugging_table_list_dedup"
            - "github_table_list_dedup"
            - "html_table_list_mapped_dedup"
            - "llm_table_list_mapped_dedup"
            - Or any combination (will try all if None)
        debug: Whether to print debug information
    
    Returns:
        List of CSV paths/basenames
    """
    if not USE_CITATIONLAKE_GET_FROM:
        raise ImportError("CitationLake get_from module not available. Please ensure CitationLake is accessible.")
    
    if table_attr:
        # Get specific table attribute
        results = generic_get_attr_from_attr(
            target_attr=table_attr,
            source_attr="modelId",
            value=model_id,
            log_path=schema_log_path,
            debug=debug
        )
        return [str(r) for r in results if r]
    else:
        # Get all table attributes
        all_tables = []
        for attr in [
            "hugging_table_list_dedup",
            "github_table_list_dedup",
            "html_table_list_mapped_dedup",
            "llm_table_list_mapped_dedup"
        ]:
            results = generic_get_attr_from_attr(
                target_attr=attr,
                source_attr="modelId",
                value=model_id,
                log_path=schema_log_path,
                debug=debug
            )
            all_tables.extend([str(r) for r in results if r])
        return list(set(all_tables))  # Deduplicate


def get_modelids_from_table(
    table_path: str,
    schema_log_path: str = "data_citationlake/logs/parquet_schema.log",
    debug: bool = False
) -> List[str]:
    """
    Get model IDs that have a specific table, using CitationLake's get_from.
    
    Args:
        table_path: CSV path or basename to search for
        schema_log_path: Path to parquet_schema.log
        debug: Whether to print debug information
    
    Returns:
        List of model IDs
    """
    if not USE_CITATIONLAKE_GET_FROM:
        raise ImportError("CitationLake get_from module not available.")
    
    # Try different table attributes as source
    all_model_ids = []
    for attr in [
        "hugging_table_list_dedup",
        "github_table_list_dedup",
        "html_table_list_mapped_dedup",
        "llm_table_list_mapped_dedup"
    ]:
        results = generic_get_attr_from_attr(
            target_attr="modelId",
            source_attr=attr,
            value=table_path,
            log_path=schema_log_path,
            debug=debug
        )
        all_model_ids.extend([str(r) for r in results if r])
    
    return list(set(all_model_ids))  # Deduplicate


def load_relationship_parquet(parquet_path: str) -> pd.DataFrame:
    """
    Load relationship parquet file that maps modelId to CSV basenames.
    Fallback method when CitationLake get_from is not available.
    
    Args:
        parquet_path: Path to relationship parquet file
    
    Returns:
        DataFrame with columns: modelId, csv_basename
    """
    if USE_CITATIONLAKE_GET_FROM:
        # If we have get_from, we don't need to load the full parquet
        # But we keep this for backward compatibility
        pass
    from src.modelsearch.compare_baselines import _read_relationships
    return _read_relationships(parquet_path)


def get_tables_for_model(
    model_id: str,
    relationship_df: Optional[pd.DataFrame] = None,
    schema_log_path: str = "data_citationlake/logs/parquet_schema.log",
    use_citationlake: bool = True
) -> List[str]:
    """
    Get list of table CSV basenames for a given model ID.
    
    Uses CitationLake's get_from if available, otherwise falls back to DataFrame lookup.
    
    Args:
        model_id: Hugging Face model ID
        relationship_df: Optional DataFrame from load_relationship_parquet (fallback)
        schema_log_path: Path to parquet_schema.log (for CitationLake approach)
        use_citationlake: Whether to use CitationLake get_from (default: True)
    
    Returns:
        List of CSV basenames
    """
    if use_citationlake and USE_CITATIONLAKE_GET_FROM:
        return get_tables_for_model_from_citationlake(
            model_id=model_id,
            schema_log_path=schema_log_path,
            table_attr=None,  # Get all table types
            debug=False
        )
    else:
        # Fallback to DataFrame approach
        if relationship_df is None:
            raise ValueError("relationship_df is required when use_citationlake=False")
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
    schema_log_path: str = "data_citationlake/logs/parquet_schema.log",
    use_citationlake: bool = True,
    output_json: str = "data/card2tab2card_results.json"
) -> List[str]:
    """
    Search for model cards via table search.
    
    Process:
    1. Get tables for the query model using CitationLake's get_from (or relationship parquet)
    2. Use tab2tab to search for similar tables
    3. Map similar tables back to model cards using CitationLake's get_from (or relationship)
    
    Args:
        model_id: Hugging Face model ID to search from
        relationship_parquet: Optional path to relationship parquet file (fallback if CitationLake not available)
        query: Optional query data for table search. If None, uses tables from the model
        search_type: Type of table search - "single_column", "multi_column", or "keyword"
        k: Number of table results to retrieve
        schema_log_path: Path to parquet_schema.log (for CitationLake approach)
        use_citationlake: Whether to use CitationLake get_from (default: True)
        output_json: Optional path to save results as JSON
    
    Returns:
        List of model IDs that have similar tables
    """
    # Get tables for the query model
    if use_citationlake and USE_CITATIONLAKE_GET_FROM:
        query_tables = get_tables_for_model(
            model_id=model_id,
            schema_log_path=schema_log_path,
            use_citationlake=True
        )
    else:
        if relationship_parquet is None:
            raise ValueError("relationship_parquet is required when use_citationlake=False")
        relationship_df = load_relationship_parquet(relationship_parquet)
        query_tables = get_tables_for_model(
            model_id=model_id,
            relationship_df=relationship_df,
            use_citationlake=False
        )
    
    if not query_tables:
        print(f"Warning: No tables found for model {model_id}")
        return []
    
    print(f"Found {len(query_tables)} tables for model {model_id}")
    
    # If query is provided, use it; otherwise search based on query_tables
    if query is None:
        # Use the query model's tables as the search query
        # For keyword search, use table basenames
        if search_type == "keyword":
            query = [os.path.basename(str(t)) for t in query_tables[:10]]  # Limit to first 10
        else:
            # For other search types, we'd need to load the actual table data
            # For now, default to keyword search
            query = [os.path.basename(str(t)) for t in query_tables[:10]]
            search_type = "keyword"
    
    # Search for similar tables using tab2tab (lazy import)
    try:
        search_table2table = _get_search_table2table()
        similar_table_ids = search_table2table(query, search_type, k)
    except Exception as e:
        print(f"Error in table search: {e}")
        return []
    
    # Map similar tables back to model cards
    similar_model_ids = set()
    
    # For each retrieved table ID, find which model IDs have it
    # Note: This requires mapping Blend_internal table IDs to CSV paths/basenames
    # For now, we'll assume table IDs can be used directly or need conversion
    # This is a simplified approach - you may need to adjust based on Blend_internal's table ID format
    
    # If using CitationLake get_from, we can map table paths to model IDs
    if use_citationlake and USE_CITATIONLAKE_GET_FROM:
        for table_id in similar_table_ids:
            # Convert table ID to string and try to find model IDs
            # Note: You may need to map Blend_internal table IDs to actual CSV paths
            table_path = str(table_id)
            model_ids = get_modelids_from_table(
                table_path=table_path,
                schema_log_path=schema_log_path,
                debug=False
            )
            similar_model_ids.update(model_ids)
    else:
        # Fallback: use relationship_df
        if relationship_parquet is None:
            raise ValueError("relationship_parquet is required when use_citationlake=False")
        relationship_df = load_relationship_parquet(relationship_parquet)
        
        # Convert table IDs to basenames (simplified - may need adjustment)
        table_basenames = [os.path.basename(str(tid)) for tid in similar_table_ids]
        
        similar_model_ids = set(relationship_df.loc[
            relationship_df["csv_basename"].isin(table_basenames),
            "modelId"
        ].dropna().unique().tolist())
    
    # Remove the query model itself
    similar_model_ids = [mid for mid in similar_model_ids if mid != model_id]
    
    # Save if requested
    if output_json:
        result = {
            "query_model": model_id,
            "query_tables": query_tables,
            "similar_models": list(similar_model_ids)
        }
        os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            import json
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved to {output_json}")
    
    return list(similar_model_ids)


def search_card2tab2card_from_tables(
    model_id: str,
    table_search_results: Dict[str, List[str]],
    relationship_parquet: Optional[str] = None,
    schema_log_path: str = "data_citationlake/logs/parquet_schema.log",
    use_citationlake: bool = True,
    k: int = 10
) -> List[str]:
    """
    Search for model cards using pre-computed table search results.
    
    This is useful when you have already run table search and have the results.
    
    Args:
        model_id: Hugging Face model ID to search from
        table_search_results: Dictionary mapping CSV basename/path to list of similar CSV basenames/paths
        relationship_parquet: Optional path to relationship parquet file (fallback)
        schema_log_path: Path to parquet_schema.log (for CitationLake approach)
        use_citationlake: Whether to use CitationLake get_from (default: True)
        k: Maximum number of results to return
    
    Returns:
        List of model IDs that have similar tables
    """
    # Get tables for the query model
    if use_citationlake and USE_CITATIONLAKE_GET_FROM:
        query_tables = get_tables_for_model(
            model_id=model_id,
            schema_log_path=schema_log_path,
            use_citationlake=True
        )
    else:
        if relationship_parquet is None:
            raise ValueError("relationship_parquet is required when use_citationlake=False")
        relationship_df = load_relationship_parquet(relationship_parquet)
        query_tables = get_tables_for_model(
            model_id=model_id,
            relationship_df=relationship_df,
            use_citationlake=False
        )
    
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
    
    # Map retrieved tables to model IDs
    similar_model_ids = set()
    
    if use_citationlake and USE_CITATIONLAKE_GET_FROM:
        # Use CitationLake get_from to map tables to model IDs
        for table_path in retrieved_tables:
            model_ids = get_modelids_from_table(
                table_path=table_path,
                schema_log_path=schema_log_path,
                debug=False
            )
            similar_model_ids.update(model_ids)
    else:
        # Fallback: use relationship_df
        if relationship_parquet is None:
            raise ValueError("relationship_parquet is required when use_citationlake=False")
        relationship_df = load_relationship_parquet(relationship_parquet)
        
        # Try both full paths and basenames
        table_basenames = [os.path.basename(str(t)) for t in retrieved_tables]
        all_keys = list(retrieved_tables) + table_basenames
        
        similar_model_ids = set(relationship_df.loc[
            relationship_df["csv_basename"].isin(all_keys),
            "modelId"
        ].dropna().unique().tolist())
    
    # Remove the query model itself
    similar_model_ids = [mid for mid in similar_model_ids if mid != model_id]
    
    # Limit to top k
    return list(similar_model_ids)[:k]


def main():
    """CLI entry point for card2tab2card search"""
    parser = argparse.ArgumentParser(description="Card to Tab to Card Search")
    parser.add_argument('--model_id', required=True,
                       help='Hugging Face model ID to search from')
    parser.add_argument('--relationship_parquet', default=None,
                       help='Path to relationship parquet file (fallback if CitationLake not available)')
    parser.add_argument('--schema_log', default='data_citationlake/logs/parquet_schema.log',
                       help='Path to parquet_schema.log (for CitationLake approach)')
    parser.add_argument('--query', default=None,
                       help='Query data for table search (comma-separated for single_column/keyword, CSV path for multi_column). If None, uses model tables.')
    parser.add_argument('--search_type', choices=['single_column', 'multi_column', 'keyword'],
                       default='keyword',
                       help='Type of table search')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of table results to retrieve')
    parser.add_argument('--table_search_json', default=None,
                       help='Optional: Path to pre-computed table search results JSON')
    parser.add_argument('--use_citationlake', action='store_true', default=True,
                       help='Use CitationLake get_from approach (default: True)')
    parser.add_argument('--no_citationlake', dest='use_citationlake', action='store_false',
                       help='Disable CitationLake approach, use relationship_parquet instead')
    parser.add_argument('--output_json', default='data/card2tab2card_results.json',
                       help='Path to save results as JSON (default: data/card2tab2card_results.json)')
    
    args = parser.parse_args()
    
    try:
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
                elif args.search_type == 'keyword':
                    query = [x.strip() for x in args.query.split(',')]
            
            results = search_card2tab2card(
                model_id=args.model_id,
                relationship_parquet=args.relationship_parquet,
                query=query,
                search_type=args.search_type,
                k=args.k,
                schema_log_path=args.schema_log,
                use_citationlake=args.use_citationlake,
                output_json=args.output_json or "data/card2tab2card_results.json"
            )
        
        print(f"Found {len(results)} similar model cards for {args.model_id}:")
        for i, model_id in enumerate(results, 1):
            print(f"  {i}. {model_id}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

