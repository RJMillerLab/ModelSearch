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
import time
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
    table_search_k: Optional[int] = None,
    modelcard_k: Optional[int] = None,
    schema_log_path: str = "data_citationlake/logs/parquet_schema.log",
    use_citationlake: bool = True,
    output_json: str = "data/card2tab2card_results.json",
    db_path: Optional[str] = None
) -> List[str]:
    """
    Search for model cards via table search.
    
    Pipeline: Query -> ModelCard -> Tables -> Retrieved Tables -> Corresponding ModelCards
    
    Process:
    1. Get tables for the query model using CitationLake's get_from (or relationship parquet)
    2. Use tab2tab to search for similar tables
    3. Map retrieved tables back to model cards using CitationLake's get_from (or relationship)
    
    Args:
        model_id: Hugging Face model ID to search from
        relationship_parquet: Optional path to relationship parquet file (fallback if CitationLake not available)
        query: Optional query data for table search. If None, uses tables from the model
        search_type: Type of table search - "single_column", "multi_column", "keyword", or "unionable"
        k: Legacy parameter for backward compatibility. If table_search_k or modelcard_k are None, uses k for both.
        table_search_k: Number of table results to retrieve (defaults to k if not provided)
        modelcard_k: Number of final model card results to return (defaults to k if not provided)
        schema_log_path: Path to parquet_schema.log (for CitationLake approach)
        use_citationlake: Whether to use CitationLake get_from (default: True)
        output_json: Optional path to save results as JSON
        db_path: Path to modellake.db (default: data_citationlake/modellake.db)
    
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
    # Get tables for the query model
    if use_citationlake and USE_CITATIONLAKE_GET_FROM:
        query_tables = get_tables_for_model(
            model_id=model_id,
            schema_log_path=schema_log_path,
            use_citationlake=True
        )
    elif use_citationlake and not USE_CITATIONLAKE_GET_FROM:
        # User wants CitationLake but it's not available, try to find default relationship_parquet
        if relationship_parquet is None:
            # Try to find default relationship parquet files
            default_paths = [
                "data_citationlake/processed/modelcard_step3_dedup.parquet",
                "data_citationlake/processed/modelcard_step3.parquet",
            ]
            for default_path in default_paths:
                if os.path.exists(default_path):
                    relationship_parquet = default_path
                    print(f"⚠️  CitationLake not available, found default relationship_parquet: {relationship_parquet}")
                    break
        
        if relationship_parquet is None:
            raise ValueError(
                "CitationLake get_from is not available and no relationship_parquet found. Please either:\n"
                "  1. Install/configure CitationLake (ensure CitationLake/src/data_analysis/get_from.py exists), or\n"
                "  2. Use --relationship_parquet to specify the path to modelcard_step3_dedup.parquet, or\n"
                "  3. Place modelcard_step3_dedup.parquet in data_citationlake/processed/"
            )
        
        if not os.path.exists(relationship_parquet):
            raise FileNotFoundError(
                f"relationship_parquet not found: {relationship_parquet}\n"
                "Please check the path or use --relationship_parquet to specify the correct path."
            )
        
        print(f"⚠️  CitationLake not available, using relationship_parquet: {relationship_parquet}")
        relationship_df = load_relationship_parquet(relationship_parquet)
        query_tables = get_tables_for_model(
            model_id=model_id,
            relationship_df=relationship_df,
            use_citationlake=False
        )
    else:
        # use_citationlake=False, use relationship_parquet
        if relationship_parquet is None:
            raise ValueError("relationship_parquet is required when use_citationlake=False")
        relationship_df = load_relationship_parquet(relationship_parquet)
        query_tables = get_tables_for_model(
            model_id=model_id,
            relationship_df=relationship_df,
            use_citationlake=False
        )
    
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
    
    # Step 1: Query -> ModelCard -> Tables (Part of Pipeline)
    print(f"\n{'='*60}")
    print(f"📊 Step 1: Query -> ModelCard -> Tables (Part of Pipeline)")
    print(f"{'='*60}")
    print(f"✅ Query Model ID: {model_id}")
    print(f"✅ Found {len(query_tables)} tables for model {model_id}")
    print(f"📝 Sample tables (showing first 2):")
    for i, table in enumerate(query_tables[:2], 1):
        print(f"   {i}. {os.path.basename(str(table))}")
    if len(query_tables) > 2:
        print(f"   ... and {len(query_tables) - 2} more tables")
    
    # If query is provided, use it; otherwise search based on query_tables
    if query is None:
        # Use the query model's tables as the search query
        # For keyword search, we need to load the actual CSV files to get headers
        # (Blend_internal uses rowid=-1 which represents headers in the index)
        print(f"ℹ️  No query provided, using model's tables as query")
        if search_type == "keyword":
            # Load headers from model's tables (consistent with Blend_internal)
            all_headers = []
            for table_path in query_tables[:10]:  # Limit to first 10 tables
                try:
                    # Try to find the CSV file
                    csv_path = None
                    if os.path.exists(table_path):
                        csv_path = table_path
                    else:
                        # Try common locations
                        basename = os.path.basename(str(table_path))
                        for base_dir in [
                            "data_citationlake/processed/deduped_hugging_csvs",
                            "data_citationlake/processed/deduped_github_csvs",
                            "data_citationlake/processed/tables_output"
                        ]:
                            full_path = os.path.join(base_dir, basename)
                            if os.path.exists(full_path):
                                csv_path = full_path
                                break
                    
                    if csv_path:
                        df_temp = pd.read_csv(csv_path, nrows=0)
                        headers = [str(col).lower().strip() for col in df_temp.columns]
                        headers = [h for h in headers if h]  # Filter empty
                        all_headers.extend(headers)
                except Exception:
                    continue
            query = list(set(all_headers))  # Remove duplicates
            if not query:
                # Fallback: use table basenames if no headers found
                query = [os.path.basename(str(t)) for t in query_tables[:10]]
        else:
            # For other search types, we'd need to load the actual table data
            # For now, default to keyword search
            query = None
            search_type = "keyword"
    else:
        print(f"ℹ️  Using provided query (not model's tables)")
    
    # Step 2: Tables -> Retrieved Tables
    print(f"\n{'='*60}")
    print(f"🔍 Step 2: Tables -> Retrieved Tables")
    print(f"{'='*60}")
    print(f"✅ Search type: {search_type}")
    if search_type == "keyword":
        print(f"✅ Query keywords: {query[:5]}{'...' if len(query) > 5 else ''}")
        print(f"   (Total {len(query)} keywords)")
    elif search_type == "single_column":
        print(f"✅ Query values: {query[:5]}{'...' if len(query) > 5 else ''}")
        print(f"   (Total {len(query)} values)")
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
    
    try:
        print(f"🔎 Getting search_table2table function...")
        sys.stdout.flush()
        search_table2table = _get_search_table2table()
        print(f"✅ Got search_table2table function")
        sys.stdout.flush()
        print(f"🔎 Searching for similar tables...")
        print(f"   Query type: {type(query)}, Search type: {search_type}, table_search_k: {table_search_k}, db_path: {db_path}")
        sys.stdout.flush()
        
        # Handle correlation search specially - need to extract source and target columns
        if search_type == "correlation" and isinstance(query, pd.DataFrame):
            # Use first column as source, first numeric column as target
            source_col = query[query.columns[0]].astype(str).tolist()
            numeric_cols = query.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                target_col = query[numeric_cols[0]].tolist()
                print(f"   Correlation: source='{query.columns[0]}', target='{numeric_cols[0]}'")
                similar_table_ids = search_table2table(
                    query, search_type, table_search_k, db_path=db_path,
                    source_column=source_col, target_column=target_col
                )
            else:
                print(f"⚠️  No numeric column found for correlation search, skipping...")
                similar_table_ids = []
        else:
            similar_table_ids = search_table2table(query, search_type, table_search_k, db_path=db_path)
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
        con = duckdb.connect(db_path, read_only=True)
        try:
            # Get distinct filenames for the retrieved table IDs
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
        finally:
            con.close()
            
    except Exception as e:
        print(f"❌ Error in table search: {e}")
        import traceback
        traceback.print_exc()
        # Still save intermediate results even on error
        if output_json:
            result = {
                "query_model": model_id,
                "query_tables": query_tables,
                "model_ids": [],
                "intermediate": {
                    "retrieved_table_ids": [],
                    "retrieved_table_filenames": [],
                    "table_id_to_filename": {},
                    "table_to_models": {},
                    "error": str(e)
                }
            }
            os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
            with open(output_json, 'w', encoding='utf-8') as f:
                import json
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ Results saved to {output_json} (with error)")
        return []
    
    # Step 3: Retrieved Tables -> Corresponding ModelCards
    print(f"\n{'='*60}")
    print(f"🔄 Step 3: Retrieved Tables -> Corresponding ModelCards")
    print(f"{'='*60}")
    print(f"✅ Mapping {len(similar_table_ids)} retrieved tables to model cards...")
    
    # Get filenames for retrieved tables from database
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Get distinct filenames for the retrieved table IDs
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
        retrieved_filenames = list(tableid_to_filename.values())
        print(f"✅ Retrieved {len(retrieved_filenames)} unique filenames from database")
        
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
    finally:
        con.close()
    
    # Map similar tables back to model cards
    similar_model_ids = set()
    table_to_models = {}  # Map table filename to list of model IDs
    
    # If using CitationLake get_from, we can map table paths to model IDs
    if use_citationlake and USE_CITATIONLAKE_GET_FROM:
        print(f"📋 Using CitationLake get_from to map tables to model cards...")
        for filename in retrieved_filenames:
            model_ids = get_modelids_from_table(
                table_path=filename,
                schema_log_path=schema_log_path,
                debug=False
            )
            similar_model_ids.update(model_ids)
            table_to_models[filename] = list(model_ids)
    else:
        # Fallback: use relationship_df
        if relationship_parquet is None:
            raise ValueError("relationship_parquet is required when use_citationlake=False")
        relationship_df = load_relationship_parquet(relationship_parquet)
        print(f"📋 Using relationship_parquet to map tables to model cards...")
        
        # Use filenames (basenames) to match with relationship_df
        table_basenames = [os.path.basename(fname) for fname in retrieved_filenames]
        print(f"📝 Matching {len(table_basenames)} table basenames against relationship data...")
        print(f"   Sample basenames: {table_basenames[:3]}{'...' if len(table_basenames) > 3 else ''}")
        
        # Check what columns are available in relationship_df
        print(f"   Relationship DF columns: {list(relationship_df.columns)[:10]}{'...' if len(relationship_df.columns) > 10 else ''}")
        print(f"   Relationship DF shape: {relationship_df.shape}")
        
        # Try multiple column name variations
        basename_col = None
        for col in ["csv_basename", "basename", "filename", "table_basename"]:
            if col in relationship_df.columns:
                basename_col = col
                break
        
        if basename_col is None:
            print(f"⚠️  Could not find basename column in relationship_df. Available columns: {list(relationship_df.columns)}")
            # Try to find a column that might contain basenames
            for col in relationship_df.columns:
                if "basename" in col.lower() or "filename" in col.lower() or "csv" in col.lower():
                    basename_col = col
                    print(f"   Using column: {col}")
                    break
        
        if basename_col is None:
            print(f"❌ No suitable column found for matching basenames")
            similar_model_ids = set()
        else:
            for filename in retrieved_filenames:
                basename = os.path.basename(filename)
                matched_models = relationship_df.loc[
                    relationship_df[basename_col] == basename,
                    "modelId"
                ].dropna().unique().tolist()
                if matched_models:
                    similar_model_ids.update(matched_models)
                    table_to_models[filename] = matched_models
                    print(f"   ✅ Matched {basename} -> {len(matched_models)} models")
                else:
                    print(f"   ⚠️  No match for {basename}")
        
        print(f"✅ Matched {len(similar_model_ids)} model cards from relationship data")
    
    # Remove the query model itself
    similar_model_ids = [mid for mid in similar_model_ids if mid != model_id]
    
    # Also remove query model from table_to_models
    for filename in table_to_models:
        table_to_models[filename] = [mid for mid in table_to_models[filename] if mid != model_id]
    
    print(f"✅ Found {len(similar_model_ids)} unique model cards (excluding query model)")
    
    # Limit to top modelcard_k
    final_results = list(similar_model_ids)[:modelcard_k]
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 Final Results Summary")
    print(f"{'='*60}")
    print(f"✅ Query Model: {model_id}")
    print(f"✅ Found {len(final_results)} similar model cards (top {modelcard_k})")
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
        schema_log_path: Path to parquet_schema.log (for CitationLake approach)
        use_citationlake: Whether to use CitationLake get_from (default: True)
        k: Legacy parameter for backward compatibility. If modelcard_k is None, uses k.
        modelcard_k: Maximum number of model card results to return (defaults to k if not provided)
    
    Returns:
        List of model IDs that have similar tables
    """
    # Handle backward compatibility
    if modelcard_k is None:
        modelcard_k = k
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
    
    # Limit to top modelcard_k
    final_results = list(similar_model_ids)[:modelcard_k]
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 Final Results Summary")
    print(f"{'='*60}")
    print(f"✅ Query Model: {model_id}")
    print(f"✅ Found {len(final_results)} similar model cards (top {modelcard_k})")
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
    classifications: Optional[Dict[int, str]] = None
) -> List[str]:
    """
    Search for model cards via table search with classification filtering.
    
    This is similar to search_card2tab2card but uses tab2tab_by_type which filters
    search results to only include tables with the same classification as the query table.
    
    Pipeline: Query -> ModelCard -> Tables -> Classify -> Retrieved Tables (filtered by type) -> Corresponding ModelCards
    
    Process:
    1. Get tables for the query model using CitationLake's get_from (or relationship parquet)
    2. Classify the query table
    3. Use tab2tab_by_type to search for similar tables (filtered by classification)
    4. Map retrieved tables back to model cards using CitationLake's get_from (or relationship)
    
    Args:
        model_id: Hugging Face model ID to search from
        relationship_parquet: Optional path to relationship parquet file (fallback if CitationLake not available)
        query: Optional query data for table search. If None, uses tables from the model
        search_type: Type of table search - "single_column", "multi_column", "keyword", or "unionable"
        k: Legacy parameter for backward compatibility. If table_search_k or modelcard_k are None, uses k for both.
        table_search_k: Number of table results to retrieve (defaults to k if not provided)
        modelcard_k: Number of final model card results to return (defaults to k if not provided)
        schema_log_path: Path to parquet_schema.log (for CitationLake approach)
        use_citationlake: Whether to use CitationLake get_from (default: True)
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
    
    # Get tables for the query model (same as search_card2tab2card)
    if use_citationlake and USE_CITATIONLAKE_GET_FROM:
        query_tables = get_tables_for_model(
            model_id=model_id,
            schema_log_path=schema_log_path,
            use_citationlake=True
        )
    elif use_citationlake and not USE_CITATIONLAKE_GET_FROM:
        # User wants CitationLake but it's not available, try to find default relationship_parquet
        if relationship_parquet is None:
            default_paths = [
                "data_citationlake/processed/modelcard_step3_dedup.parquet",
                "data_citationlake/processed/modelcard_step3.parquet",
            ]
            for default_path in default_paths:
                if os.path.exists(default_path):
                    relationship_parquet = default_path
                    print(f"⚠️  CitationLake not available, found default relationship_parquet: {relationship_parquet}")
                    break
        
        if relationship_parquet is None:
            raise ValueError(
                "CitationLake get_from is not available and no relationship_parquet found. Please either:\n"
                "  1. Install/configure CitationLake (ensure CitationLake/src/data_analysis/get_from.py exists), or\n"
                "  2. Use --relationship_parquet to specify the path to modelcard_step3_dedup.parquet, or\n"
                "  3. Place modelcard_step3_dedup.parquet in data_citationlake/processed/"
            )
        
        if not os.path.exists(relationship_parquet):
            raise FileNotFoundError(
                f"relationship_parquet not found: {relationship_parquet}\n"
                "Please check the path or use --relationship_parquet to specify the correct path."
            )
        
        print(f"⚠️  CitationLake not available, using relationship_parquet: {relationship_parquet}")
        relationship_df = load_relationship_parquet(relationship_parquet)
        query_tables = get_tables_for_model(
            model_id=model_id,
            relationship_df=relationship_df,
            use_citationlake=False
        )
    else:
        # use_citationlake=False, use relationship_parquet
        if relationship_parquet is None:
            raise ValueError("relationship_parquet is required when use_citationlake=False")
        relationship_df = load_relationship_parquet(relationship_parquet)
        query_tables = get_tables_for_model(
            model_id=model_id,
            relationship_df=relationship_df,
            use_citationlake=False
        )
    
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
    if query is None:
        # Use the query model's tables as the search query
        if search_type == "keyword":
            # Load headers from model's tables
            all_headers = []
            for table_path in query_tables[:10]:  # Limit to first 10 tables
                try:
                    csv_path = None
                    if os.path.exists(table_path):
                        csv_path = table_path
                    else:
                        basename = os.path.basename(str(table_path))
                        for base_dir in [
                            "data_citationlake/processed/deduped_hugging_csvs",
                            "data_citationlake/processed/deduped_github_csvs",
                            "data_citationlake/processed/tables_output"
                        ]:
                            full_path = os.path.join(base_dir, basename)
                            if os.path.exists(full_path):
                                csv_path = full_path
                                break
                    
                    if csv_path:
                        df_temp = pd.read_csv(csv_path, nrows=0)
                        headers = [str(col).lower().strip() for col in df_temp.columns]
                        headers = [h for h in headers if h]
                        all_headers.extend(headers)
                except Exception:
                    continue
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
    
    try:
        print(f"🔎 Getting search_table2table_by_type function...")
        sys.stdout.flush()
        search_table2table_by_type = _get_search_table2table_by_type()
        print(f"✅ Got search_table2table_by_type function")
        sys.stdout.flush()
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
        con = duckdb.connect(db_path, read_only=True)
        try:
            table_ids_str = ','.join(str(tid) for tid in similar_table_ids)
            filename_query = f"""
                SELECT DISTINCT tableid, filename 
                FROM modellake_index 
                WHERE tableid IN ({table_ids_str}) AND rowid = -1
            """
            filename_results = con.execute(filename_query).fetchall()
            tableid_to_filename = {tid: filename for tid, filename in filename_results}
            retrieved_filenames = list(tableid_to_filename.values())
            print(f"✅ Retrieved {len(retrieved_filenames)} unique filenames from database")
        finally:
            con.close()
            
    except Exception as e:
        print(f"❌ Error in table search: {e}")
        import traceback
        traceback.print_exc()
        if output_json:
            result = {
                "query_model": model_id,
                "query_tables": query_tables,
                "model_ids": [],
                "intermediate": {
                    "retrieved_table_ids": similar_table_ids if similar_table_ids else [],
                    "retrieved_table_filenames": retrieved_filenames if retrieved_filenames else [],
                    "table_id_to_filename": tableid_to_filename if tableid_to_filename else {},
                    "table_to_models": {},
                    "error": str(e)
                }
            }
            os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
            with open(output_json, 'w', encoding='utf-8') as f:
                import json
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ Results saved to {output_json} (with error)")
        return []
    
    # Step 3: Retrieved Tables -> Corresponding ModelCards (same as search_card2tab2card)
    print(f"\n{'='*60}")
    print(f"🔄 Step 3: Retrieved Tables -> Corresponding ModelCards")
    print(f"{'='*60}")
    print(f"✅ Mapping {len(similar_table_ids)} retrieved tables to model cards...")
    
    # Map similar tables back to model cards
    similar_model_ids = set()
    table_to_models = {}
    
    if use_citationlake and USE_CITATIONLAKE_GET_FROM:
        print(f"📋 Using CitationLake get_from to map tables to model cards...")
        for filename in retrieved_filenames:
            model_ids = get_modelids_from_table(
                table_path=filename,
                schema_log_path=schema_log_path,
                debug=False
            )
            similar_model_ids.update(model_ids)
            table_to_models[filename] = list(model_ids)
    else:
        # Fallback: use relationship_df
        if relationship_parquet is None:
            raise ValueError("relationship_parquet is required when use_citationlake=False")
        relationship_df = load_relationship_parquet(relationship_parquet)
        print(f"📋 Using relationship_parquet to map tables to model cards...")
        
        table_basenames = [os.path.basename(fname) for fname in retrieved_filenames]
        basename_col = None
        for col in ["csv_basename", "basename", "filename", "table_basename"]:
            if col in relationship_df.columns:
                basename_col = col
                break
        
        if basename_col is None:
            print(f"⚠️  Could not find basename column in relationship_df")
            similar_model_ids = set()
        else:
            for filename in retrieved_filenames:
                basename = os.path.basename(filename)
                matched_models = relationship_df.loc[
                    relationship_df[basename_col] == basename,
                    "modelId"
                ].dropna().unique().tolist()
                if matched_models:
                    similar_model_ids.update(matched_models)
                    table_to_models[filename] = matched_models
        
        print(f"✅ Matched {len(similar_model_ids)} model cards from relationship data")
    
    # Remove the query model itself
    similar_model_ids = [mid for mid in similar_model_ids if mid != model_id]
    
    # Also remove query model from table_to_models
    for filename in table_to_models:
        table_to_models[filename] = [mid for mid in table_to_models[filename] if mid != model_id]
    
    print(f"✅ Found {len(similar_model_ids)} unique model cards (excluding query model)")
    
    # Limit to top modelcard_k
    final_results = list(similar_model_ids)[:modelcard_k]
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 Final Results Summary")
    print(f"{'='*60}")
    print(f"✅ Query Model: {model_id}")
    print(f"✅ Found {len(final_results)} similar model cards (top {modelcard_k}, filtered by classification)")
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
                       help='Path to parquet_schema.log (for CitationLake approach)')
    parser.add_argument('--query', default=None,
                       help='Query data for table search. For mode=single: comma-separated for single_column/keyword, CSV path for multi_column/unionable. For mode=all: CSV file path (required). If None, uses model tables.')
    parser.add_argument('--search_type', choices=['single_column', 'multi_column', 'keyword', 'unionable'],
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
                    schema_log_path=args.schema_log,
                    use_citationlake=args.use_citationlake,
                    output_json=args.output_json or "data/card2tab2card_results.json",
                    db_path=args.db_path
                )
            
            print(f"Found {len(results)} similar model cards for {args.model_id}:")
            for i, model_id in enumerate(results, 1):
                print(f"  {i}. {model_id}")

        elapsed = time.time() - start_time
        print(f"\n⏱️ Total time: {elapsed:.2f}s")
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

