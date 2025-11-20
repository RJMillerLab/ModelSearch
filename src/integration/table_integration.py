"""
Table Integration Implementation

Integrates multiple tables using Blend_internal's Union and Intersection operations.
Works with pre-searched table results to avoid re-searching.
"""

import os
import sys
import pandas as pd
import json
from typing import List, Dict, Optional, Any, Set
from collections import Counter

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


def load_table_from_file(table_path: str) -> Optional[pd.DataFrame]:
    """
    Load a table from CSV file.
    
    Args:
        table_path: Path to CSV file
        
    Returns:
        DataFrame or None if file not found
    """
    # Try multiple possible locations
    possible_paths = [
        table_path,
        os.path.join("data_citationlake/processed/deduped_hugging_csvs", os.path.basename(table_path)),
        os.path.join("data_citationlake/processed/deduped_github_csvs", os.path.basename(table_path)),
        os.path.join("data_citationlake/processed/tables_output", os.path.basename(table_path)),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df
            except Exception as e:
                print(f"⚠️  Error loading {path}: {e}")
                continue
    
    print(f"⚠️  Table not found: {table_path}")
    return None


def integrate_tables_union(
    tables: List[pd.DataFrame],
    k: int = 10
) -> Optional[pd.DataFrame]:
    """
    Integrate multiple tables using Union operation.
    
    Args:
        tables: List of DataFrames to integrate
        k: Maximum number of rows to return
        
    Returns:
        Integrated DataFrame or None if integration fails
    """
    if not tables or len(tables) == 0:
        return None
    
    if len(tables) == 1:
        return tables[0].head(k)
    
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
        
        # Union (concatenate) all tables
        integrated = pd.concat(aligned_tables, axis=0, ignore_index=True)
        
        # Remove duplicates if needed
        integrated = integrated.drop_duplicates()
        
        # Limit to k rows
        return integrated.head(k)
    
    except Exception as e:
        print(f"❌ Error in union integration: {e}")
        return None


def integrate_tables_intersection(
    tables: List[pd.DataFrame],
    k: int = 10
) -> Optional[pd.DataFrame]:
    """
    Integrate multiple tables using Intersection operation (find common rows).
    
    Args:
        tables: List of DataFrames to integrate
        k: Maximum number of rows to return
        
    Returns:
        Integrated DataFrame with common rows or None if no common rows
    """
    if not tables or len(tables) == 0:
        return None
    
    if len(tables) == 1:
        return tables[0].head(k)
    
    try:
        # Find common columns
        common_columns = set(tables[0].columns)
        for df in tables[1:]:
            common_columns = common_columns.intersection(set(df.columns))
        
        if not common_columns:
            print("⚠️  No common columns found for intersection")
            return None
        
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
        
        # Limit to k rows
        return result.head(k)
    
    except Exception as e:
        print(f"❌ Error in intersection integration: {e}")
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
        table_paths: List of paths to CSV files
        integration_type: "union" or "intersection"
        k: Maximum number of rows in result
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
    
    # Integrate tables
    if integration_type == "union":
        integrated_df = integrate_tables_union(tables, k)
    elif integration_type == "intersection":
        integrated_df = integrate_tables_intersection(tables, k)
    else:
        return {
            "success": False,
            "error": f"Unknown integration type: {integration_type}",
            "integrated_table": None,
            "stats": {}
        }
    
    if integrated_df is None:
        return {
            "success": False,
            "error": "Integration failed",
            "integrated_table": None,
            "stats": {}
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
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Integrate tables from Card2Tab2Card search results.
    
    Uses the intermediate results (retrieved_table_filenames) from search results
    to integrate tables without re-searching.
    
    Args:
        search_results_json: Path to JSON file with search results
        search_type: Which search type results to use ("single_column", "keyword", "unionable")
        integration_type: "union" or "intersection"
        k: Maximum number of rows in result
        db_path: Optional path to modellake.db
        
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
            if isinstance(card2tab2card_results, dict) and search_type in card2tab2card_results:
                if isinstance(card2tab2card_results[search_type], dict) and "intermediate" in card2tab2card_results[search_type]:
                    intermediate = card2tab2card_results[search_type]["intermediate"]
                else:
                    return {
                        "success": False,
                        "error": f"Search type '{search_type}' not found in results",
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
    retrieved_filenames = intermediate.get("retrieved_table_filenames", [])
    
    if not retrieved_filenames:
        return {
            "success": False,
            "error": "No retrieved tables found in search results",
            "integrated_table": None
        }
    
    print(f"✅ Found {len(retrieved_filenames)} retrieved tables")
    print(f"   Using first {min(k, len(retrieved_filenames))} tables for integration")
    
    # Use first k tables for integration
    table_paths = retrieved_filenames[:k]
    
    # Integrate tables
    return integrate_tables(table_paths, integration_type, k, db_path)

