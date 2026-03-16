"""
Table to Table Search (Testing Tool)

This module provides functions for table-to-table search using Blend_internal.
Wraps and reuses functionality from Blend_internal/src/Tasks.
Supports reading tables from modellake.db for testing.
"""

import os
import sys
import time
from typing import List, Dict, Optional, Iterable, Any
import pandas as pd
import numpy as np
import argparse
import duckdb

from src.config import MODELLAKE_DB, TAB2TAB_OUTPUT_JSON, BLEND_INTERNAL_REPO

blend_path_abs = os.path.abspath(BLEND_INTERNAL_REPO)
if blend_path_abs in sys.path:
    sys.path.remove(blend_path_abs)
sys.path.insert(0, blend_path_abs)

_SingleColumnJoinSearch = None
_MultiColumnJoinSearch = None
_KeywordSearch = None
_UnionSearch = None
_ComplexSearch = None
_CorrelationSearch = None
_DataImputation = None
_AugmentationByExample = None
_DependentDataSearch = None
_FeatureForMLSearch = None
_MultiColumnCollinearitySearch = None
_NegativeExampleSearch = None

import threading
_import_lock = threading.Lock()
_config_lock = threading.Lock()
_blend_config_debug_printed = False

def _update_blend_config(db_path: str):
    """Update Blend_internal config.ini with the correct db_path before importing."""
    with _config_lock:  # Thread-safe config update
        if _BLEND_INTERNAL_PATH is None:
            raise FileNotFoundError("Blend_internal not found. Please clone it first: git clone git@github.com:DoraDong-2023/Blend_internal.git src/Blend_internal")
        
        import configparser
        config_path = os.path.join(_BLEND_INTERNAL_PATH, 'config', 'config.ini')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Blend_internal config.ini not found at {config_path}")
        
        # Read and update config
        config = configparser.ConfigParser()
        read_ok = config.read(config_path)
        
        # Convert relative path to absolute if needed
        if not os.path.isabs(db_path):
            # Make path relative to ModelSearchDemo root
            modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            db_path_abs = os.path.abspath(os.path.join(modelsearch_root, db_path))
        else:
            db_path_abs = os.path.abspath(db_path)

        # DEBUG: print which config path is used and what it contains.
        # Print only once per process to avoid log spam (card2tab2card runs per-table search in parallel).
        global _blend_config_debug_printed
        if not _blend_config_debug_printed:
            try:
                st = os.stat(config_path)
                size = st.st_size
                mtime = st.st_mtime
            except Exception:
                size = None
                mtime = None
            try:
                sections = config.sections()
            except Exception:
                sections = []
            print(f"[BLEND_CONFIG_DEBUG] blend_path={_BLEND_INTERNAL_PATH}", flush=True)
            print(f"[BLEND_CONFIG_DEBUG] cwd={os.getcwd()}", flush=True)
            print(f"[BLEND_CONFIG_DEBUG] config_path={config_path} exists={os.path.exists(config_path)} size={size} mtime={mtime}", flush=True)
            print(f"[BLEND_CONFIG_DEBUG] config.read() returned={read_ok}", flush=True)
            print(f"[BLEND_CONFIG_DEBUG] sections={sections}", flush=True)
            if "Database" in config:
                db_items = dict(config.items("Database"))
                # Avoid extremely long prints; show only the keys we care about.
                print(f"[BLEND_CONFIG_DEBUG] Database.dbms={db_items.get('dbms')} Database.path={db_items.get('path')} Database.index_table={db_items.get('index_table')}", flush=True)
            else:
                print(f"[BLEND_CONFIG_DEBUG] Database section MISSING in parsed config", flush=True)
            _blend_config_debug_printed = True
        
        # Update config
        if 'Database' not in config:
            config['Database'] = {}
        config['Database']['path'] = db_path_abs
        config['Database']['dbms'] = 'duckdb'
        config['Database']['index_table'] = 'modellake_index'
        
        # Write back
        # Write atomically to avoid other processes reading partial file contents.
        tmp_path = config_path + ".tmp"
        with open(tmp_path, 'w') as f:
            config.write(f)
        os.replace(tmp_path, config_path)
        
        return config_path

def _lazy_import_blend():
    """Lazy import Blend_internal functions after config is set."""
    global _SingleColumnJoinSearch, _MultiColumnJoinSearch, _KeywordSearch, _UnionSearch, _ComplexSearch, _CorrelationSearch, _DataImputation, _AugmentationByExample, _DependentDataSearch, _FeatureForMLSearch, _MultiColumnCollinearitySearch, _NegativeExampleSearch
    # Use double-checked locking pattern for thread safety
    if _SingleColumnJoinSearch is None:
        with _import_lock:
            # Check again after acquiring lock (double-checked locking)
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
                        'src.Tasks.UnionSearch',
                        'src.Tasks.ComplexSearch',
                        'src.Tasks.CorrelationSearch',
                        'src.Tasks.DataImputation',
                        'src.Tasks.AugmentationByExample',
                        'src.Tasks.DependentDataSearch',
                        'src.Tasks.FeatureForMLSearch',
                        'src.Tasks.MultiColumnCollinearitySearch',
                        'src.Tasks.NegativeExampleSearch',
                        'src.Plan',
                        'src.Operators',
                        'src.Operators.OperatorBase',
                        'src.Operators.Seekers',
                        'src.Operators.Seekers.MultiColumnOverlap',
                        'src.Operators.Seekers.Correlation',
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
                    from src.Tasks.UnionSearch import UnionSearch
                    from src.Tasks.ComplexSearch import ComplexSearch
                    from src.Tasks.CorrelationSearch import CorrelationSearch
                    from src.Tasks.DataImputation import DataImputation
                    from src.Tasks.AugmentationByExample import AugmentationByExample
                    from src.Tasks.DependentDataSearch import DependentDataSearch
                    from src.Tasks.FeatureForMLSearch import FeatureForMLSearch
                    from src.Tasks.MultiColumnCollinearitySearch import MultiColumnCollinearitySearch
                    from src.Tasks.NegativeExampleSearch import NegativeExampleSearch
                    _SingleColumnJoinSearch = SingleColumnJoinSearch
                    _MultiColumnJoinSearch = MultiColumnJoinSearch
                    _KeywordSearch = KeywordSearch
                    _UnionSearch = UnionSearch
                    _ComplexSearch = ComplexSearch
                    _CorrelationSearch = CorrelationSearch
                    _DataImputation = DataImputation
                    _AugmentationByExample = AugmentationByExample
                    _DependentDataSearch = DependentDataSearch
                    _FeatureForMLSearch = FeatureForMLSearch
                    _MultiColumnCollinearitySearch = MultiColumnCollinearitySearch
                    _NegativeExampleSearch = NegativeExampleSearch
                finally:
                    # Restore ModelSearchDemo root to sys.path if we removed it
                    if modelsearch_removed and modelsearch_root not in sys.path:
                        sys.path.append(modelsearch_root)

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
        default_path = MODELLAKE_DB
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
        default_path = MODELLAKE_DB
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
        default_path = MODELLAKE_DB
        modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        default_path_abs = os.path.abspath(os.path.join(modelsearch_root, default_path))
        _update_blend_config(default_path_abs)
    _lazy_import_blend()
    plan = _KeywordSearch(query_values, k)
    return plan.run()


def search_unionable(
    query_dataset: pd.DataFrame,
    k: int = 10,
    db_path: Optional[str] = None
) -> List[int]:
    """
    Search for unionable tables (tables with unionable columns).
    
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
        default_path = MODELLAKE_DB
        modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        default_path_abs = os.path.abspath(os.path.join(modelsearch_root, default_path))
        _update_blend_config(default_path_abs)
    _lazy_import_blend()
    plan = _UnionSearch(query_dataset, k)
    return plan.run()


def search_complex(
    examples: pd.DataFrame,
    queries: Optional[Iterable[str]] = None,
    target: Optional[Iterable[float]] = None,
    k: int = 10,
    db_path: Optional[str] = None
) -> List[int]:
    """
    Complex search combining union, join, and correlation sub-pipelines.
    
    Args:
        examples: DataFrame with example data (used for union and join sub-pipelines)
        queries: Optional iterable of query strings (currently not used in ComplexSearch implementation)
        target: Optional iterable of target numeric values for correlation search
               If None, will try to auto-detect numeric column from examples
        k: Number of results to return
        db_path: Path to modellake.db (optional, will use config default if not provided)
    
    Returns:
        List of table IDs (integers)
    """
    # Always update config before importing
    if db_path:
        _update_blend_config(db_path)
    else:
        default_path = MODELLAKE_DB
        modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        default_path_abs = os.path.abspath(os.path.join(modelsearch_root, default_path))
        _update_blend_config(default_path_abs)
    
    _lazy_import_blend()
    
    # Auto-detect target if not provided
    if target is None:
        # Try to find a numeric column in examples
        numeric_cols = examples.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Use first numeric column as target
            target = examples[numeric_cols[0]].tolist()
            print(f"ℹ️  Auto-detected target column: {numeric_cols[0]}")
        else:
            # If no numeric column, create dummy target (all zeros)
            # This will make correlation search less effective but won't fail
            target = [0.0] * len(examples)
            print(f"⚠️  No numeric column found, using dummy target values")
    
    # Use first column as queries if not provided
    if queries is None:
        queries = examples[examples.columns[0]].astype(str).tolist()
    
    plan = _ComplexSearch(examples, queries, target, k)
    return plan.run()


def search_correlation(
    source_column: Iterable[str],
    target_column: Iterable[float],
    k: int = 10,
    db_path: Optional[str] = None
) -> List[int]:
    """
    Correlation search - finds tables with correlated categorical and numerical columns.
    
    Args:
        source_column: List of categorical/string values (source column)
        target_column: List of numeric values (target column) - must match source_column length
        k: Number of results to return
        db_path: Path to modellake.db (optional, will use config default if not provided)
    
    Returns:
        List of table IDs (integers)
    """
    # Always update config before importing
    if db_path:
        _update_blend_config(db_path)
    else:
        default_path = MODELLAKE_DB
        modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        default_path_abs = os.path.abspath(os.path.join(modelsearch_root, default_path))
        _update_blend_config(default_path_abs)
    
    _lazy_import_blend()
    
    # Convert to lists if needed
    source_list = list(source_column)
    target_list = list(target_column)
    
    # Validate lengths match
    if len(source_list) != len(target_list):
        raise ValueError(f"Source and target columns must have same length. Got {len(source_list)} and {len(target_list)}")
    
    plan = _CorrelationSearch(source_list, target_list, k)
    return plan.run()


def search_imputation(
    examples: pd.DataFrame,
    queries: Optional[Iterable[str]] = None,
    k: int = 10,
    db_path: Optional[str] = None
) -> List[int]:
    """
    Data Imputation search: find tables that can fill missing values based on examples.
    
    Args:
        examples: DataFrame with example data (rows with complete data)
        queries: Optional iterable of query strings (values to fill)
                 If None, will extract from examples DataFrame (first column values where second column is null)
        k: Number of results to return
        db_path: Path to modellake.db (optional, will use config default if not provided)
    
    Returns:
        List of table IDs (integers)
    """
    # Always update config before importing
    if db_path:
        _update_blend_config(db_path)
    else:
        default_path = MODELLAKE_DB
        modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        default_path_abs = os.path.abspath(os.path.join(modelsearch_root, default_path))
        _update_blend_config(default_path_abs)
    
    _lazy_import_blend()
    
    # If queries not provided, extract from examples DataFrame
    if queries is None:
        if len(examples.columns) < 2:
            raise ValueError("For imputation, examples DataFrame must have at least 2 columns")
        # Extract examples (rows with complete data) and queries (first column where second is null)
        # Examples: rows where second column is not null
        examples_df = examples[examples.iloc[:, 1].notna()].iloc[:, :2].copy()
        # Queries: first column values where second column is null
        queries_list = examples[examples.iloc[:, 1].isna()].iloc[:, 0].astype(str).tolist()
        
        if len(examples_df) == 0:
            raise ValueError("No examples found in DataFrame (no rows with complete data)")
        if len(queries_list) == 0:
            raise ValueError("No queries found in DataFrame (no rows with missing data)")
        
        examples = examples_df
        queries = queries_list
    
    # Convert queries to list if needed
    queries_list = list(queries) if not isinstance(queries, list) else queries
    
    plan = _DataImputation(examples, queries_list, k)
    return plan.run()


def search_augmentation(
    examples: pd.DataFrame,
    queries: Optional[Iterable[str]] = None,
    k: int = 10,
    db_path: Optional[str] = None
) -> List[int]:
    """
    Augmentation by Example search: find tables that can augment data based on examples.
    
    Args:
        examples: DataFrame with example data (rows with complete data)
        queries: Optional iterable of query strings (values to augment)
                 If None, will extract from examples DataFrame (first column values where second column is null)
        k: Number of results to return
        db_path: Path to modellake.db (optional, will use config default if not provided)
    
    Returns:
        List of table IDs (integers)
    """
    # Always update config before importing
    if db_path:
        _update_blend_config(db_path)
    else:
        default_path = MODELLAKE_DB
        modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        default_path_abs = os.path.abspath(os.path.join(modelsearch_root, default_path))
        _update_blend_config(default_path_abs)
    
    _lazy_import_blend()
    
    # If queries not provided, extract from examples DataFrame
    if queries is None:
        if len(examples.columns) < 2:
            raise ValueError("For augmentation, examples DataFrame must have at least 2 columns")
        # Extract examples (rows with complete data) and queries (first column where second is null)
        # Examples: rows where second column is not null
        examples_df = examples[examples.iloc[:, 1].notna()].iloc[:, :2].copy()
        # Queries: first column values where second column is null
        queries_list = examples[examples.iloc[:, 1].isna()].iloc[:, 0].astype(str).tolist()
        
        if len(examples_df) == 0:
            raise ValueError("No examples found in DataFrame (no rows with complete data)")
        if len(queries_list) == 0:
            raise ValueError("No queries found in DataFrame (no rows with missing data)")
        
        examples = examples_df
        queries = queries_list
    
    # Convert queries to list if needed
    queries_list = list(queries) if not isinstance(queries, list) else queries
    
    plan = _AugmentationByExample(examples, queries_list, k)
    return plan.run()


def search_dependent_data(
    query_dataset: pd.DataFrame,
    dependent_column_names_1: Optional[List[str]] = None,
    dependent_column_names_2: Optional[List[str]] = None,
    k: int = 10,
    db_path: Optional[str] = None
) -> List[int]:
    """
    Dependent Data Search: find tables with dependent column pairs.
    
    Args:
        query_dataset: DataFrame with query data
        dependent_column_names_1: First pair of dependent column names (default: first 2 columns)
        dependent_column_names_2: Second pair of dependent column names (default: columns 2-3)
        k: Number of results to return
        db_path: Path to modellake.db (optional)
    
    Returns:
        List of table IDs (integers)
    """
    if db_path:
        _update_blend_config(db_path)
    else:
        default_path = MODELLAKE_DB
        modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        default_path_abs = os.path.abspath(os.path.join(modelsearch_root, default_path))
        _update_blend_config(default_path_abs)
    
    _lazy_import_blend()
    
    # Auto-extract column names if not provided
    if dependent_column_names_1 is None:
        if len(query_dataset.columns) < 2:
            raise ValueError("For dependent_data search, DataFrame must have at least 2 columns")
        dependent_column_names_1 = [query_dataset.columns[0], query_dataset.columns[1]]
    
    if dependent_column_names_2 is None:
        if len(query_dataset.columns) < 4:
            # Use columns 0-1 and 2-3 if available, otherwise use columns 0-1 twice
            if len(query_dataset.columns) >= 3:
                dependent_column_names_2 = [query_dataset.columns[1], query_dataset.columns[2]]
            else:
                dependent_column_names_2 = dependent_column_names_1.copy()
        else:
            dependent_column_names_2 = [query_dataset.columns[2], query_dataset.columns[3]]
    
    plan = _DependentDataSearch(query_dataset, dependent_column_names_1, dependent_column_names_2, k)
    return plan.run()


def search_feature_for_ml(
    query_dataset: pd.DataFrame,
    source_column_name: Optional[str] = None,
    target_column_name: Optional[str] = None,
    numerical_feature_column_name: Optional[str] = None,
    k: int = 10,
    db_path: Optional[str] = None
) -> List[int]:
    """
    Feature for ML Search: find columns correlated with target but not with numerical feature.
    
    Args:
        query_dataset: DataFrame with query data
        source_column_name: Source column name (default: first column)
        target_column_name: Target column name (default: first numeric column)
        numerical_feature_column_name: Numerical feature column name (default: second numeric column)
        k: Number of results to return
        db_path: Path to modellake.db (optional)
    
    Returns:
        List of table IDs (integers)
    """
    if db_path:
        _update_blend_config(db_path)
    else:
        default_path = MODELLAKE_DB
        modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        default_path_abs = os.path.abspath(os.path.join(modelsearch_root, default_path))
        _update_blend_config(default_path_abs)
    
    _lazy_import_blend()
    
    # Auto-extract column names if not provided
    if source_column_name is None:
        source_column_name = query_dataset.columns[0]
    
    numeric_cols = query_dataset.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        raise ValueError("For feature_for_ml search, DataFrame must have at least 2 numeric columns")
    
    if target_column_name is None:
        target_column_name = numeric_cols[0]
    
    if numerical_feature_column_name is None:
        numerical_feature_column_name = numeric_cols[1]
    
    plan = _FeatureForMLSearch(query_dataset, source_column_name, target_column_name, numerical_feature_column_name, k)
    return plan.run()


def search_multi_column_collinearity(
    query_dataset: pd.DataFrame,
    source_column_name: Optional[str] = None,
    target_column_name: Optional[str] = None,
    numerical_feature_column_name: Optional[str] = None,
    multi_column_column_names: Optional[List[str]] = None,
    k: int = 10,
    db_path: Optional[str] = None
) -> List[int]:
    """
    Multi-Column Collinearity Search: find tables with correlated columns and multi-column overlap.
    
    Args:
        query_dataset: DataFrame with query data
        source_column_name: Source column name (default: first column)
        target_column_name: Target column name (default: first numeric column)
        numerical_feature_column_name: Numerical feature column name (default: second numeric column)
        multi_column_column_names: Multi-column names for overlap (default: first 2 columns)
        k: Number of results to return
        db_path: Path to modellake.db (optional)
    
    Returns:
        List of table IDs (integers)
    """
    if db_path:
        _update_blend_config(db_path)
    else:
        default_path = MODELLAKE_DB
        modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        default_path_abs = os.path.abspath(os.path.join(modelsearch_root, default_path))
        _update_blend_config(default_path_abs)
    
    _lazy_import_blend()
    
    # Auto-extract column names if not provided
    if source_column_name is None:
        source_column_name = query_dataset.columns[0]
    
    numeric_cols = query_dataset.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        raise ValueError("For multi_column_collinearity search, DataFrame must have at least 2 numeric columns")
    
    if target_column_name is None:
        target_column_name = numeric_cols[0]
    
    if numerical_feature_column_name is None:
        numerical_feature_column_name = numeric_cols[1]
    
    if multi_column_column_names is None:
        if len(query_dataset.columns) < 2:
            raise ValueError("For multi_column_collinearity search, DataFrame must have at least 2 columns")
        multi_column_column_names = [query_dataset.columns[0], query_dataset.columns[1]]
    
    plan = _MultiColumnCollinearitySearch(query_dataset, source_column_name, target_column_name, numerical_feature_column_name, multi_column_column_names, k)
    return plan.run()


def search_negative_example(
    inclusive_df: pd.DataFrame,
    exclusive_df: Optional[pd.DataFrame] = None,
    inclusive_column_name_1: Optional[str] = None,
    inclusive_column_name_2: Optional[str] = None,
    exclusive_column_name_1: Optional[str] = None,
    exclusive_column_name_2: Optional[str] = None,
    k: int = 10,
    db_path: Optional[str] = None
) -> List[int]:
    """
    Negative Example Search: find tables with exclusive but not inclusive examples.
    
    Args:
        inclusive_df: DataFrame with inclusive examples
        exclusive_df: DataFrame with exclusive examples (if None, will split inclusive_df)
        inclusive_column_name_1: First inclusive column name (default: first column)
        inclusive_column_name_2: Second inclusive column name (default: second column)
        exclusive_column_name_1: First exclusive column name (default: first column)
        exclusive_column_name_2: Second exclusive column name (default: second column)
        k: Number of results to return
        db_path: Path to modellake.db (optional)
    
    Returns:
        List of table IDs (integers)
    """
    if db_path:
        _update_blend_config(db_path)
    else:
        default_path = MODELLAKE_DB
        modelsearch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        default_path_abs = os.path.abspath(os.path.join(modelsearch_root, default_path))
        _update_blend_config(default_path_abs)
    
    _lazy_import_blend()
    
    # If exclusive_df not provided, split inclusive_df (use first half as inclusive, second half as exclusive)
    if exclusive_df is None:
        if len(inclusive_df) < 2:
            raise ValueError("For negative_example search, DataFrame must have at least 2 rows to split")
        mid = len(inclusive_df) // 2
        exclusive_df = inclusive_df.iloc[mid:].copy()
        inclusive_df = inclusive_df.iloc[:mid].copy()
    
    # Auto-extract column names if not provided
    if inclusive_column_name_1 is None:
        inclusive_column_name_1 = inclusive_df.columns[0]
    if inclusive_column_name_2 is None:
        if len(inclusive_df.columns) < 2:
            raise ValueError("For negative_example search, inclusive DataFrame must have at least 2 columns")
        inclusive_column_name_2 = inclusive_df.columns[1]
    
    if exclusive_column_name_1 is None:
        exclusive_column_name_1 = exclusive_df.columns[0]
    if exclusive_column_name_2 is None:
        if len(exclusive_df.columns) < 2:
            raise ValueError("For negative_example search, exclusive DataFrame must have at least 2 columns")
        exclusive_column_name_2 = exclusive_df.columns[1]
    
    plan = _NegativeExampleSearch(inclusive_df, inclusive_column_name_1, inclusive_column_name_2, exclusive_df, exclusive_column_name_1, exclusive_column_name_2, k)
    return plan.run()


def search_table2table(
    query: Any,
    search_type: str = "single_column",
    k: int = 10,
    db_path: Optional[str] = None,
    target: Optional[Iterable[float]] = None,
    source_column: Optional[Iterable[str]] = None,
    target_column: Optional[Iterable[float]] = None
) -> List[int]:
    """
    Unified interface for table-to-table search.
    
    Args:
        query: Query data - can be:
            - Iterable of values (for single_column)
            - pd.DataFrame (for multi_column, complex, or correlation)
            - List[str] (for keyword)
        search_type: Type of search - "single_column", "multi_column", "keyword", "unionable", "complex", "correlation", "imputation", "augmentation", "dependent_data", "feature_for_ml", "multi_column_collinearity", or "negative_example"
        k: Number of results to return
        db_path: Path to modellake.db (optional)
        target: Optional target values for complex search (iterable of floats)
        source_column: Optional source column for correlation search (iterable of strings)
        target_column: Optional target column for correlation search (iterable of floats)
    
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
    elif search_type == "unionable":
        if not isinstance(query, pd.DataFrame):
            raise ValueError("For unionable search, query must be a pandas DataFrame")
        return search_unionable(query, k, db_path=db_path)
    elif search_type == "complex":
        if not isinstance(query, pd.DataFrame):
            raise ValueError("For complex search, query must be a pandas DataFrame")
        return search_complex(query, target=target, k=k, db_path=db_path)
    elif search_type == "correlation":
        if source_column is None or target_column is None:
            # Try to extract from DataFrame if query is DataFrame
            if isinstance(query, pd.DataFrame):
                # Use first column as source, first numeric column as target
                source_column = query[query.columns[0]].astype(str).tolist()
                numeric_cols = query.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    target_column = query[numeric_cols[0]].tolist()
                else:
                    raise ValueError("For correlation search with DataFrame, at least one numeric column is required")
            else:
                raise ValueError("For correlation search, either provide source_column and target_column, or a DataFrame with numeric columns")
        return search_correlation(source_column, target_column, k, db_path=db_path)
    elif search_type == "imputation":
        if not isinstance(query, pd.DataFrame):
            raise ValueError("For imputation search, query must be a pandas DataFrame")
        return search_imputation(query, k=k, db_path=db_path)
    elif search_type == "augmentation":
        if not isinstance(query, pd.DataFrame):
            raise ValueError("For augmentation search, query must be a pandas DataFrame")
        return search_augmentation(query, k=k, db_path=db_path)
    elif search_type == "dependent_data":
        if not isinstance(query, pd.DataFrame):
            raise ValueError("For dependent_data search, query must be a pandas DataFrame")
        return search_dependent_data(query, k=k, db_path=db_path)
    elif search_type == "feature_for_ml":
        if not isinstance(query, pd.DataFrame):
            raise ValueError("For feature_for_ml search, query must be a pandas DataFrame")
        return search_feature_for_ml(query, k=k, db_path=db_path)
    elif search_type == "multi_column_collinearity":
        if not isinstance(query, pd.DataFrame):
            raise ValueError("For multi_column_collinearity search, query must be a pandas DataFrame")
        return search_multi_column_collinearity(query, k=k, db_path=db_path)
    elif search_type == "negative_example":
        if not isinstance(query, pd.DataFrame):
            raise ValueError("For negative_example search, query must be a pandas DataFrame")
        return search_negative_example(query, k=k, db_path=db_path)
    else:
        raise ValueError(f"Unknown search_type: {search_type}. Must be 'single_column', 'multi_column', 'keyword', 'unionable', 'complex', 'correlation', 'imputation', 'augmentation', 'dependent_data', 'feature_for_ml', 'multi_column_collinearity', or 'negative_example'")


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
    parser.add_argument('--search_type', choices=['single_column', 'multi_column', 'keyword', 'unionable'], default='single_column', help='Type of search to perform')
    parser.add_argument('--query', default=None, help='Query data. For single_column: comma-separated values. For multi_column: path to CSV file. For keyword: comma-separated keywords. For unionable: path to CSV file.')
    parser.add_argument('--k', type=int, default=10, help='Number of results to return')
    parser.add_argument('--test_table_id', type=int, default=None, help='Test with a specific table ID from modellake.db')
    args = parser.parse_args()
    start_time = time.time()
    db_path = MODELLAKE_DB
    output_path = TAB2TAB_OUTPUT_JSON
    _update_blend_config(db_path)
    if args.query is None and args.test_table_id is None:
        parser.error("Either --query or --test_table_id must be provided")
    if args.query:
        if args.search_type == 'single_column':
            query = [x.strip() for x in args.query.split(',')]
        elif args.search_type == 'multi_column':
            query = pd.read_csv(args.query)
        elif args.search_type == 'unionable':
            query = pd.read_csv(args.query)
        elif args.search_type == 'keyword':
            query = [x.strip() for x in args.query.split(',')]
    else:
        query = [str(args.test_table_id)]
        args.search_type = 'keyword'
    results = search_table2table(query, args.search_type, args.k, db_path=db_path)
    print(f"Found {len(results)} tables:")
    for i, table_id in enumerate(results, 1):
        print(f"  {i}. Table ID: {table_id}")
    
    # Save results as JSON
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    result_data = {
        "query": query if isinstance(query, list) else str(query),
        "search_type": args.search_type,
        "k": args.k,
        "results": [int(tid) for tid in results],
        "num_results": len(results)
    }
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Results saved to {output_path}")
    print(f"\nTotal time: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    main()

