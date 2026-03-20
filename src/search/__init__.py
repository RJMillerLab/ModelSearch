"""
Search module for ModelSearch

Provides various search functions:
- card2card: ModelCard to ModelCard search
- tab2tab: Table to Table search (using Blend_internal)
- query2modelcard: Query text to ModelCard search
- card2tab2card: ModelCard to Table to ModelCard search
"""

import os
import sys

# Ensure repo root is on path so "from src.search.*" works (run as package or from repo root)
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.search.card2card import (
    build_card_index,
    search_dense_neighbors_queries,
    search_sparse_neighbors_queries,
    search_hybrid_neighbors_queries,
)
from src.search.query2modelcard import search_query2modelcard
from src.search.card2tab2card import (
    load_modelid_to_csvlist,
    search_card2tab2card,
    search_card2tab2card_from_tables,
    search_card2tab2card_by_type,
)
from src.search.classification import (
    classify_table,
    classify_table_from_db,
    classify_datalake_batch,
    load_classifications,
    get_known_classes,
    infer_classification_method,
)
from src.search.tab2tab_by_type import search_table2table_by_type

# Lazy import for tab2tab functions to avoid DBHandler initialization
_tab2tab_module = None

def __getattr__(name):
    """Lazy import for tab2tab functions."""
    global _tab2tab_module
    
    tab2tab_functions = {
        'search_single_column',
        'search_multi_column',
        'search_keyword',
        'search_table2table',
        'search_table2table_by_type'
    }
    
    if name in tab2tab_functions:
        if _tab2tab_module is None:
            # Repo root already in sys.path at top of this file; single import, no try/except
            from src.search import tab2tab as _tab2tab_module
        return getattr(_tab2tab_module, name)
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # card2card
    'build_card_index',
    'search_dense_neighbors_queries',
    'search_sparse_neighbors_queries',
    'search_hybrid_neighbors_queries',
    # tab2tab (lazy imported via __getattr__)
    'search_single_column',
    'search_multi_column',
    'search_keyword',
    'search_table2table',
    'search_table2table_by_type',
    # query2modelcard
    'search_query2modelcard',
    # card2tab2card
    'get_tables_for_model',
    'search_card2tab2card',
    'search_card2tab2card_from_tables',
    'search_card2tab2card_by_type',
    # classification
    'classify_table',
    'classify_table_from_db',
    'classify_datalake_batch',
    'load_classifications',
    'get_known_classes',
    'infer_classification_method',
]
