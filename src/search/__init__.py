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

from src.search.classification import (
    classify_table,
    classify_table_from_db,
    classify_datalake_batch,
    load_classifications,
    get_known_classes,
    infer_classification_method,
)

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

    # Lazy import for card2card so `python -m src.search.card2card ...` does not
    # get its module pre-imported during package init (avoids runpy RuntimeWarning).
    card2card_functions = {
        'build_card_index',
        'search_dense_neighbors_queries',
        'search_sparse_neighbors_queries',
        'search_hybrid_neighbors_queries',
    }
    if name in card2card_functions:
        from src.search import card2card as _card2card_module
        return getattr(_card2card_module, name)

    # Lazy import for query2modelcard
    if name == 'search_query2modelcard':
        from src.search import query2modelcard as _q2m_module
        return getattr(_q2m_module, name)

    # Lazy import for card2tab2card (simplified entrypoints)
    if name == 'search_card2tab2card':
        from src.search import card2tab2card as _c2t2c_module
        return getattr(_c2t2c_module, name)
    
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
    # query2modelcard
    'search_query2modelcard',
    # card2tab2card
    'search_card2tab2card',
    # classification
    'classify_table',
    'classify_table_from_db',
    'classify_datalake_batch',
    'load_classifications',
    'get_known_classes',
    'infer_classification_method',
]
