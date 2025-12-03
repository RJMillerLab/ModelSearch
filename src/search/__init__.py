"""
Search module for ModelSearch

Provides various search functions:
- card2card: ModelCard to ModelCard search
- tab2tab: Table to Table search (using Blend_internal)
- query2modelcard: Query text to ModelCard search
- card2tab2card: ModelCard to Table to ModelCard search
"""

# Use relative imports to avoid issues
# Note: tab2tab is NOT imported here to avoid initializing DBHandler on import
# (Blend_internal's OperatorBase initializes DBHandler at class level)
# Import tab2tab functions only when needed (lazy import via __getattr__)

try:
    from .card2card import (
        build_card_index,
        search_card2card,
        search_card2card_batch
    )
    
    from .query2modelcard import (
        search_query2modelcard
    )
    
    from .card2tab2card import (
        load_relationship_parquet,
        get_tables_for_model,
        search_card2tab2card,
        search_card2tab2card_from_tables,
        search_card2tab2card_by_type
    )
    
    from .classification import (
        classify_table,
        classify_table_from_db,
        classify_datalake_batch,
        load_classifications
    )
    
    from .tab2tab_by_type import (
        search_table2table_by_type
    )
except ImportError:
    # Fallback for direct script execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from src.search.card2card import (
        build_card_index,
        search_card2card,
        search_card2card_batch
    )
    
    from src.search.query2modelcard import (
        search_query2modelcard
    )
    
    from src.search.card2tab2card import (
        load_relationship_parquet,
        get_tables_for_model,
        search_card2tab2card,
        search_card2tab2card_from_tables,
        search_card2tab2card_by_type
    )
    
    from src.search.classification import (
        classify_table,
        classify_table_from_db,
        classify_datalake_batch,
        load_classifications
    )
    
    from src.search.tab2tab_by_type import (
        search_table2table_by_type
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
            try:
                from . import tab2tab as _tab2tab_module
            except ImportError:
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
                from src.search import tab2tab as _tab2tab_module
        return getattr(_tab2tab_module, name)
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # card2card
    'build_card_index',
    'search_card2card',
    'search_card2card_batch',
    # tab2tab (lazy imported via __getattr__)
    'search_single_column',
    'search_multi_column',
    'search_keyword',
    'search_table2table',
    'search_table2table_by_type',
    # query2modelcard
    'search_query2modelcard',
    # card2tab2card
    'load_relationship_parquet',
    'get_tables_for_model',
    'search_card2tab2card',
    'search_card2tab2card_from_tables',
    'search_card2tab2card_by_type',
    # classification
    'classify_table',
    'classify_table_from_db',
    'classify_datalake_batch',
    'load_classifications',
]
