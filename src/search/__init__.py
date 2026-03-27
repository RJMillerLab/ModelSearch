"""
Search module for ModelSearch

Public surface (lazy where noted):
- card2card / ir_index_builder: index build; batch search in card2card_batch
- tab2tab: Blend_internal table search helpers
- Query2ModelCardSearch: text query → model card ids (dense/sparse/hybrid)
- Card2Tab2CardSearch: model card → related tables → tab2tab → model cards
- Query2Tab2CardSearch: query embedding search + card2tab2card + rerank (see query2tab2card.py)
"""

import importlib
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

# Lazy symbols: (module path, attribute name). Keeps package import light (no Blend/FAISS at import).
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # tab2tab
    "search_table2table": ("src.search.tab2tab", "search_table2table"),
    "search_table2table_with_scores": ("src.search.tab2tab", "search_table2table_with_scores"),
    "Query2ModelCardSearch": ("src.search.query2modelcard", "Query2ModelCardSearch"),
    "Card2Tab2CardSearch": ("src.search.card2tab2card", "Card2Tab2CardSearch"),
    "Query2Tab2CardSearch": ("src.search.query2tab2card", "Query2Tab2CardSearch"),
}


def __getattr__(name: str):
    spec = _LAZY_EXPORTS.get(name)
    if spec is not None:
        mod_path, attr = spec
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # tab2tab (lazy)
    "search_table2table",
    "search_table2table_with_scores",
    # pipelines (lazy)
    "Query2ModelCardSearch",
    "Card2Tab2CardSearch",
    "Query2Tab2CardSearch",
    # classification
    "classify_table",
    "classify_table_from_db",
    "classify_datalake_batch",
    "load_classifications",
    "get_known_classes",
    "infer_classification_method",
]
