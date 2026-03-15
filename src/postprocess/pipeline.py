"""
Shared pipeline-type definitions and CSV search paths for parse/postprocess consistency.

Model-search pipelines return models first (primary), then related tables (secondary).
Table-search pipelines return tables only (primary).
"""

import os
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Log name substrings that identify a model-search pipeline (models primary, tables secondary)
MODEL_SEARCH_LOG_KEYWORDS = ("card2card", "card2tab2card", "query2modelcard")

# Log name substrings that identify a table-search pipeline (tables only)
TABLE_SEARCH_LOG_KEYWORDS = ("tab2tab",)

def is_model_search_log(log_name: str) -> bool:
    """True if this log is from a model-search pipeline (models first, then related tables)."""
    return any(kw in log_name for kw in MODEL_SEARCH_LOG_KEYWORDS)


def is_table_search_log(log_name: str) -> bool:
    """True if this log is from a table-search pipeline (tables only)."""
    return any(kw in log_name for kw in TABLE_SEARCH_LOG_KEYWORDS)


def get_repo_root() -> Path:
    """Repo root (ModelSearchDemo)."""
    return _REPO_ROOT
