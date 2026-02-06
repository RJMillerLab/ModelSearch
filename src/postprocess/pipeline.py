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


def csv_search_dirs() -> List[Path]:
    """Dirs to search for CSV by basename: data_citationlake/processed, data/raw, ../ModelTables/data/processed."""
    dirs: List[Path] = [
        _REPO_ROOT / "data_citationlake/processed/deduped_hugging_csvs",
        _REPO_ROOT / "data_citationlake/processed/deduped_github_csvs",
        _REPO_ROOT / "data_citationlake/processed/tables_output",
        _REPO_ROOT / "data_citationlake/processed",
        _REPO_ROOT / "data/raw",
    ]
    mt = _REPO_ROOT.parent / "ModelTables" / "data" / "processed"
    if mt.exists():
        for sub in ("deduped_hugging_csvs", "deduped_github_csvs", "tables_output", ""):
            p = mt / sub if sub else mt
            if p.exists() and p.is_dir():
                dirs.append(p)
    return dirs


def find_csv_file(filename: str) -> Optional[str]:
    """Resolve CSV path by basename under repo, data_citationlake/processed, and ../ModelTables/data/processed."""
    basename = os.path.basename(filename)
    for d in csv_search_dirs():
        p = d / basename
        if p.exists():
            return str(p)
    if os.path.exists(filename):
        return filename
    return None


def is_model_search_log(log_name: str) -> bool:
    """True if this log is from a model-search pipeline (models first, then related tables)."""
    return any(kw in log_name for kw in MODEL_SEARCH_LOG_KEYWORDS)


def is_table_search_log(log_name: str) -> bool:
    """True if this log is from a table-search pipeline (tables only)."""
    return any(kw in log_name for kw in TABLE_SEARCH_LOG_KEYWORDS)


def get_repo_root() -> Path:
    """Repo root (ModelSearchDemo)."""
    return _REPO_ROOT
