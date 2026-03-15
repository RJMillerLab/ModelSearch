"""
Unified config for data paths: ModelTables/data, modellake.db, processed dirs, relationship parquet.
All path usages should import from here. Override via env vars if needed (see below).
"""
import os
from pathlib import Path

# Repo root (ModelSearchDemo)
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Base data root (relative to repo root by default) ---
MODELTABLES_DATA = "../ModelTables/data"
PROCESSED_DIR = os.path.join(MODELTABLES_DATA, "processed")
RAW_DIR = os.path.join(MODELTABLES_DATA, "raw")

# --- Version/tag for processed artifacts ---
DATA_TAG = "_251117"
V2_SUFFIX = "_v2"

# --- modellake.db ---
MODELLAKE_DB = os.path.join(MODELTABLES_DATA, "modellake.db")
INDEX_TABLE = "modellake_index"

# --- Processed CSV/table dirs (deduped_hugging_csvs, deduped_github_csvs, tables_output) ---
DEDUPED_HUGGING_CSVS =  os.path.join(PROCESSED_DIR, f"deduped_hugging_csvs{V2_SUFFIX}{DATA_TAG}")
DEDUPED_GITHUB_CSVS =   os.path.join(PROCESSED_DIR, f"deduped_github_csvs{V2_SUFFIX}{DATA_TAG}")
TABLES_OUTPUT = os.path.join(PROCESSED_DIR, f"tables_output{V2_SUFFIX}{DATA_TAG}")
TABLE_BASE_DIRS = [DEDUPED_HUGGING_CSVS, DEDUPED_GITHUB_CSVS, TABLES_OUTPUT]

# --- Relationship parquet (model–table mapping) ---
RELATIONSHIP_PARQUET = os.path.join(PROCESSED_DIR, f"modelcard_step3_dedup{V2_SUFFIX}{DATA_TAG}.parquet")

# --- Output / artifact paths (single source: change here only) ---
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data")
# Repo-local data/raw (e.g. for card2card build-index when not using ModelTables)
CLASSIFICATION_JSON = os.path.join(OUTPUT_DIR, "table_classifications.json")
SCHEMA_LOG = os.path.join(os.path.dirname(MODELTABLES_DATA), "logs", "parquet_schema.log")
CARD2TAB2CARD_OUTPUT_JSON   = os.path.join(OUTPUT_DIR, "card2tab2card_results.json")
TAB2TAB_OUTPUT_JSON        = os.path.join(OUTPUT_DIR, "tab2tab_results.json")
TAB2TAB_BY_TYPE_OUTPUT_JSON   = os.path.join(OUTPUT_DIR, "tab2tab_by_type_results.json")
CARD2TAB2CARD_BY_TYPE_STANDALONE_JSON = os.path.join(OUTPUT_DIR, "card2tab2card_by_type_standalone.json")
CLASSIFICATION_OUTPUT_JSON      = CLASSIFICATION_JSON


def abs_path(relative_path: str) -> str:
    """Return absolute path for a path that may be relative to repo root."""
    p = relative_path
    if os.path.isabs(p):
        return p
    return str(REPO_ROOT / p)
