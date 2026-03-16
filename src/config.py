"""
Unified config for data paths.

- Relationship / raw / processed inputs default to ModelTables data.
- Output artifacts default to this repo's own data directory.
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

DEDUPED_HUGGING_CSVS =  os.path.join(PROCESSED_DIR, f"deduped_hugging_csvs{V2_SUFFIX}{DATA_TAG}")
DEDUPED_GITHUB_CSVS =   os.path.join(PROCESSED_DIR, f"deduped_github_csvs{V2_SUFFIX}{DATA_TAG}")
TABLES_OUTPUT = os.path.join(PROCESSED_DIR, f"tables_output{V2_SUFFIX}{DATA_TAG}")
TABLE_BASE_DIRS = [DEDUPED_HUGGING_CSVS, DEDUPED_GITHUB_CSVS, TABLES_OUTPUT]

# --- Relationship parquet (model–table mapping) ---
RELATIONSHIP_PARQUET = os.path.join(PROCESSED_DIR, f"modelcard_step3_dedup{V2_SUFFIX}{DATA_TAG}.parquet")

# --- Output / artifact paths (single source: change here only) ---
# Default outputs go to ModelSearchDemo/data rather than ../ModelTables/data.
OUTPUT_DIR = str(REPO_ROOT / "data")
# Repo-local outputs (e.g. card2card build-index, search results, temp artifacts)
CLASSIFICATION_JSON            = os.path.join(OUTPUT_DIR, "table_classifications.json")
SCHEMA_LOG                     = os.path.join(os.path.dirname(MODELTABLES_DATA), "logs", "parquet_schema.log")
CARD2TAB2CARD_OUTPUT_JSON      = os.path.join(OUTPUT_DIR, "card2tab2card_results.json")
TAB2TAB_OUTPUT_JSON            = os.path.join(OUTPUT_DIR, "tab2tab_results.json")
TAB2TAB_BY_TYPE_OUTPUT_JSON    = os.path.join(OUTPUT_DIR, "tab2tab_by_type_results.json")
CARD2TAB2CARD_BY_TYPE_STANDALONE_JSON = os.path.join(OUTPUT_DIR, "card2tab2card_by_type_standalone.json")
CLASSIFICATION_OUTPUT_JSON     = CLASSIFICATION_JSON
EMB_NPZ                        = os.path.join(OUTPUT_DIR, "card2card_embeddings.npz")
FAISS_INDEX                    = os.path.join(OUTPUT_DIR, "card2card.faiss")
SPARSE_INDEX                   = os.path.join(OUTPUT_DIR, "card2card_sparse_index")
CARD2CARD_CORPUS_JSONL         = os.path.join(OUTPUT_DIR, "card2card_corpus.jsonl")
CARD2CARD_SPARSE_CORPUS        = os.path.join(OUTPUT_DIR, "card2card_sparse_corpus")
CARD2CARD_NEIGHBORS_JSON       = os.path.join(OUTPUT_DIR, "card2card_neighbors.json")

PRESET_QUERIES_PATH = os.path.join(REPO_ROOT, "config", "preset_queries.json")

CARD2TAB2CARD_TIMEOUT = 600
USE_BY_TYPE = False
JOBS_DIR = os.path.join(OUTPUT_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# --- Other repositories ---
DIALITE_INTERNAL_REPO = os.path.join(REPO_ROOT, "others", "dialite")
BLEND_INTERNAL_REPO = os.path.join(REPO_ROOT, "others", "Blend_internal")
TAB2KNOW_REPO = os.path.join(REPO_ROOT, "others", "tab2know")
CARD2CARD_MODES = ["dense", "sparse", "hybrid"]
CARD2TAB2CARD_TYPES = ["keyword", "single_column", "unionable"]

# --- Data Structures ---
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
QUERY_RESULT_PATH = os.path.join(RESULT_DIR, "query_to_modelids.parquet")
QUERY_TIME_PATH = os.path.join(RESULT_DIR, "past_time_queries.parquet")
QUERY_MODEL_ID_PATH = os.path.join(RESULT_DIR, "model_to_modelids.parquet")
TABLE_RESULT_PATH = os.path.join(RESULT_DIR, "table_to_tables.parquet")
LOG_PATH = os.path.join(RESULT_DIR, "query_to_logs.parquet")

# --- 
VALID_MODEL_IDS_TXT = os.path.join(OUTPUT_DIR, "valid_model_ids_with_tables.txt")

def abs_path(relative_path: str) -> str:
    """Return absolute path for a path that may be relative to repo root."""
    p = relative_path
    if os.path.isabs(p):
        return p
    return str(REPO_ROOT / p)
