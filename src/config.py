"""
Unified config for data paths.

- Relationship / raw / processed inputs default to ModelTables data.
- Output artifacts default to this repo's own data directory.
"""
import os
from pathlib import Path

# Repo root (ModelSearchDemo)
REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# --- Base data root (relative to repo root by default) ---
MODELTABLES_DATA = os.path.join(ROOT_DIR, "ModelTables", "data")
PROCESSED_DIR = os.path.join(MODELTABLES_DATA, "processed")
RAW_DIR = os.path.join(MODELTABLES_DATA, "raw")
ENCODE_MODEL = "all-MiniLM-L6-v2"

# --- Version/tag for processed artifacts ---
DATA_TAG = "_251117"
V2_SUFFIX = "_v2"

# --- modellake.db ---
MODELLAKE_DB = os.path.join(ROOT_DIR, "Blend_internal", "database_251117", "modellake_v2_nomask_251117.db") # because there is no condition on Title, we don't need to use the mask file
MODELLAKE_DB_HUGGING = os.path.join(ROOT_DIR, "Blend_internal", "database_251117", "modellake_v2_251117_nomask_hugging.db")
# Paired DuckDB built from the same corpus with tables row/col transposed (build + import separately; path is conventional).
MODELLAKE_DB_HUGGING_TRANSPOSED = os.path.join(ROOT_DIR, "Blend_internal", "database_251117", "modellake_v2_251117_nomask_hugging_tr.db")
INDEX_TABLE = "modellake_index"

DEDUPED_HUGGING_CSVS =  os.path.join(PROCESSED_DIR, f"deduped_hugging_csvs{V2_SUFFIX}{DATA_TAG}")
DEDUPED_GITHUB_CSVS =   os.path.join(PROCESSED_DIR, f"deduped_github_csvs{V2_SUFFIX}{DATA_TAG}")
TABLES_OUTPUT = os.path.join(PROCESSED_DIR, f"tables_output{V2_SUFFIX}{DATA_TAG}")
TABLE_BASE_DIRS = [DEDUPED_HUGGING_CSVS, DEDUPED_GITHUB_CSVS, TABLES_OUTPUT]

# --- Model card text (step1) — used for card_readme dense index build ---
MODELCARD_STEP1_PARQUET = os.path.join(PROCESSED_DIR, f"modelcard_step1{DATA_TAG}.parquet")

# --- Relationship parquet (model–table mapping) ---
RELATIONSHIP_PARQUET = os.path.join(PROCESSED_DIR, f"modelcard_step3_dedup{V2_SUFFIX}{DATA_TAG}.parquet")

# --- Output / artifact paths (single source: change here only) ---
# Default outputs go to ModelSearchDemo/data rather than ../ModelTables/data.
OUTPUT_DIR = str(REPO_ROOT / f"data{DATA_TAG}")
# Repo-local outputs (e.g. card2card build-index, search results, temp artifacts)
CLASSIFICATION_JSON                       = os.path.join(OUTPUT_DIR, "table_classifications.json")
SCHEMA_LOG                                = os.path.join(ROOT_DIR, "ModelTables", "logs", "parquet_schema.log")
CARD2TAB2CARD_OUTPUT_JSON                 = os.path.join(OUTPUT_DIR, "card2tab2card_results.json")
TAB2TAB_OUTPUT_JSON                       = os.path.join(OUTPUT_DIR, "tab2tab_results.json")
# Augmented 4-lane tab2tab (see src.search.tab2tab_aug); do not reuse TAB2TAB_OUTPUT_JSON.
TAB2TAB_AUG_OUTPUT_JSON                   = os.path.join(OUTPUT_DIR, "tab2tab_aug_results.json")
TAB2TAB_BY_TYPE_OUTPUT_JSON               = os.path.join(OUTPUT_DIR, "tab2tab_by_type_results.json")
CARD2TAB2CARD_BY_TYPE_STANDALONE_JSON     = os.path.join(OUTPUT_DIR, "card2tab2card_by_type_standalone.json")
CLASSIFICATION_OUTPUT_JSON                = os.path.join(OUTPUT_DIR, "table_classifications_output.json")
EMB_NPZ                                   = os.path.join(OUTPUT_DIR, "card2card_embeddings.npz")
EMB_NPZ_HUGGING                           = os.path.join(OUTPUT_DIR, "card2card_embeddings_hugging.npz")
#FAISS_INDEX                               = os.path.join(OUTPUT_DIR, "card2card.faiss")
SPARSE_INDEX                              = os.path.join(OUTPUT_DIR, "card2card_sparse_index")
SPARSE_INDEX_HUGGING            = os.path.join(OUTPUT_DIR, "card2card_sparse_index_hugging")
CARD2CARD_SPARSE_CORPUS_HUGGING         = os.path.join(OUTPUT_DIR, "card2card_sparse_corpus_hugging")
CARD2CARD_CORPUS_JSONL                    = os.path.join(OUTPUT_DIR, "card2card_corpus.jsonl")
CARD2CARD_SPARSE_CORPUS                   = os.path.join(OUTPUT_DIR, "card2card_sparse_corpus")
CARD2CARD_NEIGHBORS_JSON                  = os.path.join(OUTPUT_DIR, "card2card_neighbors.json")
#MODEL_CSVS_PARQUET                        = os.path.join(OUTPUT_DIR, "model_csvs.parquet")

PRESET_QUERIES_PATH = os.path.join(REPO_ROOT, "config", "preset_queries.json")

CARD2TAB2CARD_TIMEOUT = 600


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if not s:
        return default
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


# Card2Tab2Card / query2tab2card: when table resources are hugging-only, call ``search_tab2tab_aug``
# (4-lane + RRF; canonical ``*.csv`` basenames) instead of single ``search_table2table``.
# ``BACKEND_USE_TAB2TAB_AUG=0`` forces classic one-lane tab2tab.
USE_TAB2TAB_AUG = _env_bool("BACKEND_USE_TAB2TAB_AUG", True)

USE_BY_TYPE = False
JOBS_DIR = os.path.join(OUTPUT_DIR, f"jobs{DATA_TAG}")
os.makedirs(JOBS_DIR, exist_ok=True)

# --- Other repositories ---
DIALITE_INTERNAL_REPO = os.path.join(REPO_ROOT, "others", "dialite")
BLEND_INTERNAL_REPO = os.path.join(REPO_ROOT, "others", "Blend_internal")
TAB2KNOW_REPO = os.path.join(REPO_ROOT, "others", "tab2know")
QUERY2MODELCARD_RETRIEVAL_MODES = ["dense", "sparse", "hybrid"]
CARD2TAB2CARD_TYPES = ["keyword", "single_column", "unionable"]

# Model Search (left column / query2modelcard_* fields in job JSON):
# - False (default): neighbors come from in-process query2modelcard (no separate search CLI per mode).
#   Set USE_CARD2CARD_CLI=1 or True to always invoke `python -m src.search.card2card search` (slower).
# Use False or 0 only — strings like "no" are truthy and wrongly enable the CLI.
USE_CARD2CARD_CLI = False

# Table search / Card2Tab2Card post-filtering by table filename source.
# "github", "arxiv", "hugging"
TABLE_RESOURCE_ALLOWLIST = ["hugging"]

# --- Data Structures ---
RESULT_DIR = os.path.join(OUTPUT_DIR, f"results{DATA_TAG}")
QUERY_RESULT_PATH = os.path.join(RESULT_DIR, "query_to_modelids.parquet")
QUERY_TIME_PATH = os.path.join(RESULT_DIR, "past_time_queries.parquet")
QUERY_MODEL_ID_PATH = os.path.join(RESULT_DIR, "model_to_modelids.parquet")
TABLE_RESULT_PATH = os.path.join(RESULT_DIR, "table_to_tables.parquet")
LOG_PATH = os.path.join(RESULT_DIR, "query_to_logs.parquet")

# --- Valid model ID lists (optional; Card2Tab2Card / demo narrow-down)
VALID_MODEL_IDS_TXT = os.path.join(OUTPUT_DIR, "valid_model_ids_with_tables.txt")
# Hugging-only tables subset (matches build_valid_model_ids_txt --resources hugging)
VALID_MODEL_IDS_WITH_TABLES_HUGGING_TXT = os.path.join(OUTPUT_DIR, "valid_model_ids_with_tables_hugging.txt")

# Flattened relationship index (modelId <-> csv_basename) built once from RELATIONSHIP_PARQUET.
# Kept under repo-local data/ per user workflow preference.
MODEL_TO_TABLES_EXPLODE_PARQUET = os.path.join(OUTPUT_DIR, "model_to_tables_explode_v2_251117.parquet")

def abs_path(relative_path: str) -> str:
    """Return absolute path for a path that may be relative to repo root."""
    p = relative_path
    if os.path.isabs(p):
        return p
    return str(REPO_ROOT / p)
