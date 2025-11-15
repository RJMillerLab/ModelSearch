# ModelSearch

Dense semantic search over Hugging Face model cards with baseline comparison.

## Install

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Clone Blend_internal (Required for Table Search)

The `tab2tab` and `card2tab2card` search functions require Blend_internal for table-to-table search functionality. Clone it into the `src/` directory:

**For private repository (SSH):**
```bash
git clone git@github.com:DoraDong-2023/Blend_internal.git src/Blend_internal
```

**Note:** 
- The `src/Blend_internal/` directory is excluded from this repository (see `.gitignore`). You need to clone it separately for inference.
- For SSH clone, ensure your SSH key is added to GitHub (see [GitHub SSH setup](https://docs.github.com/en/authentication/connecting-to-github-with-ssh))

## Repository Structure

- `src/search/` - Search module with reusable search functions
  - `card2card.py` - ModelCard to ModelCard search
  - `tab2tab.py` - Table to Table search (testing tool, uses Blend_internal)
  - `query2modelcard.py` - Query text to ModelCard search
  - `card2tab2card.py` - ModelCard to Table to ModelCard search
- `src/Blend_internal/` - Table search functionality (requires modellake.db)
  - **Note:** This directory is not included in the repository. Clone it separately:
    ```bash
    git clone https://github.com/DoraDong-2023/Blend_internal.git src/Blend_internal
    ```
- `src/demo/` - Interactive demo frontend and backend
  - `frontend.py` - CLI interactive interface
  - `backend.py` - REST API server
- `data_citationlake/` - Symlink to CitationLake data directory
- `data/` - All output files (embeddings, indices, results)
- `src/modelsearch/` - Legacy search scripts (preserved)

## Usage

**Note: All output files are saved to `data/` directory**

### 1. Build Embeddings (Required First Step)

Build FAISS index from model cards. This must be done before any search operations.

**Important:** This step **does NOT require `modellake.db`**. It only reads parquet files and builds embeddings using SentenceTransformer + FAISS.

**Note:** The `--raw_dir` parameter can point to either:
- `data_citationlake/raw/` - Raw data from CitationLake (symlinked). **Uses CitationLake's `load_combined_data` with `card` field** (recommended)
- `data/raw/` - Local raw data directory (uses local utils)

```bash
# Using CitationLake raw data (recommended - uses CitationLake's load_combined_data with card field)
python -m src.search.card2card build-index \
  --field card \
  --raw_dir data_citationlake/raw \
  --output_jsonl data/card2card_corpus.jsonl \
  --output_npz data/card2card_embeddings.npz \
  --output_index data/card2card.faiss \
  --device cuda

# Or using processed parquet (faster, if available)
# python -m src.search.card2card build-index \
#   --field card_readme \
#   --parquet data_citationlake/processed/modelcard_step1.parquet \
#   --output_jsonl data/card2card_corpus.jsonl \
#   --output_npz data/card2card_embeddings.npz \
#   --output_index data/card2card.faiss \
#   --device cuda
```

**Output files:**
- `data/card2card_corpus.jsonl` - Corpus in JSONL format
- `data/card2card_embeddings.npz` - Embeddings
- `data/card2card.faiss` - FAISS index

### 2. Query to ModelCard Search

Search model cards using a text query (most common use case):

```bash
python -m src.search.query2modelcard \
  --query "transformer model for code generation" \
  --emb_npz data/card2card_embeddings.npz \
  --faiss_index data/card2card.faiss \
  --top_k 20 \
  --device cuda \
  --output_json data/query2modelcard_results.json
```

### 3. Card to Card Search

Search for similar model cards given a model ID:

```bash
# Search for similar model cards (single query)
python -m src.search.card2card search \
  --model_id Salesforce/codet5-base \
  --emb_npz data/card2card_embeddings.npz \
  --faiss_index data/card2card.faiss \
  --top_k 20 \
  --output_json data/card2card_results.json

# Search for all models (batch)
# python -m src.search.card2card search-batch \
#   --emb_npz data/card2card_embeddings.npz \
#   --faiss_index data/card2card.faiss \
#   --top_k 20 \
#   --output_json data/card2card_neighbors.json
```

### 4. Table to Table Search (Testing Tool)

**⚠️ Important: This requires Blend_internal to be cloned first!**

**Prerequisites:**
1. **Blend_internal must be cloned** (see Installation step 2 above):
   ```bash
   # For private repository (SSH)
   git clone git@github.com:DoraDong-2023/Blend_internal.git src/Blend_internal
   
   # Or for public repository (HTTPS)
   git clone https://github.com/DoraDong-2023/Blend_internal.git src/Blend_internal
   ```
2. **`data/modellake.db` must be created** (see below)

**What is `modellake.db`?**
- A DuckDB database containing an indexed table (`modellake_index`) created from all CSV files
- The index table contains tokenized values from all CSV cells for fast table-to-table search
- Table structure: `tokenized`, `tableid`, `colid`, `rowid`, `filename`, `table_group`, `table_type`
- Created from CSV files using `create_index_duckdb.py` script

```bash
# change src/Blend_internal/config/config.ini to modellake path
python -m src.Blend_internal.scripts.check_db --db_path data_citationlake/modellake.db
```

**To create `modellake.db` (if not exists):**
```bash
python -m src.Blend_internal.scripts.create_index_duckdb \
  --db_path data/modellake.db \
  --data_glob "data_citationlake/processed/deduped_hugging_csvs/*.csv,data_citationlake/processed/deduped_github_csvs/*.csv,data_citationlake/processed/tables_output/*.csv" \
  --table modellake_index
  --mask ../CitationLake/data/analysis/all_valid_title_valid.txt  # Optional: list of CSV files to include
```

Test table-to-table search using Blend_internal:

```bash
# List all tables in modellake.db
python -m src.search.tab2tab --list_tables --db_path modellake.db

# Single column search
python -m src.search.tab2tab \
  --search_type single_column \
  --query "train, dataset, model" \
  --k 10 \
  --db_path modellake.db \
  --output data/tab2tab_join_sing_results.json

# Multi column search (haven't tested yet)
# python -m src.search.tab2tab \
#   --search_type multi_column \
#   --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv \
#   --k 10 \
#   --db_path data_citationlake/modellake.db \
#   --output data/tab2tab_join_multi_results.json

# Keyword search
python -m src.search.tab2tab \
  --search_type keyword \
  --query "train, dataset, model" \
  --k 10 \
  --db_path modellake.db \
  --output data/tab2tab_keyword_results.json

# Unionable search (find tables with unionable columns)
python -m src.search.tab2tab \
  --search_type unionable \
  --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv \
  --k 10 \
  --db_path modellake.db \
  --output data/tab2tab_union_results.json
```

### 5. Card to Tab to Card Search (Main Script)

Two-stage search: find tables for a model card, then find similar model cards via table search:

```bash
# Using CitationLake get_from (recommended) - Keyword search
python -m src.search.card2tab2card \
  --model_id Salesforce/codet5-base \
  --schema_log data_citationlake/logs/parquet_schema.log \
  --query "train,model,dataset" \
  --search_type keyword \
  --db_path data_citationlake/modellake.db \
  --k 10 \
  --output_json data/card2tab2card_keyword_results.json

# Single column search
python -m src.search.card2tab2card \
  --model_id Salesforce/codet5-base \
  --schema_log data_citationlake/logs/parquet_schema.log \
  --query "value1,value2,value3" \
  --search_type single_column \
  --db_path data_citationlake/modellake.db \
  --k 10 \
  --output_json data/card2tab2card_singlecol_results.json

# Multi column search
python -m src.search.card2tab2card \
  --model_id Salesforce/codet5-base \
  --schema_log data_citationlake/logs/parquet_schema.log \
  --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv \
  --search_type multi_column \
  --db_path data_citationlake/modellake.db \
  --k 10 \
  --output_json data/card2tab2card_multicol_results.json

# Without query - uses model's tables automatically
python -m src.search.card2tab2card \
  --model_id Salesforce/codet5-base \
  --schema_log data_citationlake/logs/parquet_schema.log \
  --search_type keyword \
  --db_path data_citationlake/modellake.db \
  --k 10 \
  --output_json data/card2tab2card_keyword_results.json

# Using pre-computed table search results
python -m src.search.card2tab2card \
  --model_id Salesforce/codet5-base \
  --schema_log data_citationlake/logs/parquet_schema.log \
  --table_search_json data/table_search.json \
  --k 10 \
  --output_json data/card2tab2card_keyword_results.json

# Fallback: using relationship_parquet (if CitationLake not available)
python -m src.search.card2tab2card \
  --model_id Salesforce/codet5-base \
  --relationship_parquet data_citationlake/processed/modelcard_step3_dedup.parquet \
  --no_citationlake \
  --query "train,model,dataset" \
  --search_type keyword \
  --db_path data_citationlake/modellake.db \
  --k 10 \
  --output_json data/card2tab2card_keyword_results.json
```

**Output:** `data/card2tab2card_results.json`

### 6. Interactive Demo

#### Frontend (CLI Interactive)

```bash
python -m src.demo.frontend
```

Interactive menu for testing all search functions. All results saved to `data/` directory.

#### Backend (REST API)

```bash
python -m src.demo.backend
```

Starts a Flask REST API server on `http://localhost:5000` with endpoints:

- `POST /api/build-index` - Build FAISS index
- `POST /api/query2modelcard` - Search with text query
- `POST /api/card2card` - Search similar model cards
- `POST /api/tab2tab` - Table to table search (testing)
- `POST /api/card2tab2card` - Card to tab to card search
- `GET /api/health` - Health check

**Example API call:**
```bash
curl -X POST http://localhost:5000/api/query2modelcard \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer model", "top_k": 20}'
```

All API results are saved to `data/` directory.

## Using Search Functions in Python

All search functions can be imported and used programmatically:

```python
from src.search import (
    build_card_index,
    search_card2card,
    search_table2table,
    search_query2modelcard,
    search_card2tab2card
)

# 1. Build index (first time only)
# Option 1: Using CitationLake raw data
build_card_index(
    field="card",
    raw_dir="data_citationlake/raw",  # or "data/raw" for local data
    output_index="data/card2card.faiss"
)

# Option 2: Using processed parquet (faster)
build_card_index(
    field="card_readme",
    parquet="data_citationlake/processed/modelcard_step1.parquet",
    output_index="data/card2card.faiss"
)

# 2. Query to modelcard (most common)
results = search_query2modelcard(
    query="transformer model for NLP",
    faiss_index="data/card2card.faiss",
    top_k=20
)

# 3. Card to card
neighbors = search_card2card(
    model_id="Salesforce/codet5-base",
    faiss_index="data/card2card.faiss",
    top_k=20
)

# 4. Table to table (testing)
table_ids = search_table2table(
    query=["value1", "value2"],
    search_type="single_column",
    k=10
)

# 5. Card to tab to card
similar_models = search_card2tab2card(
    model_id="Salesforce/codet5-base",
    schema_log_path="data_citationlake/logs/parquet_schema.log",
    query=["keyword1", "keyword2"],
    search_type="keyword",
    k=10
)
```

## Batch Evaluation (Optional)

For batch evaluation, you can use the batch search functions:

```python
from src.search import search_card2card_batch

# Search all models at once
all_neighbors = search_card2card_batch(
    emb_npz="data/card2card_embeddings.npz",
    faiss_index="data/card2card.faiss",
    top_k=20,
    output_json="data/card2card_neighbors.json"
)
```

## Legacy Commands

#### Build Index (Legacy)

```bash
bash src/modelsearch/base_densesearch.sh
```

#### Compare Baselines

```bash
python -m src.modelsearch.compare_baselines \
  --model_id Salesforce/codet5-base \
  --relationship_parquet data/processed/modelcard_step3_dedup.parquet \
  --starmie_json results/table_search.json \
  --dense_neighbors output/modelsearch_neighbors.json \
  --output_md output/compare.md
```
