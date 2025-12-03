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

Two-stage search: find tables for a model card, then find similar model cards via table search.

**Pipeline:** Query -> ModelCard -> Tables -> Retrieved Tables -> Corresponding ModelCards

**Default:** Uses `data_citationlake/processed/modelcard_step3_dedup.parquet` for relationship mapping.

```bash
# Mode: ALL - Run all three search types automatically
# Note: --query must be a CSV file path when --mode=all
# Outputs:
#   - {output_folder}/card2tab2card_singlecol_results.json
#   - {output_folder}/card2tab2card_keyword_results.json
#   - {output_folder}/card2tab2card_unionable_results.json
python -m src.search.card2tab2card \
  --model_id Salesforce/codet5-base \
  --query data_citationlake/processed/deduped_github_csvs/0021c79d4e1a37579ca87328864d67a5_table_0.csv \
  --mode all \
  --output_folder data \
  --db_path data_citationlake/modellake.db \
  --k 10
```

<details>
<summary>Other modes (single search type)</summary>

```bash
# Single search type examples
python -m src.search.card2tab2card \
  --model_id Salesforce/codet5-base \
  --query "train,model,dataset" \
  --search_type keyword \
  --mode single \
  --db_path data_citationlake/modellake.db \
  --k 10 \
  --output_json data/card2tab2card_keyword_results.json
```
</details>

### 5.1. Table Classification (New)

Classify tables based on their content and structure using tab2know. This is used to filter search results to only include tables of the same type.

**Classification Labels (from tab2know):**
- `Observation`: Performance/result tables (most common in academic papers)
- `Input`: Configuration/input tables
- `Other`: Other types of tables
- `Example`: Example tables

**Prerequisites:**
- Tab2Know repository should be available (set `TAB2KNOW_REPO` environment variable or ensure `TabKnow_internal` is in a standard location)

**Step 1: Classify all tables in datalake (one-time setup)**

```bash
# Batch classify all tables in modellake.db using tab2know (default)
python -m src.search.classification \
  --mode batch \
  --db_path data_citationlake/modellake.db \
  --output_json data/table_classifications.json \
  --method tab2know

# Or with a limit (for testing)
python -m src.search.classification \
  --mode batch \
  --db_path data_citationlake/modellake.db \
  --output_json data/table_classifications.json \
  --limit 100 \
  --method tab2know

# Use heuristic method (faster but less accurate)
python -m src.search.classification \
  --mode batch \
  --db_path data_citationlake/modellake.db \
  --output_json data/table_classifications.json \
  --method heuristic
```

**Step 2: Classify a single table**

```bash
# Classify from CSV file using tab2know (default)
python -m src.search.classification \
  --mode single \
  --table data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv \
  --method tab2know

# Classify from database table ID
python -m src.search.classification \
  --mode single \
  --tableid 12345 \
  --db_path data_citationlake/modellake.db \
  --method tab2know
```

### 5.2. Table to Table Search by Type (New)

Search for similar tables, filtering results to only include tables with the same classification as the query table.

**Prerequisites:** Run table classification first (see 5.1 above).

```bash
# Single column search with classification filtering
python -m src.search.tab2tab_by_type \
  --query "train,dataset,model" \
  --search_type single_column \
  --classification_json data/table_classifications.json \
  --k 10 \
  --db_path data_citationlake/modellake.db \
  --output data/tab2tab_by_type_results.json

# Keyword search with classification filtering
python -m src.search.tab2tab_by_type \
  --query "train,dataset,model" \
  --search_type keyword \
  --classification_json data/table_classifications.json \
  --k 10 \
  --db_path data_citationlake/modellake.db

# Multi-column search from CSV file
python -m src.search.tab2tab_by_type \
  --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv \
  --search_type multi_column \
  --classification_json data/table_classifications.json \
  --k 10 \
  --db_path data_citationlake/modellake.db
```

### 5.3. Card to Tab to Card Search by Type (New)

Search for model cards via table search with classification filtering. This ensures that only tables of the same type are considered in the search.

**Prerequisites:** Run table classification first (see 5.1 above).

```bash
# Search with classification filtering
python -m src.search.card2tab2card \
  --model_id Salesforce/codet5-base \
  --mode by_type \
  --classification_json data/table_classifications.json \
  --search_type keyword \
  --query "train,model,dataset" \
  --db_path data_citationlake/modellake.db \
  --k 10 \
  --output_json data/card2tab2card_by_type_results.json

# Using CSV file as query
python -m src.search.card2tab2card \
  --model_id Salesforce/codet5-base \
  --mode by_type \
  --classification_json data/table_classifications.json \
  --search_type single_column \
  --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv \
  --db_path data_citationlake/modellake.db \
  --k 10
```

### 6. Interactive Demo

Compare two search pipelines (Card2Card vs Card2Tab2Card) with a web interface:

```bash
# Start backend server (port 5000)
python -m src.demo.backend

# Start frontend server (port 5001) in another terminal
python -m src.demo.frontend

# Open http://localhost:5001 in your browser
```

**Pipeline:**
1. Input: Text query (e.g., "transformer model for code generation")
2. Extract model card from query
3. Run two parallel pipelines:
   - **Card2Card**: Dense semantic search
   - **Card2Tab2Card**: Table-based search (all three types)
4. Display progress logs in real-time
5. Visualize results side-by-side

See [docs/demo.md](docs/demo.md) for detailed documentation.

## Documentation

- [Python API Usage](docs/api.md) - Programmatic usage of search functions
- [Interactive Demo](docs/demo.md) - Web interface documentation
- [Legacy Commands](docs/legacy.md) - Legacy scripts and commands
