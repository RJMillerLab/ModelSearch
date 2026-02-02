# Build Index

Two parts: **Part 1** = index building and preparation (run once; build from scratch, no reuse from ModelTables). **Part 2** = inference and downstream (retrieval search, table search, evaluation, QA, integration). After Part 1 is done, users only need Part 2.

---

# Part 1 — Preparation & index building

## 1.1 Modelcard index (Step 1: query→modelcard retrieval)

Build corpus JSONL + embeddings NPZ + FAISS index so you can do **query→modelcard** (and later **model→similar models** via card2card). Build from scratch: use either baseline1 (3 steps) or card2card build-index (1 step).

**Output files:** `data/card2card_corpus.jsonl`, `data/card2card_embeddings.npz`, `data/card2card.faiss`

### Option A: card2card one-step build (recommended)

```bash
python -m src.search.card2card build-index --field card --raw_dir data_citationlake/raw --output_npz data/card2card_embeddings.npz --output_index data/card2card.faiss
```

Optional: `--parquet <path>`, `--output_jsonl data/card2card_corpus.jsonl`, `--device cuda`.

### Option B: baseline1 three steps

```bash
python -m src.baseline1.build_modelcard_jsonl --field card --output_jsonl data/card2card_corpus.jsonl
python -m src.baseline1.table_retrieval_pipeline encode --jsonl data/card2card_corpus.jsonl --model_name all-MiniLM-L6-v2 --batch_size 256 --output_npz data/card2card_embeddings.npz --device cuda
python -m src.baseline1.table_retrieval_pipeline build_faiss --emb_npz data/card2card_embeddings.npz --output_index data/card2card.faiss
```

### Optional: filter by mask (then encode + build_faiss)

```bash
python -m src.baseline1.table_retrieval_pipeline filter --base_path data_citationlake/processed --mask_file data_citationlake/analysis/all_valid_title_valid_251117.txt --output_jsonl data/card2card_corpus.jsonl --model_name all-MiniLM-L6-v2 --device cuda
```

Then run `encode` and `build_faiss` as in Option B.

---

## 1.2 Blend setup (for table search: tab2tab / card2tab2card)

Table search (card2tab2card, tab2tab) uses Blend_internal. Clone or symlink Blend_internal; on server you may symlink data.

```bash
# On server: symlink data (if data lives under ModelTables)
ln -s /u1/z6dong/Repo/ModelTables/data data_citationlake
# Clone Blend_internal
git clone git@github.com:DoraDong-2023/Blend_internal.git src/Blend_internal
# Or symlink existing repo
ln -s /u1/z6dong/Repo/Blend_internal src/Blend_internal
```

---

## 1.3 modellake.db (DuckDB table index for tab2tab / card2tab2card)

Table-level index used by Blend_internal tab2tab and card2tab2card. Output: `data/modellake.db` or `data/modellake.db`, table `modellake_index`.

```bash
python -m src.Blend_internal.scripts.create_index_duckdb --db_path data/modellake.db --data_glob "data_citationlake/processed/deduped_github_csvs/*.csv" --data_glob "data_citationlake/processed/deduped_hugging_csvs/*.csv" --data_glob "data_citationlake/processed/tables_output/*.csv" --table modellake_index
# --mask data_citationlake/analysis/all_valid_title_valid_251117.txt
```

---

## 1.4 Table classification (optional; for card2tab2card by_type and tab2tab_by_type)

Pre-compute table classifications so you can filter by type. **Use the same `--db_path` you passed to create_index_duckdb** (e.g. `data/modellake.db` or `data_citationlake/modellake.db`). If tab2know fails on the server (e.g. missing `others/tab2know/models/` or deps), use `--method heuristic`.

```bash
python -m src.search.classification --mode batch --db_path data/modellake.db --output_json data/table_classifications.json
# If tab2know fails (missing models/deps on server):
python -m src.search.classification --mode batch --db_path data/modellake.db --output_json data/table_classifications.json --method heuristic
```

---

# Part 2 — Inference & downstream

After Part 1, use these for retrieval search, table search, evaluation, QA, and integration. **Each script prints total runtime (e.g. `⏱️ Total time: 12.34s`) at the end**; when you redirect to logs (e.g. `> logs/xxx.log 2>&1`), the log will contain this time.

## 2.1 query2modelcard (text query → top-k model IDs)

Dense retrieval only (FAISS). Uses Step-1 index.

```bash
python -m src.search.query2modelcard --query "transformer model for code generation" --emb_npz data/card2card_embeddings.npz --faiss_index data/card2card.faiss --top_k 20 --device cuda > logs/query2modelcard.log 2>&1
```

Optional: `--output_json <path>`.

---

## 2.2 card2card (model_id → top-k similar model IDs)

Semantic retrieval: dense (FAISS), sparse (BM25), or hybrid (RRF). Uses Step-1 index (+ `data/card2card_corpus.jsonl` for sparse/hybrid).

```bash
# dense
python -m src.search.card2card search --model_id senseable/33x-coder --emb_npz data/card2card_embeddings.npz --faiss_index data/card2card.faiss --top_k 20 --retrieval_mode dense > logs/card2card_dense.log 2>&1
# sparse
python -m src.search.card2card search --model_id senseable/33x-coder --jsonl_path data/card2card_corpus.jsonl --top_k 20 --retrieval_mode sparse > logs/card2card_sparse.log 2>&1
# hybrid
python -m src.search.card2card search --model_id senseable/33x-coder --emb_npz data/card2card_embeddings.npz --faiss_index data/card2card.faiss --jsonl_path data/card2card_corpus.jsonl --top_k 20 --retrieval_mode hybrid --hybrid_method rrf > logs/card2card_hybrid.log 2>&1
```

Optional: `--output_json <path>`. Batch: `python -m src.search.card2card search-batch --emb_npz ... --faiss_index ... --top_k 20 --output_json data/card2card_neighbors.json > logs/card2card_batch.log 2>&1`.

---

## 2.3 card2tab2card (model → tables → table search → model IDs)

Structure/table retrieval: get model's tables, run table-to-table search (Blend), return model IDs linked to similar tables. Uses modellake.db and relationship parquet (not card2card FAISS). **Keyword search uses table headers (column names) as keywords**, same as ModelTables/Blend.

**Single search type (e.g. keyword):**
```bash
python -m src.search.card2tab2card --model_id senseable/33x-coder --search_type keyword --k 10 > logs/card2tab2card_keyword.log 2>&1
```

Other `--search_type`: `single_column`, `multi_column`, `unionable`. Optional: `--query <csv_path>` for multi_column/unionable; `--output_json data/card2tab2card_results.json`, `--db_path data/modellake.db`.

**All search types (single_column, keyword, unionable):**
```bash
python -m src.search.card2tab2card --model_id senseable/33x-coder --mode all --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv --output_folder data > logs/card2tab2card_all.log 2>&1
```

**By table type (requires classification JSON from 1.5):**
```bash
python -m src.search.card2tab2card --model_id senseable/33x-coder--mode by_type --classification_json data/table_classifications.json > logs/card2tab2card_by_type.log 2>&1
```

---

## 2.4 tab2tab (standalone table search; Blend_internal)

Direct table-to-table search using modellake.db. Requires Blend_internal and modellake.db. **Keyword search uses table headers (column names) as keywords**, same as ModelTables/Blend: pass comma-separated header names to match tables that have those columns.

```bash
# keyword: comma-separated column names (headers) to match
python -m src.search.tab2tab --search_type keyword --query "model_name,accuracy,task" --k 10 --db_path data/modellake.db --output data/tab2tab_results.json > logs/tab2tab_keyword.log 2>&1
```

For single_column: `--query "val1,val2,val3"` (cell values). For multi_column/unionable: `--query <path_to_csv>`. `--list_tables` to list tables in DB.

---

## 2.5 tab2tab_by_type (table search filtered by type)

Table search with type filtering. Run classification first (1.5) or use `--no_auto_classify` and provide `--classification_json`.

```bash
python -m src.search.tab2tab_by_type --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv --classification_json data/table_classifications.json --search_type single_column --k 10 --db_path data/modellake.db --output data/tab2tab_by_type_results.json > logs/tab2tab_by_type.log 2>&1
```

---

## 2.6 Evaluation, QA, table integration

Evaluation (LLM-based comparison), question answering, and table integration are available via the **demo** (backend + frontend). Start backend and frontend, then use the UI.

```bash
python -m src.demo.backend
python -m src.demo.frontend
```

Then open http://localhost:5001. The demo wires query2modelcard, card2card, card2tab2card, integration, evaluation, and QA. For scripted runs, the logic lives in `src/evaluation/`, `src/qa/`, and `src/integration/`; entry points can be invoked from the backend or from tests.

---

## 2.7 baseline2 / baseline3 (table retrieval: sparse, hybrid; copied from ModelTables)

**Note:** card2card (2.2) already provides sparse/hybrid for **model-card** retrieval. Baseline2/baseline3 here are **table retrieval** baselines (BM25 sparse, hybrid sparse+dense) for the Starmie-style evaluation pipeline. They are copied from ModelTables; full workflow is in **ModelTables/docs/scripts.md** (Section 7).

### Copy baseline2 and baseline3 from ModelTables

Run from **ModelSearchDemo** repo root. Use the path to your ModelTables repo (e.g. sibling `../ModelTables` or absolute path).

```bash
# From ModelSearchDemo repo root; ModelTables as sibling repo
cp -r ../ModelTables/src/baseline2 src/
cp -r ../ModelTables/src/baseline3 src/
```

If ModelTables is elsewhere, replace `../ModelTables` with the correct path (e.g. `/path/to/ModelTables`). This repo already provides `to_parquet` in `src/utils` for baseline2 compatibility.

**Data requirement:** Baseline2/3 expect ModelTables-style processed data under `data/` (e.g. `data/processed/` with parquet and `deduped_github_csvs*`, `deduped_hugging_csvs*`, etc.). If your data lives under `data_citationlake/`, either symlink `ln -s /path/to/ModelTables/data data` when layout matches, or set `TAG` and ensure `data/processed/`, `data/analysis/` exist. **TAG** env (e.g. `TAG=251117`) is used for versioned paths.

**Dependencies:** pyserini and Java/JDK for baseline2; baseline3 also needs dense index from baseline1.

**Expected output:** get_metadata produces `data/tmp/corpus/collection.jsonl` (e.g. ~41k docs). sparse_search produces `data/tmp/baseline2_sparse_results_<TAG>.json` (e.g. ~33k queries). Many "Could not find table file for XXXXX_tableN" warnings are normal when some arXiv CSVs are missing (path mismatch, e.g. `tables_output` vs `tables_output_v2_251117`); those queries are skipped. **Huggingface mapping:** If `data/processed/hugging_deduped_mapping.json` (or `hugging_deduped_mapping_v2_251117.json`) is missing, the script skips Huggingface and uses GitHub + arXiv only; corpus and sparse search still run.

### Baseline2: Sparse search (BM25)

Commands below match **ModelTables/docs/scripts.md** Section 7 (Baseline2). Two entry points: `get_metadata.sh` then `sparse_search.sh` (the latter runs: build Lucene index → create queries → BM25 search → postprocess).

```bash
ln -s /u1/z6dong/Repo/ModelTables/data/tmp/ data/
ln -s /u1/z6dong/Repo/ModelTables/data/analysis/ data/
ln -s /u1/z6dong/Repo/ModelTables/data/processed/ data/
ln -s /u1/z6dong/Repo/ModelTables/data/deduped/ data/
ln -s /u1/z6dong/Repo/ModelTables/arxiv_fulltext_html ./
ln -s /u1/z6dong/Repo/ModelTables/data/downloaded_github_readmes data/
ln -s /u1/z6dong/Repo/ModelTables/data/downloaded_github_readmes_251117 ./
ln -s /u1/z6dong/Repo/ModelTables/data/arxiv_fulltext_html_251117 ./

# 1) Get metadata: csv→readme mapping, then corpus data/tmp/corpus/collection.jsonl
TAG=251117 bash src/baseline2/get_metadata.sh > logs/baseline2_get_metadata_251117.log 2>&1
# 2) Build Pyserini index + create queries + search + postprocess (all-in-one)
TAG=251117 bash src/baseline2/sparse_search.sh > logs/baseline2_sparse_search_251117.log 2>&1
```

Output: `data/tmp/baseline2_sparse_results_251117.json` (and `data/tmp/search_result_251117.json`). For Starmie evaluation metrics, see ModelTables Section 6: `TAG=251117 bash scripts/step3_processmetrics_all.sh <index>` (run from ModelTables repo).

**Java (for baseline2):** Pyserini needs Java to build the Lucene index and run search. Set `JAVA_HOME` and have `javac` on PATH before running baseline2; e.g. `conda install -c conda-forge openjdk`.

### Baseline3: Hybrid (sparse + dense)

Option A — use Pyserini hybrid (single script):

```bash
# Prerequisite: sparse index from baseline2 (data/tmp/index_251117), and dense index:
mkdir -p data/tmp/index_dense_251117
python -m src.baseline1.table_retrieval_pipeline encode \
  --jsonl data/tmp/corpus/collection.jsonl --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --batch_size 256 --output_npz data/tmp/index_dense_251117/embeddings.npz --device cuda
python -m src.baseline1.table_retrieval_pipeline build_faiss \
  --emb_npz data/tmp/index_dense_251117/embeddings.npz --output_index data/tmp/index_dense_251117/index.faiss

TAG=251117 python src/baseline2/search_with_pyserini_hybrid.py \
  --sparse-index data/tmp/index_251117 --dense-index data/tmp/index_dense_251117 \
  --queries data/tmp/queries_table.tsv --mapping data/tmp/queries_table_mapping.json \
  --k 11 --alpha 0.45 --device cuda > logs/baseline3_hybrid_search_251117.log 2>&1
```

Option B — sparse-first then hybrid rerank (see `src/baseline3/hybrid_search.sh`):

```bash
TAG=251117 bash src/baseline3/hybrid_search.sh > logs/baseline3_hybrid_251117.log 2>&1
```

Postprocess hybrid results if needed (same as baseline2): `python src/baseline2/postprocess.py --input ... --output data/tmp/baseline3_hybrid_results_251117.json --top1-list ...`.

---

# Quick reference

| Goal | Command |
|------|---------|
| **Part 1** | |
| Modelcard index (1 step) | `python -m src.search.card2card build-index --field card --raw_dir data_citationlake/raw --output_npz data/card2card_embeddings.npz --output_index data/card2card.faiss` |
| Modelcard index (3 steps) | build_modelcard_jsonl → table_retrieval_pipeline encode → table_retrieval_pipeline build_faiss |
| Push to servers | `./scripts/push_index_to_servers.sh` |
| Blend setup | clone/symlink `src/Blend_internal`; optional `ln -s ... data_citationlake` |
| Copy baseline2/3 from ModelTables | `cp -r ../ModelTables/src/baseline2 src/` and `cp -r ../ModelTables/src/baseline3 src/` |
| DuckDB table index | `python -m src.Blend_internal.scripts.create_index_duckdb --db_path ... --data_glob ... --table modellake_index` |
| Table classification | `python -m src.search.classification --mode batch --db_path data/modellake.db --output_json data/table_classifications.json` |
| **Part 2** | |
| query2modelcard | `python -m src.search.query2modelcard --query "..." --emb_npz data/card2card_embeddings.npz --faiss_index data/card2card.faiss --top_k 20 --device cuda` |
| card2card dense | `python -m src.search.card2card search --model_id <id> --retrieval_mode dense --top_k 20` |
| card2card sparse | `... --retrieval_mode sparse --jsonl_path data/card2card_corpus.jsonl` |
| card2card hybrid | `... --retrieval_mode hybrid --jsonl_path data/card2card_corpus.jsonl --hybrid_method rrf` |
| card2tab2card | `python -m src.search.card2tab2card --model_id <id> --search_type keyword --k 10` |
| tab2tab (keyword = headers) | `python -m src.search.tab2tab --search_type keyword --query "model_name,accuracy,task" --k 10 --db_path data/modellake.db` |
| tab2tab_by_type | `python -m src.search.tab2tab_by_type --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv --classification_json data/table_classifications.json --search_type single_column --k 10` |
| baseline2 (sparse table retrieval) | `TAG=251117 bash src/baseline2/get_metadata.sh` then `bash src/baseline2/sparse_search.sh` |
| baseline3 (hybrid table retrieval) | Build dense index (baseline1 encode + build_faiss), then `python src/baseline2/search_with_pyserini_hybrid.py ...` or `bash src/baseline3/hybrid_search.sh` |
| Demo (evaluation, QA, integration) | `python -m src.demo.backend` + `python -m src.demo.frontend` → http://localhost:5001 |

**Reference:** Full data pipeline and baseline commands (including baseline1 unified, Starmie steps): **ModelTables/docs/scripts.md**.
