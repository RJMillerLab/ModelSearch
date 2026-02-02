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
python -m src.search.card2tab2card --model_id senseable/33x-coder --search_type keyword --k 10
```

Other `--search_type`: `single_column`, `multi_column`, `unionable`. Optional: `--query <csv_path>` for multi_column/unionable; `--output_json data/card2tab2card_results.json`, `--db_path data/modellake.db`.

**All search types (single_column, keyword, unionable):**
```bash
python -m src.search.card2tab2card --model_id senseable/33x-coder --mode all --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv --output_folder data
```

**By table type (requires classification JSON from 1.5):**
```bash
python -m src.search.card2tab2card --model_id <model_id> --mode by_type --classification_json data/table_classifications.json
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

## 2.7 baseline2 / baseline3 (sparse, hybrid; from ModelTables)

Additional retrieval strategies (e.g. baseline2, baseline3 from ModelTables/src — sparse/hybrid variants) can be copied and tested later; after testing, rename and organize under this repo. For now, card2card search already provides sparse and hybrid (see 2.2).

---

# Quick reference

| Goal | Command |
|------|---------|
| **Part 1** | |
| Modelcard index (1 step) | `python -m src.search.card2card build-index --field card --raw_dir data_citationlake/raw --output_npz data/card2card_embeddings.npz --output_index data/card2card.faiss` |
| Modelcard index (3 steps) | build_modelcard_jsonl → table_retrieval_pipeline encode → table_retrieval_pipeline build_faiss |
| Push to servers | `./scripts/push_index_to_servers.sh` |
| Blend setup | clone/symlink `src/Blend_internal`; optional `ln -s ... data_citationlake` |
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
| Demo (evaluation, QA, integration) | `python -m src.demo.backend` + `python -m src.demo.frontend` → http://localhost:5001 |
