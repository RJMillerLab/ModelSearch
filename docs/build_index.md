# Build Index

**Paths:** Data paths (ModelTables/data, modellake.db, processed dirs, relationship parquet) are defined in `src.config`. Override with env: `MODELTABLES_DATA` or `DATA_ROOT`, `MODELLAKE_DB`, `DATA_TAG`. Examples below use config defaults.

**ModelCard IR (dense NPZ + sparse Lucene):** Use **`src.search.ir_index_builder`** to **build** indexes (`DenseCardIndexBuilder`, `SparseCardIndexBuilder`) and **`src.search.ir_searcher`** to **retrieve** (`DenseSearcher`, `SparseSearcher`). Outputs and defaults are `EMB_NPZ`, `SPARSE_INDEX`, and hugging subsets `EMB_NPZ_HUGGING`, `SPARSE_INDEX_HUGGING` in `src.config`.

**Part 1** = index building (run once). **Part 2** = inference (retrieval, table search, demo). When a step has multiple modes, run all and save to separate log/json.
# Part 1 — Must run (in order)


## 1.0 Blend + data

```bash
# Build DuckDB index from csvs, for later search
git clone git@github.com:DoraDong-2023/Blend_internal.git 
ln -s ../Blend_internal others/Blend_internal
# create DuckDB index from csvs, by following the instructions from Blend README. Output: modellake_v2_251117_nomask.db

# optional: create subset of duckdb index for later search
# under blend_internal/scripts/create_index_duckdb.py
# python -m scripts.create_index_duckdb --db_path database/modellake_v2_251117_nomask_hugging.db --data_glob "../ModelTables/data/processed/deduped_hugging_csvs_v2_251117/*.csv,../ModelTables/data/processed/deduped_hugging_csvs_v2_251117_tr/*.csv,../ModelTables/data/processed/deduped_hugging_csvs_v2_251117_str/*.csv" --workers 8 --insert-batch 4 --skip-large-tables
# here we don't use the mask file, as we don't need <MUST CONTAIN TITLE> condition
# but we skip table with size > 200 rows or 100 columns
```

## 1.1 Valid model IDs for Table Search

Extract model_id that have tables (non-empty `csv_basename` in relationship parquet) into a txt so inference only loads it. Run once after parquet is available.

```bash
# explode parquet, get model-csv pair relationship from modelcard_step3_dedup_v2_251117.parquet (data from ModelTables)
python scripts/build_model_to_tables_explode_parquet.py --output_parquet data_251117/model_to_tables_explode_v2_251117.parquet --relationship_parquet ../ModelTables/data/processed/modelcard_step3_dedup_v2_251117.parquet

# valid model ids which table exist in DuckDB index
python -m src.utils.build_valid_model_ids_txt --output data_251117/valid_model_ids_with_tables_hugging.txt --resources hugging --duckdb_path ../Blend_internal/database_251117/modellake_v2_251117_nomask_hugging.db --explode_parquet data_251117/model_to_tables_explode_v2_251117.parquet
# update explode parquet in-place by DuckDB table list (filter hugging rows only)
python -m scripts.update_model_to_tables_explode_parquet_by_db_tables --parquet_path data_251117/model_to_tables_explode_v2_251117.parquet --resources hugging --duckdb_path ../Blend_internal/database_251117/modellake_v2_251117_nomask_hugging.db > logs/update_model_to_tables_explode_parquet_by_db_tables_hugging.log 2>&1 # so it won't affect downstream getting csvs, otherwise it will get csv that don't exist in DuckDB index
```

## 1.2 Modelcard index (`ir_index_builder`)

From `modelcard_step1` parquet (see `MODELCARD_STEP1_PARQUET`): dense = SentenceTransformer → `EMB_NPZ`; sparse = corpus JSONL + Pyserini Lucene → `SPARSE_INDEX`.

```python
# Run from repo root (PYTHONPATH=. or `pip install -e .`)
from src.search.ir_index_builder import DenseCardIndexBuilder, SparseCardIndexBuilder

DenseCardIndexBuilder().build(batch_size=256)
SparseCardIndexBuilder().build(threads=4)
```

```bash
# Same as above, via module CLI
python -m src.search.ir_index_builder build-dense-index --batch_size 256
python -m src.search.ir_index_builder build-sparse-index --threads 4
```

```bash
# (Optional) Build dense + sparse subset for "hugging" only (filters NPZ + Lucene by id list)
python -m src.utils.build_subset_from_embeddings_and_ids --model_ids_txt data_251117/valid_model_ids_with_tables_hugging.txt --threads 4
# Test doc count on hugging Lucene index
# python -c "from pyserini.search.lucene import LuceneSearcher; s=LuceneSearcher('data_251117/card2card_sparse_index_hugging'); print('index_docs=', s.num_docs)"
```

<details><summary>(Optional) 1.3 Table classification (optional; for by_type flows)</summary>

Train vs inference = explicit arg only, no fallback. Train = run below with `--mode batch`. Inference = run 2.3 by_type / 2.5 with `--classification_json data/table_classifications.json`. Same `--db_path` for 1.4 and 2.3/2.5.

```bash
# train (full datalake)
python -m src.search.classification --mode batch --output_json data/table_classifications.json
# heuristic if tab2know fails
python -m src.search.classification --mode batch --output_json data/table_classifications.json --method heuristic
```

</details>

---

# Part 2 — Inference

## 2.1 ModelCard retrieval (`ir_searcher`)

**Text query:** dense = encode query with the same encoder as index build, then FAISS inner product; sparse = BM25 on the Lucene index. Use `EMB_NPZ` / `SPARSE_INDEX` for full corpus, or `EMB_NPZ_HUGGING` / `SPARSE_INDEX_HUGGING` for the hugging subset.

```python
from src.config import EMB_NPZ_HUGGING, SPARSE_INDEX_HUGGING
from src.search.ir_searcher import DenseSearcher, SparseSearcher

q = "transformer model for code generation"
top_k = 20

dense = DenseSearcher(EMB_NPZ_HUGGING)
model_ids, scores = dense.search(q, top_k=top_k)

sparse = SparseSearcher(SPARSE_INDEX_HUGGING)
model_ids, scores = sparse.search(q, top_k=top_k)
```

**Neighbor search by seed `model_id`:** dense uses the **stored embedding row** as the query vector; sparse uses that doc’s **indexed text** as the BM25 query (not an embedding).

```python
seed = "google-bert/bert-base-uncased"
n_dense, s_dense = dense.search_by_model_id(seed, top_k=20)
n_sparse, s_sparse = sparse.search_by_model_id(seed, top_k=20)
```

**Hybrid** (sparse candidates → dense rerank) is not a single call on `DenseSearcher`; use `Query2ModelCardSearch` with hybrid mode or `src.search.card2card_batch.search_hybrid_neighbors_queries` if you need the legacy pipeline.

Alternative: `query2modelcard` CLI

```bash
python -m src.search.query2modelcard --query "transformer model for code generation" --top_k 20 --retrieval_mode dense --resources hugging --output_json tmp/query2modelcard_dense_hugging.json
python -m src.search.query2modelcard --query "transformer model for code generation" --top_k 20 --retrieval_mode sparse --resources hugging --output_json tmp/query2modelcard_sparse_hugging.json
python -m src.search.query2modelcard --query "transformer model for code generation" --top_k 20 --retrieval_mode hybrid --resources hugging --output_json tmp/query2modelcard_hybrid_hugging.json
```

## 2.3 tab2tab (test all modes: keyword, single_column, multi_column, unionable) (aug mode: ori, tr, str)

```bash
# unified input as table query
python -m src.search.tab2tab --resources hugging --search_type keyword --query "../ModelTables/data/processed/deduped_hugging_csvs_v2_251117/00007f0e43_table1.csv" --k 10 --output_json tmp/tab2tab_keyword.json --augmentation_types ori
python -m src.search.tab2tab --resources hugging --search_type single_column --query "../ModelTables/data/processed/deduped_hugging_csvs_v2_251117/00007f0e43_table1.csv" --k 10 --output_json tmp/tab2tab_single_column.json --augmentation_types ori
python -m src.search.tab2tab --resources hugging --search_type multi_column --query "../ModelTables/data/processed/deduped_hugging_csvs_v2_251117/00007f0e43_table1.csv" --k 10 --output_json tmp/tab2tab_multi_column.json --augmentation_types ori
python -m src.search.tab2tab --resources hugging --search_type unionable --query "../ModelTables/data/processed/deduped_hugging_csvs_v2_251117/00007f0e43_table1.csv" --k 10 --output_json tmp/tab2tab_unionable.json --augmentation_types ori
```

## 2.3b Augmented tab2tab reranker based on scores from ori/tr/str -> ori/tr/str search (`tab2tab_aug`)

mode for reranker: table_level, score_max, flat_noscore
mode for query augmentation: ori, tr, str
mode for candidate augmentation: ori, tr, str
mode for search type: keyword, single_column, multi_column, unionable

```bash
# Module: src.search.tab2tab_aug (run from repo root with PYTHONPATH or `python -m` from env that has the package)
python -m src.search.tab2tab_aug --search_type keyword --query "../ModelTables/data/processed/deduped_hugging_csvs_v2_251117/00007f0e43_table1.csv" --k 3 --query_augmentation_types ori,tr,str --candidate_augmentation_types ori,tr,str --rerank_mode score_max --output_json tmp/tab2tab_aug_keyword.json
python -m src.search.tab2tab_aug --search_type single_column --query "../ModelTables/data/processed/deduped_hugging_csvs_v2_251117/00007f0e43_table1.csv" --k 3 --query_augmentation_types ori,tr,str --candidate_augmentation_types ori,tr,str --rerank_mode score_max --output_json tmp/tab2tab_aug_single_column.json
python -m src.search.tab2tab_aug --search_type multi_column --query "../ModelTables/data/processed/deduped_hugging_csvs_v2_251117/00007f0e43_table1.csv" --k 3 --query_augmentation_types ori,tr,str --candidate_augmentation_types ori,tr,str --rerank_mode score_max --output_json tmp/tab2tab_aug_multi_column.json
python -m src.search.tab2tab_aug --search_type unionable --query "../ModelTables/data/processed/deduped_hugging_csvs_v2_251117/00007f0e43_table1.csv" --k 3 --query_augmentation_types ori,tr,str --candidate_augmentation_types ori,tr,str --rerank_mode score_max --output_json tmp/tab2tab_aug_unionable.json
```

## 2.4 query2tab2card

```bash
python -m src.search.card2tab2card --resources hugging --model_id google-bert/bert-base-uncased --search_type keyword --table_top_k 3 --output_json tmp/card2tab2card_keyword_hugging.json
python -m src.search.card2tab2card --resources hugging --model_id google-bert/bert-base-uncased --search_type single_column --table_top_k 3 --output_json tmp/card2tab2card_single_column_hugging.json
python -m src.search.card2tab2card --resources hugging --model_id google-bert/bert-base-uncased --search_type multi_column --table_top_k 3 --output_json tmp/card2tab2card_multi_column_hugging.json
python -m src.search.card2tab2card --resources hugging --model_id google-bert/bert-base-uncased --search_type unionable --table_top_k 3 --output_json tmp/card2tab2card_unionable_hugging.json
```

is wrapped in below script.

```bash
# query2tab2card (new default): query -> seed card -> tab2tab -> related models
python -m src.search.query2tab2card --resources hugging --query "bert base uncased" --search_type keyword --table_top_k 3 --output_json tmp/query2tab2card_keyword_hugging.json
python -m src.search.query2tab2card --resources hugging --query "bert base uncased" --search_type single_column --table_top_k 3 --output_json tmp/query2tab2card_single_column_hugging.json
python -m src.search.query2tab2card --resources hugging --query "bert base uncased" --search_type multi_column --table_top_k 3 --output_json tmp/query2tab2card_multi_column_hugging.json
python -m src.search.query2tab2card --resources hugging --query "bert base uncased" --search_type unionable --table_top_k 3 --output_json tmp/query2tab2card_unionable_hugging.json

# Optional controls for query2tab2card:
#   --model_top_k 5
#   --q2m_top_k 20 --seed_rank_index 0 --disable_query_rerank
```

<details><summary>Optional 2.5 tab2tab_by_type</summary>
## (Optional) 2.5 tab2tab_by_type (test all modes; needs 1.3) (deprecated)

Same modes as tab2tab, filtered by table type. Paths from `src.config`. **--query** = table ID or CSV path.

```bash
python -m src.search.tab2tab_by_type --query 3690 --classification_json data/table_classifications.json --search_type keyword --k 10 --output_json data/tab2tab_by_type_keyword_results.json > logs/tab2tab_by_type_keyword.log 2>&1
python -m src.search.tab2tab_by_type --query 3690 --classification_json data/table_classifications.json --search_type single_column --k 10 --output_json data/tab2tab_by_type_single_column_results.json > logs/tab2tab_by_type_single_column.log 2>&1
python -m src.search.tab2tab_by_type --query 3690 --classification_json data/table_classifications.json --search_type multi_column --k 10 --output_json data/tab2tab_by_type_multi_column_results.json > logs/tab2tab_by_type_multi_column.log 2>&1
python -m src.search.tab2tab_by_type --query 3690 --classification_json data/table_classifications.json --search_type unionable --k 10 --output_json data/tab2tab_by_type_unionable_results.json > logs/tab2tab_by_type_unionable.log 2>&1 
```

If **multi_column** fails with `Scalar Function with name to_bitstring does not exist`: DuckDB version mismatch. In **Blend_internal** edit `src/Blend_internal/src/Operators/Seekers/MultiColumnOverlap.py` and replace `TO_BITSTRING(super_key)` with `to_binary(super_key)` (or the bitstring function your DuckDB provides). Then re-run.

## 2.6 Generate table comparison markdown (src/postprocess)

Generate markdown to view/compare tables by table ID or model ID; or from all logs (query + search results → md/).

```bash
python -m src.utils.generate_table_md --table_ids 3690 46228 26307 --output table_comparison.md
python -m src.utils.generate_table_md --model_id google-bert/bert-base-uncased --output model_tables.md
python -m src.utils.generate_md_from_logs --logs_dir logs --output_dir md
python -m src.utils.generate_md_from_logs --log_file logs/card2tab2card_by_type.log --output_dir md
```

**Outputs:** `logs/` (input); `md/<log_basename>.md` (one per log); `md/<log_basename>_materials/csv_integrated/integrated.csv` when integration finds CSVs. **Generated** = md file written for that log; **Failed** = no result JSON path in log (run that search with `--output_json` to fix). Pipeline: model-search = models first, then tables; table-search = tables only.

</details>


## 2.7 Demo (evaluation, QA, integration)

```bash
python -m src.demo.backend
python -m src.demo.frontend
# open http://localhost:5001
# if on server, redirect port 5001 to localhost:5001 and port 5002 to localhost:5002
ssh -L 5001:127.0.0.1:5001 -L 5002:127.0.0.1:5002 chippie.cs.uwaterloo.ca
```

**Backend warmup (default):** On startup, `python -m src.demo.backend` preloads the same NPZ + FAISS + `SentenceTransformer` caches used by jobs (`Query2ModelCard-FULL` npz + `Query2Tab2Card` npz from `TABLE_RESOURCE_ALLOWLIST`). The console prints `[startup]`, `[warmup]`, and `[load]` lines (NPZ path, FAISS build, encoder) so long stalls are visible. Skip warmup with `--no-warmup` or `BACKEND_SKIP_WARMUP=1`. Silence detailed `[load]` steps with `BACKEND_LOAD_QUIET=1`. Jobs still call `get_query2modelcard_dense_runtime` but hit **process-level** cache (no second disk read).

**Card2Tab2Card tab2tab:** If the seed model has **multiple** local CSVs, tab2tab over each table runs in parallel (up to 8 threads) when env `CARD2TAB2CARD_PARALLEL_TAB2TAB` is unset/`1`/`true`. Set to `0` to force sequential (e.g. if SQLite/Blend misbehaves under concurrency). The **three** `search_type` workers (keyword / single_column / unionable) stay **serialized** in the backend (`ThreadPoolExecutor(max_workers=1)`) to avoid CUDA init races. **Only one seed CSV → no parallel tab2tab** (wall time unchanged vs before).

**Job timing:** `pipeline_run.log` **Total time** = `/api/search` only (preload + query2modelcard + three `query2tab2card` + Card2Card list prep). **Integration** is a **separate** `POST /api/integrate` (or `/api/integrate-model-search`); it is **not** included in that total. After calling integrate, the log gains appended lines and JSON includes `integration_elapsed_s`.


## 3. Helpful scripts

mimic user for batch running
```bash
python scripts/batch_run_preset_queries.py \
  --backend_url http://localhost:5002 \
  --preset_path config/preset_queries.json \
  --run_integration \
  --integration_type alite \
  --integration_search_types single_column unionable keyword
```

generate markdown for query table and retrieved tables and integrated tables
```bash
python scripts/check_retrieval_integration_consistency.py \
  --jobs-root data_251117/jobs_251117 \
  --search-types single_column unionable keyword \
  --per-job-md \
  --preview-max-rows 0 --preview-max-cols 0
```

---

# Inference fast checklist

1. **Run Part 1 once** (modelcard indexes via **`ir_index_builder`**, Blend DuckDB, optional table classification for by_type).
2. **Do not** re-run index builders or `classification --mode batch` during serving or per-query scripts.
3. Inference **loads** artifacts only: card2card **`EMB_NPZ` + Lucene `SPARSE_INDEX`** (or hugging subsets), `modellake.db`, optional `table_classifications.json`. Use **`ir_searcher`** (`DenseSearcher` / `SparseSearcher`) for modelcard retrieval.
4. For tab2know: batch classification (optional 1.3) runs tab2know **inference** per table and writes JSON; at query time we only `load_classifications(json)`. Tab2know’s own model training lives in TabKnow_internal (separate repo).
