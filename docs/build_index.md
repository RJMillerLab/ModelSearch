# Build Index

**Part 1** = index building (run once). **Part 2** = inference (retrieval, table search, demo). When a step has multiple modes, run all and save to separate log/json.

---

# Preprocessing vs inference (what to run once vs per query)

| Phase | What | When |
|-------|------|------|
| **Preprocessing (run once)** | **Build modelcard index** — encode full corpus → `.npz` + `.faiss`; optional `.jsonl`. | Before query2modelcard / card2card. |
| | **Build sparse index** — BM25 over full corpus → `data/card2card_bm25.pkl` (1.1b). | Before card2card sparse/hybrid; inference then only loads it. |
| | **Blend + data** — clone/symlink Blend_internal and data dirs. | Once per env. |
| | **DuckDB table index** — `create_index_duckdb` → `data/modellake.db` with `modellake_index`. | Before card2tab2card, tab2tab, tab2tab_by_type. |
| | **Table classification (batch)** — `classification --mode batch` → `data/table_classifications.json`. Uses **tab2know inference** per table (tab2know’s own “training” is in TabKnow repo; we only run its pretrained type/column models here). | Once; required only for **by_type** flows (card2tab2card by_type, tab2tab_by_type). |
| | **Baseline2** — mapping scripts + pyserini Lucene index (`data/tmp/index`). | Before baseline2 search. |
| | **Baseline3** — build dense index. | Before baseline3 hybrid search. |
| **Inference (per query / serving)** | query2modelcard, card2card search, card2tab2card, tab2tab, tab2tab_by_type, demo backend, generate_md_from_logs. | They **only load** prebuilt artifacts (faiss, npz, jsonl, modellake.db, table_classifications.json). No index build, no batch classification. |

**Making inference fast:** Run Part 1 once (all steps you need for your flows). Then Part 2 is just loading those artifacts and running retrieval. Do **not** run build-index or classification batch at request time; keep them as one-off preprocessing.

---

# Part 1 — Must run (in order)

## 1.1 Modelcard index (dense)

```bash
# one-step (recommended)
python -m src.search.card2card build-index --field card --raw_dir data_citationlake/raw --output_npz data/card2card_embeddings.npz --output_index data/card2card.faiss
# optional: --output_jsonl data/card2card_corpus.jsonl --device cuda
```

Optional 3-step (baseline1): `build_modelcard_jsonl` → `table_retrieval_pipeline encode` → `table_retrieval_pipeline build_faiss`.

## 1.1b Sparse index (for card2card sparse/hybrid)

Run after 1.1 so that inference **only loads** the prebuilt BM25 file (no jsonl load/tokenize/build at request time).

```bash
python -m src.search.card2card build-sparse-index --jsonl_path data/card2card_corpus.jsonl --output_bm25 data/card2card_bm25.pkl
```

## 1.2 Blend + data

```bash
# clone or symlink Blend_internal; symlink data if needed
git clone git@github.com:DoraDong-2023/Blend_internal.git src/Blend_internal
# ln -s /path/to/ModelTables/data data_citationlake
```

## 1.3 DuckDB table index

```bash
python -m src.Blend_internal.scripts.create_index_duckdb --db_path data/modellake.db --data_glob "data_citationlake/processed/deduped_github_csvs/*.csv" --data_glob "data_citationlake/processed/deduped_hugging_csvs/*.csv" --data_glob "data_citationlake/processed/tables_output/*.csv" --table modellake_index
```

## 1.4 Table classification (optional; for by_type flows)

Train vs inference = explicit arg only, no fallback. Train = run below with `--mode batch`. Inference = run 2.3 by_type / 2.5 with `--classification_json data/table_classifications.json`. Same `--db_path` for 1.4 and 2.3/2.5.

```bash
# train (full datalake)
python -m src.search.classification --mode batch --db_path data/modellake.db --output_json data/table_classifications.json
# heuristic if tab2know fails
python -m src.search.classification --mode batch --db_path data/modellake.db --output_json data/table_classifications.json --method heuristic
```

---

# Part 2 — Inference (test all modes per tool)

## 2.1 query2modelcard

```bash
python -m src.search.query2modelcard --query "transformer model for code generation" --emb_npz data/card2card_embeddings.npz --faiss_index data/card2card.faiss --top_k 20 --device cuda > logs/query2modelcard.log 2>&1
```

## 2.2 card2card (dense, sparse, hybrid)

**Fixed (Part 1):** Run `build-sparse-index` (1.1b) once to save `data/card2card_bm25.pkl`. Then inference with `--bm25_index_path data/card2card_bm25.pkl` **only loads** that file (no jsonl load/tokenize/build). Dense already uses only prebuilt FAISS+npz. **Per-step timings:** card2card prints `[timing]` lines; grep logs for `[timing]` to debug.

```bash
# dense (fast: FAISS lookup only)
python -m src.search.card2card search --model_id google-bert/bert-base-uncased --emb_npz data/card2card_embeddings.npz --faiss_index data/card2card.faiss --top_k 20 --retrieval_mode dense > logs/card2card_dense.log 2>&1
# sparse (use prebuilt BM25 from 1.1b so inference only loads, no rebuild)
python -m src.search.card2card search --model_id google-bert/bert-base-uncased --bm25_index_path data/card2card_bm25.pkl --top_k 20 --retrieval_mode sparse > logs/card2card_sparse.log 2>&1
# hybrid (same: prebuilt BM25 + FAISS)
python -m src.search.card2card search --model_id google-bert/bert-base-uncased --emb_npz data/card2card_embeddings.npz --faiss_index data/card2card.faiss --bm25_index_path data/card2card_bm25.pkl --top_k 20 --retrieval_mode hybrid --hybrid_method rrf > logs/card2card_hybrid.log 2>&1
```

## 2.3 card2tab2card (keyword, all, by_type)

Use `--db_path` to point to modellake.db (default: `data_citationlake/modellake.db`). Example with `data/modellake.db`:

```bash
# single type: keyword
python -m src.search.card2tab2card --model_id google-bert/bert-base-uncased --search_type keyword --k 10 --db_path data/modellake.db > logs/card2tab2card_keyword.log 2>&1
# all search types (single_column, keyword, unionable)
python -m src.search.card2tab2card --model_id google-bert/bert-base-uncased --mode all --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv --output_folder data --db_path data/modellake.db > logs/card2tab2card_all.log 2>&1
# by table type (needs 1.4)
python -m src.search.card2tab2card --model_id google-bert/bert-base-uncased --mode by_type --classification_json data/table_classifications.json --db_path data/modellake.db > logs/card2tab2card_by_type.log 2>&1
```

## 2.4 tab2tab (test all modes: keyword, single_column, multi_column, unionable)

Keyword = match column names (headers). Single_column = match cell values in one column (often 0 results if values are unique). Multi_column/unionable = CSV path.

```bash
# keyword: comma-separated column names
python -m src.search.tab2tab --search_type keyword --query "model_name,accuracy,task" --k 10 --db_path data/modellake.db --output data/tab2tab_keyword_results.json > logs/tab2tab_keyword.log 2>&1
# single_column: comma-separated cell values
python -m src.search.tab2tab --search_type single_column --query "val1,val2,val3" --k 10 --db_path data/modellake.db --output data/tab2tab_single_column_results.json > logs/tab2tab_single_column.log 2>&1
# multi_column: path to CSV
python -m src.search.tab2tab --search_type multi_column --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv --k 10 --db_path data/modellake.db --output data/tab2tab_multi_column_results.json > logs/tab2tab_multi_column.log 2>&1
# unionable: path to CSV
python -m src.search.tab2tab --search_type unionable --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv --k 10 --db_path data/modellake.db --output data/tab2tab_unionable_results.json > logs/tab2tab_unionable.log 2>&1
```

## 2.5 tab2tab_by_type (test all modes; needs 1.4)

Same modes as tab2tab, filtered by table type. **--query** = table ID (from modellake.db) or CSV path. Table ID: load from db (filename → same resolve as elsewhere) so one source, no local path needed.

```bash
# keyword (table ID or path)
python -m src.search.tab2tab_by_type --query 3690 --classification_json data/table_classifications.json --search_type keyword --k 10 --db_path data/modellake.db --output data/tab2tab_by_type_keyword_results.json > logs/tab2tab_by_type_keyword.log 2>&1
# single_column
python -m src.search.tab2tab_by_type --query 3690 --classification_json data/table_classifications.json --search_type single_column --k 10 --db_path data/modellake.db --output data/tab2tab_by_type_single_column_results.json > logs/tab2tab_by_type_single_column.log 2>&1
# multi_column
python -m src.search.tab2tab_by_type --query 3690 --classification_json data/table_classifications.json --search_type multi_column --k 10 --db_path data/modellake.db --output data/tab2tab_by_type_multi_column_results.json > logs/tab2tab_by_type_multi_column.log 2>&1
# unionable
python -m src.search.tab2tab_by_type --query 3690 --classification_json data/table_classifications.json --search_type unionable --k 10 --db_path data/modellake.db --output data/tab2tab_by_type_unionable_results.json > logs/tab2tab_by_type_unionable.log 2>&1 
```

If **multi_column** fails with `Scalar Function with name to_bitstring does not exist`: DuckDB version mismatch. In **Blend_internal** edit `src/Blend_internal/src/Operators/Seekers/MultiColumnOverlap.py` and replace `TO_BITSTRING(super_key)` with `to_binary(super_key)` (or the bitstring function your DuckDB provides). Then re-run.

## 2.6 Generate table comparison markdown (src/postprocess)

Generate markdown to view/compare tables by table ID or model ID; or from all logs (query + search results → md/).

```bash
# By table ID(s) or model ID
python -m src.postprocess.generate_table_md --table_ids 3690 46228 26307 --output table_comparison.md
python -m src.postprocess.generate_table_md --model_id google-bert/bert-base-uncased --output model_tables.md
# From all logs → md/<log_basename>.md (run from repo root, e.g. on remote)
python -m src.postprocess.generate_md_from_logs --logs_dir logs --output_dir md --db_path data/modellake.db
# Single log
python -m src.postprocess.generate_md_from_logs --log_file logs/card2tab2card_by_type.log --output_dir md
```

**Outputs:** `logs/` (input); `md/<log_basename>.md` (one per log); `md/<log_basename>_materials/csv_integrated/integrated.csv` when integration finds CSVs. **Generated** = md file written for that log; **Failed** = no result JSON path in log (run that search with `--output_json` to fix). Pipeline: model-search = models first, then tables; table-search = tables only.

## 2.7 Demo (evaluation, QA, integration)

```bash
python -m src.demo.backend
python -m src.demo.frontend
# open http://localhost:5001
```

## 2.8 baseline2 / baseline3 (table retrieval; from ModelTables)

Copy from ModelTables then run. Full flow: **ModelTables/docs/scripts.md** Section 7.

```bash
cp -r ../ModelTables/src/baseline2 src/
cp -r ../ModelTables/src/baseline3 src/
# symlinks for data/tmp, data/analysis, data/processed, etc. as in ModelTables
# TAG=251117 bash src/baseline2/get_metadata.sh > logs/baseline2_get_metadata_251117.log 2>&1
# TAG=251117 bash src/baseline2/sparse_search.sh > logs/baseline2_sparse_search_251117.log 2>&1
# baseline3: build dense index then search_with_pyserini_hybrid.py or hybrid_search.sh
```

---

# Quick reference

| Part 1 | |
|--------|---|
| Modelcard index | card2card build-index (or baseline1 3 steps) |
| Sparse index (card2card) | card2card build-sparse-index --jsonl_path ... --output_bm25 data/card2card_bm25.pkl |
| Blend | clone/symlink src/Blend_internal |
| DuckDB index | create_index_duckdb --db_path data/modellake.db --data_glob ... |
| Table classification | classification --mode batch --db_path data/modellake.db --output_json data/table_classifications.json (or --method heuristic) |
| **Part 2** | |
| query2modelcard | query2modelcard --query "..." --emb_npz ... --faiss_index ... --top_k 20 |
| card2card | search --retrieval_mode dense \| sparse \| hybrid (test all 3) |
| card2tab2card | --search_type keyword, --mode all, --mode by_type (test all) |
| tab2tab | --search_type keyword \| single_column \| multi_column \| unionable (test all 4) |
| tab2tab_by_type | same 4 search_type with --classification_json (test all 4) |
| Generate table MD | `python -m src.postprocess.generate_table_md` (--table_ids / --model_id); `generate_md_from_logs` for logs → md/ |
| Demo | backend + frontend → http://localhost:5001 |

Scripts print Total time at exit; redirect to `logs/*.log` to keep timings.

---

# Log timings (from logs/*.log, latest run)

| Log file | Total time (s) | Device | Note |
|----------|----------------|--------|------|
| query2modelcard.log | 10.09 | cuda | |
| card2card_dense.log | 11.11 | cuda | load npz 1.7s, FAISS 1.3s, search 7.7s |
| card2card_sparse.log | 728.44 | cuda | see breakdown below |
| card2card_hybrid.log | 752.79 | cuda | sparse-dominated; dense ~10s |
| tab2tab_keyword.log | 0.11 | cpu | |
| tab2tab_single_column.log | 0.09 | cpu | |
| tab2tab_by_type.log | 0.21 | — | (re-run to get device in line) |
| card2tab2card_* / tab2tab_by_type_* | (see prior rows if present) | | |
| tab2tab_by_type_multi_column.log | — | | DuckDB to_bitstring; fix in Blend_internal (2.5) |

**Sparse (728s) per-step breakdown (1.1M docs):** load jsonl 22.7s, tokenize 57.2s, build BM25 102.6s, **get_scores + sort 539s** (main cost). Hybrid: same sparse branch ~742s then load npz + FAISS ~10s.

Run `grep -h "Total time\|\[timing\]" logs/*.log` to refresh.

---

# Inference fast checklist

1. **Run Part 1 once** (modelcard index, Blend, DuckDB index; table classification only if you use by_type).
2. **Do not** call `build-index` or `classification --mode batch` during serving or per-query scripts.
3. Inference scripts only **load** prebuilt files: `emb_npz`, `faiss_index`, `jsonl`, `modellake.db`, `table_classifications.json`.
4. For tab2know: batch classification (Part 1.4) runs tab2know **inference** per table and writes JSON; at query time we only `load_classifications(json)`. Tab2know’s own model training lives in TabKnow_internal (separate repo).
