# Build Index

**Paths:** Data paths (ModelTables/data, modellake.db, processed dirs, relationship parquet) are defined in `src.config`. Override with env: `MODELTABLES_DATA` or `DATA_ROOT`, `MODELLAKE_DB`, `DATA_TAG`. Examples below use config defaults.

**Part 1** = index building (run once). **Part 2** = inference (retrieval, table search, demo). When a step has multiple modes, run all and save to separate log/json.

---

# Preprocessing vs inference (what to run once vs per query)

| Phase | What | When |
|-------|------|------|
| **Preprocessing (run once)** | **Build modelcard index** — encode full corpus → `.npz` + `.faiss`. | Before query2modelcard / card2card. |
| | **Build sparse index** — Pyserini Lucene BM25 → `data/card2card_sparse_index` (1.1b). | Train once; inference then only loads index. |
| | **Blend + data** — clone/symlink Blend_internal and data dirs. | Once per env. |
| | **Valid model IDs txt** — `scripts/build_valid_model_ids_txt.py` → `data/valid_model_ids_with_tables.txt`. | Optional; before demo “Narrow down” (seed with tables). |
| | **DuckDB table index** — `create_index_duckdb` → `data/modellake.db` with `modellake_index`. | Before card2tab2card, tab2tab, tab2tab_by_type. |
| | **Table classification (batch)** — `classification --mode batch` → `data/table_classifications.json`. Uses **tab2know inference** per table (tab2know’s own “training” is in TabKnow repo; we only run its pretrained type/column models here). | Once; required only for **by_type** flows (card2tab2card by_type, tab2tab_by_type). |
| | **Baseline2** — mapping scripts + pyserini Lucene index (`data/tmp/index`). | Before baseline2 search. |
| | **Baseline3** — build dense index. | Before baseline3 hybrid search. |
| **Inference (per query / serving)** | query2modelcard, card2card search, card2tab2card, tab2tab, tab2tab_by_type, demo backend, generate_md_from_logs. | They **only load** prebuilt artifacts (faiss, npz, jsonl, modellake.db, table_classifications.json). No index build, no batch classification. |

**Making inference fast:** Run Part 1 once (all steps you need for your flows). Then Part 2 is just loading those artifacts and running retrieval. Do **not** run build-index or classification batch at request time; keep them as one-off preprocessing.

---

# Part 1 — Must run (in order)

```bash
git clone git@github.com:RJMillerLab/ModelTables.git
# Install dependencies and download data, as remaining steps depend on it
```

## 1.1 Modelcard index (dense)

```bash
# build index for dense retrieval (FAISS) and sparse retrieval (Pyserini Lucene) from modelcard_step1.parquet
python -m src.search.card2card build-dense-index
python -m src.search.card2card build-sparse-index
```

## 1.2 Blend + data

```bash
# Build DuckDB index from csvs, for later search
git clone git@github.com:DoraDong-2023/Blend_internal.git 
ln -s ../Blend_internal others/Blend_internal
# create DuckDB index from csvs, by following the instructions from Blend README. Output: modellake_v2_251117.db
```

## 1.2b Valid model IDs for Table Search (optional; for demo “Narrow down”)

Extract model_id that have tables (non-empty `csv_basename` in relationship parquet) into a txt so inference only loads it. Run once after parquet is available.

```bash
python -m src.utils.build_valid_model_ids_txt --output data_251117/valid_model_ids_with_tables.txt
```

## (Optional) 1.3 Table classification (optional; for by_type flows)

Train vs inference = explicit arg only, no fallback. Train = run below with `--mode batch`. Inference = run 2.3 by_type / 2.5 with `--classification_json data/table_classifications.json`. Same `--db_path` for 1.4 and 2.3/2.5.

```bash
# train (full datalake)
python -m src.search.classification --mode batch --output_json data/table_classifications.json
# heuristic if tab2know fails
python -m src.search.classification --mode batch --output_json data/table_classifications.json --method heuristic
```

---

# Part 2 — Inference

## 2.1 query2modelcard

```bash
python -m src.search.query2modelcard --query "transformer model for code generation" --top_k 20 --retrieval_mode dense > logs/query2modelcard.log 2>&1
```

## 2.2 card2card (dense, sparse, hybrid)

**Train vs inference:** Part 1 (1.1, 1.1b) = train: build modelcard index and sparse Lucene index. Part 2 = inference: only load those artifacts and run retrieval. Sparse uses Pyserini (same as ModelTables baseline2).

```bash
# model ids file (single query also works)
echo "google-bert/bert-base-uncased" > data/tmp/card2card_model_ids.txt
# dense (FAISS only)
# python -m src.search.card2card search --model_id ... is deprecated; use --model_ids_file instead
python -m src.search.card2card search --model_ids_file data/tmp/card2card_model_ids.txt --top_k 20 --retrieval_mode dense > logs/card2card_dense.log 2>&1
# sparse (Pyserini Lucene index from 1.1b)
python -m src.search.card2card search --model_ids_file data/tmp/card2card_model_ids.txt --top_k 20 --retrieval_mode sparse > logs/card2card_sparse.log 2>&1
# hybrid (sparse + FAISS)
python -m src.search.card2card search --model_ids_file data/tmp/card2card_model_ids.txt --top_k 20 --retrieval_mode hybrid > logs/card2card_hybrid.log 2>&1
```

## 2.3 tab2tab (test all modes: keyword, single_column, multi_column, unionable)

Paths from `src.config`. Keyword = match column names; single_column = cell values; multi_column/unionable = CSV path.

```bash
# keyword
python -m src.search.tab2tab --search_type keyword --query "model_name,accuracy,task" --k 10 > logs/tab2tab_keyword.log 2>&1
python -m src.search.tab2tab --search_type single_column --query "val1,val2,val3" --k 10 > logs/tab2tab_single_column.log 2>&1
python -m src.search.tab2tab --search_type multi_column --query "$(python scripts/get_config_paths.py sample_csv)" --k 10 > logs/tab2tab_multi_column.log 2>&1
python -m src.search.tab2tab --search_type unionable --query "$(python scripts/get_config_paths.py sample_csv)" --k 10 > logs/tab2tab_unionable.log 2>&1
```

## 2.4 card2tab2card (keyword, single_column, unionable)

```bash
python -m src.search.card2tab2card --model_id google-bert/bert-base-uncased --search_type keyword --k 10 > logs/card2tab2card_keyword.log 2>&1
python -m src.search.card2tab2card --model_id google-bert/bert-base-uncased --search_type single_column --k 10 > logs/card2tab2card_single_column.log 2>&1
python -m src.search.card2tab2card --model_id google-bert/bert-base-uncased --search_type unionable --k 10 > logs/card2tab2card_unionable.log 2>&1
```

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
python -m src.postprocess.generate_table_md --table_ids 3690 46228 26307 --output table_comparison.md
python -m src.postprocess.generate_table_md --model_id google-bert/bert-base-uncased --output model_tables.md
python -m src.postprocess.generate_md_from_logs --logs_dir logs --output_dir md
python -m src.postprocess.generate_md_from_logs --log_file logs/card2tab2card_by_type.log --output_dir md
```

**Outputs:** `logs/` (input); `md/<log_basename>.md` (one per log); `md/<log_basename>_materials/csv_integrated/integrated.csv` when integration finds CSVs. **Generated** = md file written for that log; **Failed** = no result JSON path in log (run that search with `--output_json` to fix). Pipeline: model-search = models first, then tables; table-search = tables only.

## 2.7 Demo (evaluation, QA, integration)

```bash
python -m src.demo.backend
python -m src.demo.frontend
# open http://localhost:5001
```




---

# Quick reference

| Part 1 | |
|--------|---|
| Modelcard index | card2card build-index (or baseline1 3 steps) |
| Sparse index (card2card) | card2card build-sparse-index |
| Blend | clone/symlink src/Blend_internal |
| DuckDB index | create_index_duckdb --db_path data/modellake.db --data_glob ... |
| Table classification | classification --mode batch --db_path data/modellake.db --output_json data/table_classifications.json (or --method heuristic) |
| **Part 2** | |
| query2modelcard | query2modelcard --query "..." --top_k 20 |
| card2card | search --retrieval_mode dense \| sparse \| hybrid (test all 3) |
| card2tab2card | --search_type keyword/single_column/unionable (simplified) |
| tab2tab | --search_type keyword \| single_column \| multi_column \| unionable (test all 4) |
| tab2tab_by_type | same 4 search_type with --classification_json (test all 4) |
| Generate table MD | `python -m src.postprocess.generate_table_md` (--table_ids / --model_id); `generate_md_from_logs` for logs → md/ |
| Demo | backend + frontend → http://localhost:5001 |

Scripts print Total time at exit; redirect to `logs/*.log` to keep timings.

---

# Log timings (from logs/*.log, latest run)

**Inference time:** Use the line `[timing] inference (min)` in logs. Load time (sparse index, npz, FAISS) is not counted. With **Pyserini** sparse, inference is get query text + BM25 top-k search (fast); older rank_bm25 sparse scored all docs (~630s).

| Log file | Total (s) | Inference (min) | Device | Note |
|----------|-----------|-----------------|--------|------|
| query2modelcard.log | 10.09 | — | cuda | |
| card2card_dense.log | 11.11 | ~7.7 | cuda | FAISS search only |
| card2card_sparse.log | 4.04 | 0.67 | cpu | load index 3.37s, get query+truncate 0.03s, BM25 top-k 0.65s |
| card2card_hybrid.log | 17.22 | 7.92 | cuda | sparse branch 2.75s, load npz 1.76s, FAISS 4.68s, search 7.60s |
| tab2tab_keyword.log | 0.11 | — | cpu | |
| tab2tab_single_column.log | 0.09 | — | cpu | |
| tab2tab_by_type.log | 0.21 | — | — | (script may print ⏱️ Total time without device) |
| card2tab2card_* / tab2tab_by_type_* | (see prior rows if present) | | | |
| tab2tab_by_type_multi_column.log | — | — | | DuckDB to_bitstring; fix in Blend_internal (2.5) |

**Sparse (Pyserini):** inference (min) = get query text + truncate + BM25 top-k search (~0.67s). **Hybrid:** inference (min) = sparse step + FAISS search + combine (~7.92s).

Run `grep -h "Total time\|inference (min)\|\[timing\]" logs/*.log` to refresh.

---

# Inference fast checklist

1. **Run Part 1 once** (modelcard index, Blend, DuckDB index; table classification only if you use by_type).
2. **Do not** call `build-index` or `classification --mode batch` during serving or per-query scripts.
3. Inference scripts only **load** prebuilt files: `emb_npz`, `faiss_index`, `jsonl`, `modellake.db`, `table_classifications.json`.
4. For tab2know: batch classification (Part 1.4) runs tab2know **inference** per table and writes JSON; at query time we only `load_classifications(json)`. Tab2know’s own model training lives in TabKnow_internal (separate repo).
