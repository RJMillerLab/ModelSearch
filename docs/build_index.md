# Build Index

**Part 1** = index building (run once). **Part 2** = inference (retrieval, table search, demo). When a step has multiple modes, run all and save to separate log/json.

---

# Part 1 — Must run (in order)

## 1.1 Modelcard index

```bash
# one-step (recommended)
python -m src.search.card2card build-index --field card --raw_dir data_citationlake/raw --output_npz data/card2card_embeddings.npz --output_index data/card2card.faiss
# optional: --output_jsonl data/card2card_corpus.jsonl --device cuda
```

Optional 3-step (baseline1): `build_modelcard_jsonl` → `table_retrieval_pipeline encode` → `table_retrieval_pipeline build_faiss`.

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

**IMPORTANT:** Use the **same `--db_path`** for classification (1.4) and card2tab2card/tab2tab_by_type (2.3/2.5). Different db_path = different table IDs = 0 filtered results.

```bash
# tab2know (default)
python -m src.search.classification --mode batch --db_path data/modellake.db --output_json data/table_classifications.json
# heuristic if tab2know fails on server
python -m src.search.classification --mode batch --db_path data/modellake.db --output_json data/table_classifications.json --method heuristic
```

---

# Part 2 — Inference (test all modes per tool)

## 2.1 query2modelcard

```bash
python -m src.search.query2modelcard --query "transformer model for code generation" --emb_npz data/card2card_embeddings.npz --faiss_index data/card2card.faiss --top_k 20 --device cuda > logs/query2modelcard.log 2>&1
```

## 2.2 card2card (dense, sparse, hybrid)

```bash
# dense
python -m src.search.card2card search --model_id google-bert/bert-base-uncased --emb_npz data/card2card_embeddings.npz --faiss_index data/card2card.faiss --top_k 20 --retrieval_mode dense > logs/card2card_dense.log 2>&1
# sparse
python -m src.search.card2card search --model_id google-bert/bert-base-uncased --jsonl_path data/card2card_corpus.jsonl --top_k 20 --retrieval_mode sparse > logs/card2card_sparse.log 2>&1
# hybrid
python -m src.search.card2card search --model_id google-bert/bert-base-uncased --emb_npz data/card2card_embeddings.npz --faiss_index data/card2card.faiss --jsonl_path data/card2card_corpus.jsonl --top_k 20 --retrieval_mode hybrid --hybrid_method rrf > logs/card2card_hybrid.log 2>&1
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

Same modes as tab2tab, with results filtered to same table type as query.

```bash
# keyword
python -m src.search.tab2tab_by_type --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv --classification_json data/table_classifications.json --search_type keyword --k 10 --db_path data/modellake.db --output data/tab2tab_by_type_keyword_results.json > logs/tab2tab_by_type_keyword.log 2>&1
# single_column (often 0; falls back to keyword in code)
python -m src.search.tab2tab_by_type --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv --classification_json data/table_classifications.json --search_type single_column --k 10 --db_path data/modellake.db --output data/tab2tab_by_type_single_column_results.json > logs/tab2tab_by_type_single_column.log 2>&1
# multi_column
python -m src.search.tab2tab_by_type --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv --classification_json data/table_classifications.json --search_type multi_column --k 10 --db_path data/modellake.db --output data/tab2tab_by_type_multi_column_results.json > logs/tab2tab_by_type_multi_column.log 2>&1
# unionable
python -m src.search.tab2tab_by_type --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv --classification_json data/table_classifications.json --search_type unionable --k 10 --db_path data/modellake.db --output data/tab2tab_by_type_unionable_results.json > logs/tab2tab_by_type_unionable.log 2>&1
```

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

Output: table metadata, classification (if available), CSV preview, column info. Table IDs are read from result JSON when present, else from log lines `N. Table ID: <id>`.

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

Scripts print `⏱️ Total time` at exit; redirect to `logs/*.log` to keep timings.
