# Evaluate

## Content

1. [Model Card Extract Nuggets](#1-model-card-extract-nuggets)
2. [Get Query-Nuggets List Mapping](#2-get-query-nuggets-list-mapping)
3. [Evaluate](#3-evaluate)
4. [Reference](#4-reference)

## 1. Model Card Extract Nuggets


Batch mode (recommended):

```bash
python -m src.evaluate.card2nugget_extraction --model-ids-file data_251117/query/toy_data/model_ids.txt
```

Outputs:

- `data_251117/evaluate/<model_id_with___>.csv` — parsed tuples table (columns: `Model`, `Base_model`, `Dataset`, `Train_dataset`, `Test_dataset`, `Model_hyperparameters`, `Model_variant_type`, `Metric`, `Metric_value`)
- `data_251117/evaluate/<model_id_with___>_meta.yaml` — metadata + full `prompt` + raw `response`
- Batch mode outputs are written to `data_251117/evaluate/batch/` (including per-model csv/meta and Batch API input/output jsonl files for debugging).

The LLM sees the **full model card text** from parquet (tables + prose mixed). **No local Hugging Face CSV tables** are appended. Optional env: `MODELSEARCHDEMO_CARD_MAX_CHARS` (default `100000`) truncates the card in the prompt; hparam regex fallback still uses the full card after citation stripping.

Optional: `python -m src.evaluate.card2nugget_extraction --read-csv` prints stats for a CSV path.

## 2. Query → nugget schema headers (LLM)

Map a **search query** to the same column headers as `card2nugget_extraction` (`Model`, `Dataset`, `Metric`, …). The LLM returns, for each relevant header, a **keyword list** explaining how the query ties to that field.

Prompt: `src/evaluate/query2nugget_prompts.yaml` (`query_to_nugget_headers`).

```bash
python -m src.evaluate.query2nugget_layer_mapping --queries-file data_251117/query/toy_data/queries.txt
```

`query2nugget_layer_mapping` now always uses OpenAI Batch API.  

**Output file (`query_header_keyword_mapping.json`)** — intermediate artifact from the LLM: your search `query`, plus `related` entries (`header` + `keywords` per nugget column). Single-query runs store one JSON object; multiple queries store `{"queries": [...]}`. Entries may include `batch_input_jsonl` / `batch_output_jsonl` for Batch debugging.

**Pipeline:** run **card2nugget** first (per-model CSVs under `evaluate/` or `evaluate/batch/`), then **query2nugget** as above, then match CSV rows to keywords to build pyndeval inputs:

```bash
python -m src.evaluate.query_csv_to_qrels_run
# optional: --mapping path/to/query_header_keyword_mapping.json
# optional: --csv-root path/to/dir  (repeatable; defaults to evaluate/ + evaluate/batch/)
# optional: --qrels / --run / --debug-json paths
```

Or do the same matching directly from query2nugget in one command:

```bash
python -m src.evaluate.query2nugget_layer_mapping \
  --queries-file data_251117/query/toy_data/queries.txt \
  --build-qrels-run
# optional: --csv-root (repeatable), --qrels-output, --run-output, --match-debug-output
```

Matching rules (simple, debug-friendly): for each query, **every** `related` header must have a **nonempty** cell in the row; each header’s keywords are **OR**’d; a keyword counts if it appears as a **case-insensitive substring** in that cell (e.g. keyword `bert` matches `bert-base-uncased`). Doc ids are `{csv_stem}#{0_based_row}` (csv stem is the safe model id, e.g. `google-bert__bert-base-uncased`). Also writes `query_csv_match_debug.json` (per-query hits with row snapshots).

One-shot wrapper (two model clusters, query file input):

```bash
python -m src.evaluate.wrap_card_query_eval \
  --queries-file data_251117/query/toy_data/queries.txt \
  --cluster-a-model-ids-file data_251117/query/toy_data/cluster_a_model_ids.txt \
  --cluster-b-model-ids-file data_251117/query/toy_data/cluster_b_model_ids.txt
```

This wrapper imports and chains the 3 modules above: `card2nugget_extraction`, `query2nugget_layer_mapping`, `query_csv_to_qrels_run`. Outputs are written under `data_251117/evaluate/pipeline/` by default, including per-cluster `.qrels`, `.run`, match debug json, and `pipeline_summary.json`.

## 3. Evaluate
```bash
# Toy data:
python -m src.evaluate.evaluate_pyndeval --run data_251117/toy_data/toy_initial.run --qrels data_251117/toy_data/toy_subtopic.qrels --cutoff 20

# Real data:
python -m src.evaluate.evaluate_pyndeval --run data_251117/evaluate/real_initial.run --qrels data_251117/evaluate/real_subtopic.qrels --cutoff 20
```


## 4. Repo I tested

| Name | Tested | Why Not Used / Issue | Year | Type |
| --- | --- | --- | --- | --- |
| [AxCell](https://aclanthology.org/2020.emnlp-main.692/) | Yes | Non-prompt extraction; mainly focuses on tables; local pytest failed and requires Docker; Docker not tested; input is TeX-based, not ModelTables card text. | 2020 | Regular extraction |
| [MetaLead](https://researchtrend.ai/papers/2601.22420) | Yes | Prompt-based; extracts from PDF/HTML text, so table structure is lost; OpenAI issue was fixed locally, but on ModelTables it extracted too few nuggets and was incomplete. | 2026 | Prompt-based |
| [SciLead](https://aclanthology.org/2024.emnlp-main.453/) | Not implemented | Dataset could not be downloaded; table display is invalid. | 2024 | Prompt-based |
