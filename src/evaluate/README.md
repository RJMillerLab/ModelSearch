# Evaluate

## Content

1. [Model Card Extract Nuggets](#1-model-card-extract-nuggets)
2. [Get Query-Nuggets List Mapping](#2-get-query-nuggets-list-mapping)
3. [Evaluate](#3-evaluate)
4. [Reference](#4-reference)

## 1. Model Card Extract Nuggets

```bash
python -m src.evaluate.card2nugget_extraction --model-ids-file data_251117/query/toy_data/model_ids.txt
```

Writes per-model CSV + meta under `data_251117/evaluate/batch/` (batch API).

## 2. Query → nugget schema headers (LLM)

After step 1, map queries (OpenAI Batch) and optionally build `qrels` / `.run` from those CSVs:

```bash
python -m src.evaluate.query2nugget_layer_mapping --queries-file data_251117/query/toy_data/queries.txt --build-qrels-run
```

Writes `query_header_keyword_mapping.json`, `real_subtopic.qrels`, `real_initial.run`, and `query_csv_match_debug.json` under `data_251117/evaluate/` by default.

End-to-end (two clusters + eval):

```bash
python -m src.evaluate.wrap_card_query_eval \
  --queries-file data_251117/query/toy_data/queries.txt \
  --cluster-a-model-ids-file data_251117/query/toy_data/cluster_a_model_ids.txt \
  --cluster-b-model-ids-file data_251117/query/toy_data/cluster_b_model_ids.txt
```

From saved batch jobs (`jobs_251117/batch_runs/…json`):

```bash
python -m src.evaluate.wrap_card_query_eval \
  --jobs-batch-json jobs_251117/batch_runs/batch_preset_queries_20260330_025232.json \
  --max-job-sets 6 \
  --dedupe-model-sets
```

## 3. Evaluate

```bash
python -m src.evaluate.evaluate_pyndeval \
  --run data_251117/evaluate/real_initial.run \
  --qrels data_251117/evaluate/real_subtopic.qrels \
  --cutoff 20
```


## 4. Repo I tested

| Name | Tested | Why Not Used / Issue | Year | Type |
| --- | --- | --- | --- | --- |
| [AxCell](https://aclanthology.org/2020.emnlp-main.692/) | Yes | Non-prompt extraction; mainly focuses on tables; local pytest failed and requires Docker; Docker not tested; input is TeX-based, not ModelTables card text. | 2020 | Regular extraction |
| [MetaLead](https://researchtrend.ai/papers/2601.22420) | Yes | Prompt-based; extracts from PDF/HTML text, so table structure is lost; OpenAI issue was fixed locally, but on ModelTables it extracted too few nuggets and was incomplete. | 2026 | Prompt-based |
| [SciLead](https://aclanthology.org/2024.emnlp-main.453/) | Not implemented | Dataset could not be downloaded; table display is invalid. | 2024 | Prompt-based |
