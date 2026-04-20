# Evaluate

## Content

1. [Model Card Extract Nuggets](#1-model-card-extract-nuggets)
2. [Get Query-Nuggets List Mapping](#2-get-query-nuggets-list-mapping)
3. [Evaluate](#3-evaluate)
4. [Reference](#4-reference)

## 1. Model Card Extract Nuggets


Single-card test:

```bash
python -m src.evaluate.card2nugget_extraction --model-id google-bert/bert-base-uncased
```

Batch test (multiple model ids in one OpenAI Batch job):

```bash
python -m src.evaluate.card2nugget_extraction \
  --model-ids google/bert_uncased_L-12_H-768_A-12 baidu/ERNIE-4.5-VL-424B-A47B-Base-Paddle

# from file
# python -m src.evaluate.card2nugget_extraction --model-ids-file path/to/model_ids.txt
```

Outputs:

- `data_251117/evaluate/<model_id_with___>.csv` — parsed tuples table (columns: `Model`, `Base_model`, `Dataset`, `Train_dataset`, `Test_dataset`, `Model_hyperparameters`, `Model_variant_type`, `Metric`, `Metric_value`)
- `data_251117/evaluate/<model_id_with___>_meta.yaml` — metadata + full `prompt` + raw `response`
- Batch mode outputs are written to `data_251117/evaluate/batch/` (including per-model csv/meta and Batch API input/output jsonl files for debugging).

The LLM sees the **full model card text** from parquet (tables + prose mixed). **No local Hugging Face CSV tables** are appended. Optional env: `MODELSEARCHDEMO_CARD_MAX_CHARS` (default `100000`) truncates the card in the prompt; hparam regex fallback still uses the full card after citation stripping.

Optional: `python -m src.evaluate.card2nugget_extraction --read-csv` prints stats for a CSV path.

## 2. Get Query-Nuggets List Mapping

Input:

- `data_251117/evaluate/modelcard_nuggets.jsonl`

Run:

```bash
python -m src.evaluate.query2nugget_layer_mapping
```

Outputs:

- `data_251117/evaluate/query_nugget_mapping.json`
- `data_251117/evaluate/real_subtopic.qrels`
- `data_251117/evaluate/real_initial.run`

## 3. Evaluate

Toy data:

```bash
python -m src.evaluate.evaluate_pyndeval \
  --run src/evaluate/toy_data/toy_initial.run \
  --qrels src/evaluate/toy_data/toy_subtopic.qrels \
  --cutoff 20
```

Real data:

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
