# Evaluate

## Content

1. [Model Card Extract Nuggets](#1-model-card-extract-nuggets)
2. [Get Query-Nuggets List Mapping](#2-get-query-nuggets-list-mapping)
3. [Evaluate](#3-evaluate)
4. [Reference](#4-reference)

## 1. Model Card Extract Nuggets

Input:

- `src.config.CARD_CONTENT_RAW`
- `../ModelTables/data/raw_251117/train-0000*-of-00006.parquet`
- columns: `modelId`, `card`

Single-card test:

```bash
python -m src.evaluate.card2nugget_extraction single \
  --model-id test-model \
  --card "paste one model card here"
```

Single output:

- `data_251117/evaluate/single_modelcard_nuggets.json`

Batch:

```bash
python -m src.evaluate.card2nugget_extraction batch
```

Batch outputs:

- `data_251117/evaluate/modelcard_nuggets.jsonl`
- `data_251117/evaluate/modelcard_nuggets.json`

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

## 4. Reference

| Name | Tested | Why Not Used / Issue | Year | Type |
| --- | --- | --- | --- | --- |
| [AxCell](https://aclanthology.org/2020.emnlp-main.692/) | Yes | Non-prompt extraction; mainly focuses on tables; local pytest failed and requires Docker; Docker not tested; input is TeX-based, not ModelTables card text. | 2020 | Regular extraction |
| [MetaLead](https://researchtrend.ai/papers/2601.22420) | Yes | Prompt-based; extracts from PDF/HTML text, so table structure is lost; OpenAI issue was fixed locally, but on ModelTables it extracted too few nuggets and was incomplete. | 2026 | Prompt-based |
| [SciLead](https://aclanthology.org/2024.emnlp-main.453/) | Not implemented | Dataset could not be downloaded; table display is invalid. | 2024 | Prompt-based |
