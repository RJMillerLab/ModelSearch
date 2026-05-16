# Query2ModelCard Nugget Batch Experiment

This folder is for offline batch experiments. It is separate from the frontend / one-job wrap path so experimental runs do not overwrite existing UI or pipeline artifacts.

## Two Paths

Frontend / one-job path:

```bash
python -m src.evaluate.wrap_card_query_eval --jobs-json <jobs.json> --job-id <job_id> --llm-mode iter
```

Unified backend batch experiment path:

```bash
python experiments/query2modelcard_nugget_batch/run_batch_eval.py \
  --queries-file data_251117/query/query_rewrite_polished.jsonl
```

Reuse an existing query2modelcard intermediate:

```bash
python experiments/query2modelcard_nugget_batch/run_batch_eval.py --jobs-json <batch_preset_queries.json>
```

## Batch Stages

1. `query2modelcard`: either run through the backend from `--queries-file`, or loaded from `--jobs-json`.
2. `modelcard2nugget`: batch-generate missing model-card nugget CSVs, reusing existing `data_251117/card2nugget/*.csv`.
3. `query2nugget`: batch-map all queries to selected headers and optional constraints.
4. `filter`: for each query + method + top-k group, filter candidate nugget CSVs and record method scores/rankings.

## Smoke Run

Backend mode, first 5 queries:

```bash
python experiments/query2modelcard_nugget_batch/run_batch_eval.py \
  --queries-file data_251117/query/query_rewrite_polished.jsonl \
  --limit-jobs 5 \
  --top-k 1 3 \
  --query-llm-mode batch \
  --card-llm-mode batch \
  --match-build structured
```

Reuse mode, first saved job:

```bash
python experiments/query2modelcard_nugget_batch/run_batch_eval.py \
  --jobs-json jobs_251117/batch_runs/batch_preset_queries_20260501_154026.json \
  --limit-jobs 1 \
  --top-k 1 3 \
  --query-llm-mode batch \
  --card-llm-mode batch \
  --match-build structured
```

## Full Run

```bash
python experiments/query2modelcard_nugget_batch/run_batch_eval.py \
  --queries-file data_251117/query/query_rewrite_polished.jsonl \
  --top-k 1 3 5 10 \
  --query-llm-mode batch \
  --card-llm-mode batch \
  --match-build structured
```

Outputs are written under:

```text
data_251117/evaluate/query2modelcard_nugget_batch/<run_id>/
```

In backend mode the saved intermediate is:

```text
data_251117/evaluate/query2modelcard_nugget_batch/<run_id>/query2modelcard_backend_intermediate.json
```

## Plot

```bash
python experiments/query2modelcard_nugget_batch/plot_batch_eval.py \
  --summary-json data_251117/evaluate/query2modelcard_nugget_batch/<run_id>/aggregate_summary.json \
  --output data_251117/evaluate/query2modelcard_nugget_batch/<run_id>/nugget_eval_figure.png \
  --compare-top-k 10
```

The plot contains:

- mean nugget score by method and top-k;
- query-wise win/tie/loss for table-search methods against the best semantic baseline;
- one `.png` and one `.pdf` output.
