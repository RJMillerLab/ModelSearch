# Query2ModelCard Nugget Batch Experiment

This folder is for offline batch experiments. It is separate from the frontend / one-job wrap path so experimental runs do not overwrite existing UI or pipeline artifacts.

## Two Paths

Frontend / one-job path:

```bash
python -m src.evaluate.wrap_card_query_eval --jobs-json <jobs.json> --job-id <job_id> --llm-mode iter
```

Unified backend batch experiment path:

```bash
python src/batch_exp/run_batch_eval.py \
  --queries-file data_251117/query/query_rewrite_polished.jsonl
```

Reuse an existing query2modelcard intermediate:

```bash
python src/batch_exp/run_batch_eval.py --jobs-json <batch_preset_queries.json>
```

## Batch Stages

1. `query2modelcard`: either run through the backend from `--queries-file`, or loaded from `--jobs-json`.
2. `modelcard2nugget`: batch-generate missing model-card nugget CSVs, reusing existing `data_251117/card2nugget/*.csv`.
3. `query2nugget`: batch-map all queries to selected headers and optional constraints.
4. `filter`: for each query + method + top-k group, filter candidate nugget CSVs and record method scores/rankings.

## Recommended Run

Use the polished model recommendation query set here, not the old smoke / 15-example files:

```bash
python src/batch_exp/run_batch_eval.py \
  --queries-file data_251117/query/query_rewrite_polished.jsonl \
  --top-k 1 3 5 10 \
  --query-llm-mode batch \
  --card-llm-mode batch \
  --match-build structured
```

`run_batch_eval.py` will automatically request enough `query2modelcard` results to cover the largest requested `--top-k`, then reuse the deduped model-card list for batch `modelcard2nugget` extraction and per-query/per-top-k filtering.

Reuse an existing query2modelcard intermediate:

```bash
python src/batch_exp/run_batch_eval.py \
  --jobs-json jobs_251117/batch_runs/batch_preset_queries_20260501_154026.json \
  --top-k 1 3 5 10 \
  --query-llm-mode batch \
  --card-llm-mode batch \
  --match-build structured
```

## Full Run

```bash
python src/batch_exp/run_batch_eval.py \
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

## Distribution Plot

If you want a separate figure that emphasizes per-method nugget distributions, use:

```bash
python src/batch_exp/plot_distribution_eval.py \
  --summary-json src/batch_exp/fake_aggregate_summary.json \
  --per-query-jsonl src/batch_exp/fake_per_query_method_scores_int.jsonl \
  --output data_251117/statistic.png \
  --compare-top-k 10
```

This alternative figure is organized as:

- six method-wise histogram panels on the left, one panel per method, with a mean line;
- a right-side stacked bar chart showing top-1 through top-6 proportions by method;
- blue-toned histograms on the left and red-toned rank shares on the right;
- a separate `.png` output so it does not overwrite the current main figure.
