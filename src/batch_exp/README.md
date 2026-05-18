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

If you want a separate figure that emphasizes per-method nugget distributions, use the copied local batch outputs:

```bash
python src/batch_exp/plot_distribution_eval.py \
  --summary-json data_251117/evaluate/query2modelcard_nugget_batch/aggregate_summary.json \
  --per-query-jsonl data_251117/evaluate/query2modelcard_nugget_batch/per_query_method_scores.jsonl \
  --output data_251117/statistic.png \
  --compare-top-k 10
```

This alternative figure is organized as:

- two `top-k` blocks per row, each block split into a left distribution panel and a right rank-share panel;
- each `top-k` panel uses the corresponding `top_k` slice from `per_query_method_scores.jsonl`;
- blue-toned histograms on the left and red-toned rank shares on the right;
- a separate `.png` output so it does not overwrite the current main figure.

If `per_query_method_scores.jsonl` sits next to the summary JSON, `--per-query-jsonl` can be omitted and the plot script will resolve it automatically.

By default the script plots every `top-k` present in the summary. Pass one or more values to restrict the figure, for example `--compare-top-k 1 3 5 10`.

If the ranking changes across `top-k`, keep the per-`k` panels in the figure and describe the sensitivity explicitly in the writeup instead of collapsing everything to `top-10`.
