# Legacy Commands

## Build Index (Legacy)

```bash
bash src/modelsearch/base_densesearch.sh
```

## Compare Baselines

```bash
python -m src.modelsearch.compare_baselines \
  --model_id Salesforce/codet5-base \
  --relationship_parquet data/processed/modelcard_step3_dedup.parquet \
  --starmie_json results/table_search.json \
  --dense_neighbors output/modelsearch_neighbors.json \
  --output_md output/compare.md
```

