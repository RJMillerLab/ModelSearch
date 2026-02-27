# Python API Usage

All search functions can be imported and used programmatically:

```python
from src.search import (
    build_card_index,
    search_card2card,
    search_table2table,
    search_query2modelcard,
    search_card2tab2card
)
```

## 1. Build Index (First Time Only)

### Option 1: Using data_citationlake raw data

```python
build_card_index(
    field="card",
    raw_dir="data_citationlake/raw",  # or "data/raw" for local data
    output_index="data/card2card.faiss"
)
```

### Option 2: Using processed parquet (faster)

```python
build_card_index(
    field="card_readme",
    parquet="data_citationlake/processed/modelcard_step1.parquet",
    output_index="data/card2card.faiss"
)
```

## 2. Query to ModelCard (Most Common)

```python
results = search_query2modelcard(
    query="transformer model for NLP",
    faiss_index="data/card2card.faiss",
    top_k=20
)
```

## 3. Card to Card Search

```python
neighbors = search_card2card(
    model_id="Salesforce/codet5-base",
    faiss_index="data/card2card.faiss",
    top_k=20
)
```

## 4. Table to Table Search (Testing)

```python
table_ids = search_table2table(
    query=["value1", "value2"],
    search_type="single_column",
    k=10
)
```

## 5. Card to Tab to Card Search

```python
similar_models = search_card2tab2card(
    model_id="Salesforce/codet5-base",
    schema_log_path="data_citationlake/logs/parquet_schema.log",
    query=["keyword1", "keyword2"],
    search_type="keyword",
    k=10
)
```

## Batch Evaluation (Optional)

For batch evaluation, you can use the batch search functions:

```python
from src.search import search_card2card_batch

# Search all models at once
all_neighbors = search_card2card_batch(
    emb_npz="data/card2card_embeddings.npz",
    faiss_index="data/card2card.faiss",
    top_k=20,
    output_json="data/card2card_neighbors.json"
)
```

