# Python API Usage

Data paths (raw dir, processed dirs, modellake.db, relationship parquet) are centralized in `src.config`; use `from src.config import RAW_DIR, MODELLAKE_DB, RELATIONSHIP_PARQUET` etc. in your code.

All search functions can be imported and used programmatically:

```python
from src.search import (
    build_card_index,
    search_dense_neighbors_queries,
    search_sparse_neighbors_queries,
    search_hybrid_neighbors_queries,
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
    raw_dir="...",  # use src.config.RAW_DIR (default ../ModelTables/data/raw)
    output_index="data/card2card.faiss"
)
```

### Option 2: Using processed parquet (faster)

```python
build_card_index(
    field="card_readme",
    parquet="...",  # or src.config.RELATIONSHIP_PARQUET for model–table parquet
    output_index="data/card2card.faiss"
)
```

## 2. Query to ModelCard (Most Common)

```python
results = search_query2modelcard(
    query="transformer model for NLP",
    top_k=20
)
```

## 3. Card to Card Search

```python
neighbors_map = search_dense_neighbors_queries(
    query_model_ids=["Salesforce/codet5-base"],
    emb_npz="data/card2card_embeddings.npz",
    top_k=20,
)
neighbors = neighbors_map["Salesforce/codet5-base"]
 
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
    schema_log_path="../ModelTables/logs/parquet_schema.log",
    query=["keyword1", "keyword2"],
    search_type="keyword",
    k=10
)
```

## Batch Evaluation (Optional)

For batch evaluation, you can use the batch search functions:

```python
neighbors_map = search_hybrid_neighbors_queries(
    query_model_ids=["Salesforce/codet5-base", "gpt2"],
    emb_npz="data/card2card_embeddings.npz",
    sparse_index_path="data/card2card_sparse_index",
    top_k=20,
)
```

