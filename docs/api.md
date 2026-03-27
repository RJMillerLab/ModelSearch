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
    Query2ModelCardSearch,
    Card2Tab2CardSearch,
    Query2Tab2CardSearch,
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
from src.search.ir_searcher import DenseSearcher

dense = DenseSearcher(emb_npz_path="data/card2card_embeddings.npz")
q2m = Query2ModelCardSearch(query="transformer model for NLP", top_k=20)
q2m.search_dense(top_k=20, dense=dense)
results = q2m.results.get("dense", [])
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
import duckdb
from src.config import MODELLAKE_DB_HUGGING  # or full DB per your resource set

con_data = duckdb.connect(MODELLAKE_DB_HUGGING, read_only=True)
table_ids = search_table2table(
    search_type="single_column",
    query="path_or_indexed_table_name.csv",
    k=10,
    con_data=con_data,
    augmentation_types=["ori"],
)
con_data.close()
```

## 5. Card to Tab to Card Search

```python
import duckdb
from src.utils import _paths_for_resource_set

_, _, db_path = _paths_for_resource_set(["hugging"])
con_data = duckdb.connect(db_path, read_only=True)
c2t2c = Card2Tab2CardSearch()
c2t2c.card2tab2card_pipeline(
    modelid_list=["Salesforce/codet5-base"],
    con_data=con_data,
    table_resources=["hugging"],
    search_type="keyword",
    table_top_k=10,
)
con_data.close()
full = c2t2c.get_full_map()  # card2tab_map, tab2tab_map, tab2card_map
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

