# Query Sources

| title | link | note |
| --- | --- | --- |
| UniDocBench | [`query_example_unidocbench.py`](./query_example_unidocbench.py) | QA query synthesis template for scientific documents; we only keep a small filtered subset for model selection, and most generated queries are unrelated. |
| LitSearch | https://aclanthology.org/2024.emnlp-main.840/ | Closest explicit literature-search query benchmark; for our task we reuse the query text and replace `paper` / `studies` with `models`. |
| LitSearch query subset | https://huggingface.co/datasets/yale-nlp/LitSearch-NLP-Class/viewer/query?row=6 | Query-only subset we can reuse directly as recommendation-style query data. |
| SPRD / Scholarly Paper Recommendation Dataset | https://link.springer.com/article/10.1007/s00799-022-00339-w | Manual relevance judgments for scholarly paper recommendation; good for evaluation, not a natural-language query benchmark. |
| CiteULike | https://link.springer.com/article/10.1007/s10115-023-01901-x | User-item interaction data for scholarly papers; good for personalization, not a query benchmark. |
| RARD / Mr. DLib | https://mr-dlib.org/blog/2017/06/12/rard-the-related-article-recommendation-dataset/ | Real recommendation logs and click feedback from a related-article service; closer to recommendation behavior than query synthesis. |
| unarXive | https://link.springer.com/article/10.1007/s11192-020-03382-z | Full-text papers with citation contexts; useful raw material for citation-recommendation or query synthesis. |
| unarXive open subset | https://zenodo.org/records/7752615 | Open subset of unarXive with structured full text and citation network. |

## Download

Download the LitSearch query subset `query` and save it locally as JSONL:

```bash
python -c "from datasets import load_dataset; ds = load_dataset('yale-nlp/LitSearch-NLP-Class', split='query'); ds.to_json('data/query/litsearch_nlp_class_query.jsonl')"
```

## Analysis on groundtruth for extracting model from paper corpusid
Extract unique `corpusids` into a plain text file:

```bash
python -m src.query.extract_corpusids_to_txt \
  --input src/query/data/query/litsearch_query.jsonl \
  --output src/query/data/query/new_corpusids.txt
```

Test whether we could extract hf links from full text, if so we could infer model recommendation from paper recommendation
```bash
# go to z6dong@watgpu:~/shared_data/se_s2orc_250218, where we store se_s2orc_corpus data
PYTHONNOUSERSITE=1 python extract_corpus_hf_links.py   --ids_file new_corpusids.txt   --db_path paper_index_mini.db   --data_directory ./   --output_parquet corpus_hf_links.parquet --keep_full_text --full_text_dir fulltexts --limit 20
```
Analysis on corpus_hf_links.parquet
```bash
python3 -m src.query.stats_litsearch_corpus_links \
  --parquet corpus_hf_links.parquet \
  --query_jsonl src/query/data/query/litsearch_query.jsonl
```


## Query substitution
```bash

```