---
title: ModelSearch Demo
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# ModelSearch Demo

A comprehensive model search system that combines multiple retrieval methods, table integration, and LLM-based evaluation/QA.

## Deploy on Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), choose **Docker** as SDK.
2. Push this repo (or copy `Dockerfile` and ensure `README.md` has the YAML block above with `sdk: docker`, `app_port: 7860`).
3. The container runs the backend with `SERVE_UI=1` so the demo UI and API are served on one port (7860).
4. **Data**: The pipeline expects `data/` (e.g. `card2card_embeddings.npz`, `card2card.faiss`, `modellake.db`). For a minimal demo you can add a small dataset to the Space or use [persistent storage](https://huggingface.co/docs/hub/spaces-storage) and upload files at runtime.

## Implementation Overview

### Search Components

**Card2Card Retrieval (Model-to-Model Search)**
- **Dense Retrieval**: FAISS + sentence-transformers (all-MiniLM-L6-v2)
- **Sparse Retrieval**: rank-bm25 (BM25 algorithm)
- **Hybrid Retrieval**: RRF (Reciprocal Rank Fusion) combining sparse + dense

**Card2Tab2Card Retrieval (Model-to-Table-to-Model Search)**
- **Table Search**: Blend_internal (table-to-table search engine)
- **Table Types**: keyword, single_column, multi_column, unionable

### Table Integration

- **Methods**: Union, Intersection, Outer Join
- **Implementation**: pandas-based integration with column alignment

### LLM-Based Evaluation & QA

**Evaluation Module** (`src/evaluation/`)
- **LLM**: OpenAI API (via internal wrapper)
- **Retry**: tenacity for API retry handling
- **Prompt**: Quality comparison between two integrated tables

<details>
<summary>Evaluation Prompt</summary>

```
You are an expert data analyst evaluating the diversity of two integrated tables from a model search system.

## Task
Compare two integrated tables and provide a diversity score (0-100) and detailed analysis.

## Original Query
{query}

## Table 1: {table1_source}
{table1_serialized}

## Table 2: {table2_source}
{table2_serialized}

## Evaluation Criteria
Please compare the QUALITY of results from these two search methods:
1. **Data Quality**: Completeness, consistency, structure
2. **Relevance**: How well results match the query
3. **Coverage**: Breadth and depth of information
4. **Usefulness**: Practical value for the user's needs

## Output Format
Please provide your evaluation in the following JSON format:
{
    "comparison_score": {
        "table_search_quality": <number between 0-100>,
        "model_search_quality": <number between 0-100>,
        "overall_difference": <difference in scores>,
        "winner": "<table_search or model_search>"
    },
    "quality_analysis": {
        "table_search": {
            "score": <number between 0-100>,
            "strengths": ["<strength 1>", "<strength 2>", ...],
            "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
            "summary": "<brief summary of table search quality>"
        },
        "model_search": {
            "score": <number between 0-100>,
            "strengths": ["<strength 1>", "<strength 2>", ...],
            "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
            "summary": "<brief summary of model search quality>"
        }
    },
    "key_differences": [
        "<difference 1>",
        "<difference 2>",
        ...
    ],
    "recommendation": "<recommendation on which method is better and why>",
    "comparison_summary": "<brief overall comparison summary>"
}
```

</details>

**QA Module** (`src/qa/`)
- **LLM**: OpenAI API (via internal wrapper)
- **Retry**: tenacity for API retry handling
- **Prompt**: Model ID reranking based on integrated table cells and model card information

<details>
<summary>QA Prompt</summary>

```
You are an expert AI/ML consultant specializing in model selection and reranking. Your task is to rerank model ID candidates based on the user's query requirements, using available data sources.

## Analysis Mode: Card2Tab2Card (Table + Model Card Based)
- **Primary Sources**: 
  - Integrated table (contains model card data extracted into structured format)
  - Model card information (original text and metadata)
- **Focus**: Analyze both table cells AND model card information
- **Evidence**: Extract supporting evidence from:
  - Specific cell values in the integrated table
  - Original model card text and descriptions
- **Context Limitation**: Model cards may have context length limits - prioritize key information

## Task
Perform model ID reranking by:
1. **For each model ID in the list above**:
   - Search the integrated table (if available) to find rows/cells containing this model ID
   - Extract model card information (capabilities, performance metrics, specifications, use cases)
   - Identify supporting evidence from:
     * **Table cells**: Specific values, metrics, descriptions from the integrated table
     * **Model card text**: Original descriptions, capabilities, performance claims
   
2. **Extract Supporting Evidence**: For each model ID, extract:
   - **From integrated table cells**: Actual values, metrics, specifications (cite specific cells)
   - **From model card**: Key capabilities, performance claims, use case descriptions
   - **Mark the source**: Clearly indicate whether evidence comes from table cells or model card text
   
3. **Rerank Model IDs**: Sort the provided model IDs based on:
   - How well each model matches the query requirements
   - Evidence from integrated table cells (if available)
   - Evidence from model card information
   - Quality and relevance of supporting evidence
   
4. **Provide Analysis**: For each model ID, provide:
   - Suitability score (0-100)
   - Detailed analysis explaining the ranking
   - Supporting evidence with source attribution (table cell or model card)
   - Key metrics and capabilities relevant to the query

## Output Format
{
    "answer": "<comprehensive answer explaining model ID reranking>",
    "model_ranking": [
        {
            "rank": 1,
            "model_id": "<actual model ID>",
            "suitability_score": <number between 0-100>,
            "analysis": "<detailed analysis>",
            "supporting_evidence": [
                {
                    "claim": "<specific claim>",
                    "source": "<'table_cell' or 'model_card'>",
                    "evidence": "<exact value or description>",
                    "relevance": "<how this evidence supports the ranking>"
                }
            ],
            "key_metrics": {"<metric_name>": "<value>"},
            "strengths": ["<strength 1>", ...],
            "limitations": ["<limitation 1>", ...],
            "use_case": "<best use case>"
        }
    ],
    "summary": {
        "total_models_analyzed": <number>,
        "top_recommendations": ["<model_id1>", ...],
        "key_criteria_used": ["<criterion 1>", ...],
        "evidence_sources": {
            "table_cells_used": <true/false>,
            "model_cards_used": <true/false>
        }
    }
}
```

</details>

### Frontend & Backend

- **Backend**: Flask REST API (`src/demo/backend.py`)
- **Frontend**: Flask web interface (`src/demo/frontend.py`)
- **CORS**: flask-cors for cross-origin requests

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Special installation (FAISS, Pyserini)

- **FAISS** (dense similarity search for card2card / query2modelcard): not in `requirements.txt`; install separately.  
  - CPU: `pip install faiss-cpu`  
  - GPU: `pip install faiss-gpu`  
  - Docs: [faiss](https://github.com/facebookresearch/faiss)

- **Pyserini** (sparse/hybrid table retrieval baselines, baseline2/3): requires **Java/JDK**.  
  - Install: `pip install pyserini` and e.g. `conda install -c conda-forge openjdk`  
  - Docs: [Pyserini](https://github.com/castorini/pyserini)

### 3. Clone Blend_internal (Required for Table Search)

```bash
git clone git@github.com:DoraDong-2023/Blend_internal.git src/Blend_internal
```

### 4. Set Up Environment Variables

Create `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

## Quick Start

### Build Index

```bash
python -m src.search.card2card build-index \
  --field card \
  --raw_dir data_citationlake/raw \
  --output_npz data/card2card_embeddings.npz \
  --output_index data/card2card.faiss
```

### Run Demo

**Backend:**
```bash
python -m src.demo.backend
```

**Frontend:**
```bash
python -m src.demo.frontend
```

Access the web interface at `http://localhost:5001`

### Batch-run preset template queries (simulate frontend clicks)

With the backend running (default `http://localhost:5002`), you can batch-run all preset queries from `config/preset_queries.json` — this simulates clicking “Start Search” (and integration, if enabled) for each template query:

```bash
# while running backend
python -m src.demo.backend --port 5002
# batch run preset queries
python scripts/batch_run_preset_queries.py --backend_url http://localhost:5002 --preset_path config/preset_queries.json --run_integration 
```

## Key libraries & acknowledgments

We use the following libraries and borrowed components (with install needs):

| Component | Role | Link |
|-----------|------|------|
| **FAISS** | Dense vector similarity search | [faiss](https://github.com/facebookresearch/faiss) |
| **rank-bm25** | Sparse retrieval (BM25) for card2card | PyPI: `rank-bm25` |
| **sentence-transformers** | Text embeddings | [sentence-transformers](https://github.com/UKPLab/sentence-transformers) |
| **Pyserini** | BM25 (Lucene) + hybrid for **table retrieval** baselines (baseline2/3); requires **Java/JDK** | [Pyserini](https://github.com/castorini/pyserini) |
| **ModelTables** | Data pipeline, baseline2/3 scripts (get_metadata, sparse_search, search_with_pyserini_hybrid), Starmie-style evaluation | [ModelTables](https://github.com/RJMillerLab/ModelTables) |
| **Blend** | Original table-to-table search design | [Blend](https://github.com/LUH-DBS/Blend) |
| **Blend_internal** | Blend reimplementation adapted for ModelTables; used for card2tab2card & tab2tab | [Blend_internal](https://github.com/DoraDong-2023/Blend_internal) |
| **OpenAI API** | LLM calls for evaluation/QA | openai |
| **Flask** | Web framework for demo | flask |

**Install:** Core deps: `pip install -r requirements.txt`. Table search: clone/symlink [Blend_internal](https://github.com/DoraDong-2023/Blend_internal) (see Installation above). Baseline2/3: install **pyserini** and **Java** in the same env (e.g. `conda install -c conda-forge openjdk`). One env (e.g. faiss_gpu_env + pyserini + Java) can run dense, sparse, and hybrid; see `docs/build_index.md`.
