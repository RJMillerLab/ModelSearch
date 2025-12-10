# ModelSearch Demo

A comprehensive model search system that combines multiple retrieval methods, table integration, and LLM-based evaluation/QA.

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
- **LLM**: OpenAI API (via CitationLake wrapper)
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
- **LLM**: OpenAI API (via CitationLake wrapper)
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

### 2. Clone Blend_internal (Required for Table Search)

```bash
git clone git@github.com:DoraDong-2023/Blend_internal.git src/Blend_internal
```

### 3. Set Up Environment Variables

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

## Key Libraries

- **FAISS**: Dense vector similarity search
- **rank-bm25**: Sparse retrieval (BM25)
- **sentence-transformers**: Text embeddings
- **Blend_internal**: Table-to-table search
- **OpenAI API**: LLM calls for evaluation/QA
- **tenacity**: API retry handling
- **Flask**: Web framework
