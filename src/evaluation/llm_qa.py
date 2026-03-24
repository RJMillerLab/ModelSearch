"""
QA entry: answer_question_with_llm. Core LLM helpers live in evaluation.llm.
"""
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from evaluation.llm import (
    call_llm_api,
    parse_llm_response_qa,
    serialize_table_for_prompt,
)


def get_qa_prompt(
    query: str,
    table_serialized: str,
    table_source: str = "Integrated Table",
    qa_mode: str = "card2tab2card",
    model_ids_to_rank: Optional[List[str]] = None,
    model_card_info: Optional[Dict] = None,
) -> str:
    """
    Generate prompt for model ID reranking based on integrated table cells and model card information.
    """
    model_ids_section = ""
    if model_ids_to_rank:
        model_ids_lines = "\n".join(f"- {model_id}" for model_id in model_ids_to_rank)
        model_ids_section = (
            f"\n## Model IDs to Rank\n"
            f"The following model IDs were found from the search results ({qa_mode.upper()} mode). "
            f"You MUST rank ALL of these model IDs:\n\n"
            f"{model_ids_lines}\n\n"
            f"**Important**: You must rank ALL {len(model_ids_to_rank)} model IDs listed above. "
            f"Do not add or remove any model IDs from this list.\n"
        )

    if qa_mode == "query2modelcard":
        mode_instructions = """
## Analysis Mode: Card2Card (Model Card Based)
- **Primary Source**: Model card information (text descriptions, metadata, specifications)
- **Secondary Source**: Integrated table (if available, may be empty or minimal)
- **Focus**: Analyze model cards to extract capabilities, performance, and suitability
- **Evidence**: Extract supporting evidence from model card text and any available table cells
- **Context Limitation**: Model cards may have context length limits - extract the most relevant information
"""
        table_note = (
            "Note: The integrated table below may be empty or minimal. "
            "Focus on model card information if table data is limited."
        )
    else:
        mode_instructions = """
## Analysis Mode: Card2Tab2Card (Table + Model Card Based)
- **Primary Sources**:
  - Integrated table (contains model card data extracted into structured format)
  - Model card information (original text and metadata)
- **Focus**: Analyze both table cells AND model card information
- **Evidence**: Extract supporting evidence from:
  - Specific cell values in the integrated table
  - Original model card text and descriptions
- **Context Limitation**: Model cards may have context length limits - prioritize key information
"""
        table_note = (
            "The integrated table below contains structured data extracted from model cards. "
            "Use this along with model card information."
        )

    prompt = f"""You are an expert AI/ML consultant specializing in model selection and reranking. Your task is to rerank model ID candidates based on the user's query requirements, using available data sources.
{mode_instructions}

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

## User Query/Requirements
{query}
{model_ids_section}
{table_note}

## Integrated Table: {table_source}
{table_serialized}

## Instructions
1. **Extract Information for Each Model ID**:
   - **From Integrated Table** (if available): Find rows containing the model ID, extract cell values
   - **From Model Card**: Extract key information (considering context length limits):
     * Performance metrics and benchmarks
     * Model capabilities and supported tasks
     * Architecture and specifications
     * Use cases and applications
     * Limitations and constraints

2. **Identify Supporting Evidence**: For each piece of information, mark its source:
   - **Table Cell**: "From table cell [column_name]: [value]" or "Table shows [metric] = [value]"
   - **Model Card**: "Model card states [claim]" or "According to model card: [description]"
   - Be specific: cite exact values, quotes, or cell locations when possible

3. **Evaluate Against Query**: For each model ID, assess:
   - Relevance to query requirements
   - Quality of supporting evidence
   - Strength of match (strong/medium/weak)

4. **Rerank and Summarize**:
   - Sort model IDs from best to worst match
   - Provide a global summary comparing top models
   - Highlight trade-offs and complementary strengths

## Output Format (JSON Only)
Return **ONLY** valid JSON with the following structure:
{{
  "ordered_model_ids": [
    "model_id_1",
    "model_id_2",
    "model_id_3"
  ],
  "analysis": {{
    "model_id_1": {{
      "score": 0-100,
      "summary": "Short summary of why this model is ranked here",
      "evidence": {{
        "table": [
          "Evidence sentence 1 from table cells",
          "Evidence sentence 2 from table cells"
        ],
        "model_card": [
          "Evidence sentence 1 from model card",
          "Evidence sentence 2 from model card"
        ]
      }},
      "strength_of_match": "strong|medium|weak",
      "key_metrics": [
        "Key metric or capability 1",
        "Key metric or capability 2"
      ],
      "limitations": [
        "Limitation 1",
        "Limitation 2"
      ]
    }}
  }},
  "global_summary": {{
    "best_overall": "model_id_x",
    "notes": "Summary of key trade-offs, complementary strengths, and recommendations"
  }}
}}

Do not include any extra text outside of the JSON. The response **must** be valid JSON.
"""
    return prompt


def answer_question_with_llm(
    query: str,
    table: pd.DataFrame,
    table_source: str = "Integrated Table",
    qa_mode: str = "card2tab2card",
    model_ids_to_rank: Optional[List[str]] = None,
    search_results_data: Optional[Dict] = None,
    model: str = "gpt-4",
) -> Dict[str, Any]:
    """
    Answer a question based on an integrated table using LLM.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set OPENAI_API_KEY in your environment or use fake response mode."
        )

    table_serialized = (
        serialize_table_for_prompt(
            table,
            max_rows=100,
            max_cols=20,
            data_label="Data:",
            include_extra_cols=True,
        )
        if not table.empty
        else "No integrated table available. Analysis will be based on model card information only."
    )
    model_card_info: Dict[str, Any] = {}
    if search_results_data:
        model_card_info = {}

    prompt = get_qa_prompt(
        query=query,
        table_serialized=table_serialized,
        table_source=table_source,
        qa_mode=qa_mode,
        model_ids_to_rank=model_ids_to_rank,
        model_card_info=model_card_info,
    )

    response_text = call_llm_api(
        prompt,
        model=model,
        api_key=api_key,
        system_message="You are an expert data analyst. Provide answers in valid JSON format.",
    )
    answer = parse_llm_response_qa(response_text)
    return {
        "success": True,
        "answer": answer,
        "source": "llm",
        "model": model,
    }
