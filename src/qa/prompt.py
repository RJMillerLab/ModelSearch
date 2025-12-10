"""
Prompt templates for QA (Question Answering) based on integrated tables
Focuses on model ID reranking based on integrated table cells and model card information
"""
from typing import Optional, List, Dict

def get_qa_prompt(
    query: str,
    table_serialized: str,
    table_source: str = "Integrated Table",
    qa_mode: str = "card2tab2card",  # "card2card" or "card2tab2card"
    model_ids_to_rank: Optional[List[str]] = None,
    model_card_info: Optional[Dict] = None
) -> str:
    """
    Generate prompt for model ID reranking based on integrated table cells and model card information.
    
    Args:
        query: Original search query or question (user's requirements)
        table_serialized: Serialized representation of the integrated table (CSV format with actual cell data)
        table_source: Description of where the table came from
        qa_mode: "card2card" (model card only, table optional) or "card2tab2card" (table + model card)
        model_ids_to_rank: List of model IDs from search results that need to be ranked
        model_card_info: Optional dictionary with model card information
    
    Returns:
        Formatted prompt string for LLM
    """
    model_ids_section = ""
    if model_ids_to_rank and len(model_ids_to_rank) > 0:
        model_ids_section = f"""
## Model IDs to Rank
The following model IDs were found from the search results ({qa_mode.upper()} mode). You MUST rank ALL of these model IDs:

{chr(10).join([f"- {model_id}" for model_id in model_ids_to_rank])}

**Important**: You must rank ALL {len(model_ids_to_rank)} model IDs listed above. Do not add or remove any model IDs from this list.
"""
    
    # Mode-specific instructions
    if qa_mode == "card2card":
        mode_instructions = """
## Analysis Mode: Card2Card (Model Card Based)
- **Primary Source**: Model card information (text descriptions, metadata, specifications)
- **Secondary Source**: Integrated table (if available, may be empty or minimal)
- **Focus**: Analyze model cards to extract capabilities, performance, and suitability
- **Evidence**: Extract supporting evidence from model card text and any available table cells
- **Context Limitation**: Model cards may have context length limits - extract the most relevant information
"""
        table_note = "Note: The integrated table below may be empty or minimal. Focus on model card information if table data is limited."
    else:  # card2tab2card
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
        table_note = "The integrated table below contains structured data extracted from model cards. Use this along with model card information."
    
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
   - Key factors that make it suitable or unsuitable

4. **Rerank with Scores**: Assign suitability scores (0-100) and rank all model IDs

5. **Provide Comprehensive Analysis**: Include:
   - Overall answer explaining the reranking
   - For each model: score, analysis, and supporting evidence with sources
   - Clear attribution of evidence (table vs. model card)

## Output Format
Please provide your reranking results in the following JSON format:
{{
    "answer": "<comprehensive answer explaining model ID reranking. Reference specific evidence sources (table cells or model card text).>",
    "model_ranking": [
        {{
            "rank": 1,
            "model_id": "<actual model ID, e.g., 'Salesforce/codet5-base'>",
            "model_name": "<model name if available>",
            "suitability_score": <number between 0-100>,
            "analysis": "<detailed analysis explaining why this model is ranked at this position, referencing evidence sources>",
            "supporting_evidence": [
                {{
                    "claim": "<specific claim or fact about the model>",
                    "source": "<'table_cell' or 'model_card'>",
                    "evidence": "<exact value, quote, or description from the source>",
                    "relevance": "<how this evidence supports the ranking>"
                }},
                ...
            ],
            "key_metrics": {{
                "<metric_name>": "<value>",
                ...
            }},
            "strengths": [
                "<strength 1 with source attribution>",
                "<strength 2>",
                ...
            ],
            "limitations": [
                "<limitation 1 with source attribution if available>",
                ...
            ],
            "use_case": "<best use case based on query and evidence>"
        }},
        {{
            "rank": 2,
            ...
        }},
        ...
    ],
    "summary": {{
        "total_models_analyzed": <number of model IDs ranked>,
        "top_recommendations": ["<model_id1>", "<model_id2>", ...],
        "key_criteria_used": [
            "<criterion 1>",
            "<criterion 2>",
            ...
        ],
        "evidence_sources": {{
            "table_cells_used": <true/false>,
            "model_cards_used": <true/false>,
            "data_quality": "<assessment of data completeness>"
        }}
    }},
    "confidence": "<high/medium/low>",
    "limitations": [
        "<limitation 1 if any>",
        "<limitation 2 if any>",
        ...
    ]
}}

## Critical Requirements
- **MUST rank ALL model IDs from the list above** - do not add or remove any model IDs
- **MUST extract supporting evidence** from both table cells (if available) and model card information
- **MUST mark evidence sources**: Clearly indicate whether evidence comes from "table_cell" or "model_card"
- **MUST cite specific values**: When referencing table cells, cite exact values. When referencing model cards, cite specific claims or descriptions
- **MUST consider context limits**: Model cards may have length limits - extract the most relevant information
- **For Card2Card mode**: If table is empty, focus on model card information
- **For Card2Tab2Card mode**: Use both table cells AND model card information
- **Be specific**: Include exact values, quotes, or cell references in supporting evidence
- If information is missing, note this in limitations but still rank the model

## Example
If the table has a row with:
- model_id: "Salesforce/codet5-base"
- accuracy: "0.92"
- task: "code generation"
- parameters: "220M"

And the query is "code generation model", you should:
- Extract model_id "Salesforce/codet5-base"
- Note the accuracy of 0.92 from the cell
- Note the task matches "code generation"
- Rank it highly and explain using these actual cell values

Please provide your model ID reranking and answer now:"""

    return prompt

