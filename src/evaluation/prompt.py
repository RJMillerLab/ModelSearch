"""
Prompt templates for LLM-based evaluation
"""

def get_diversity_evaluation_prompt(
    query: str,
    table1_serialized: str,
    table2_serialized: str,
    table1_source: str = "Table Search Integration",
    table2_source: str = "Model Search Integration"
) -> str:
    """
    Generate prompt for evaluating diversity between two integrated tables.
    
    Args:
        query: Original search query
        table1_serialized: Serialized representation of first table (CSV format or description)
        table2_serialized: Serialized representation of second table (CSV format or description)
        table1_source: Description of where table1 came from
        table2_source: Description of where table2 came from
    
    Returns:
        Formatted prompt string for LLM
    """
    prompt = f"""You are an expert data analyst evaluating the diversity of two integrated tables from a model search system.

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
{{
    "comparison_score": {{
        "table_search_quality": <number between 0-100>,
        "model_search_quality": <number between 0-100>,
        "overall_difference": <difference in scores>,
        "winner": "<table_search or model_search>"
    }},
    "quality_analysis": {{
        "table_search": {{
            "score": <number between 0-100>,
            "strengths": ["<strength 1>", "<strength 2>", ...],
            "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
            "summary": "<brief summary of table search quality>"
        }},
        "model_search": {{
            "score": <number between 0-100>,
            "strengths": ["<strength 1>", "<strength 2>", ...],
            "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
            "summary": "<brief summary of model search quality>"
        }}
    }},
    "key_differences": [
        "<difference 1>",
        "<difference 2>",
        ...
    ],
    "recommendation": "<recommendation on which method is better and why>",
    "comparison_summary": "<brief overall comparison summary>"
}}

## Instructions
- Compare the QUALITY of results from both search methods
- Assign quality scores (0-100) to each method
- Identify which method performs better and by how much
- List specific strengths and weaknesses of each method
- Provide actionable recommendation on which to use
- Focus on data quality, relevance, and usefulness, not just diversity

Please provide your evaluation now:"""

    return prompt

