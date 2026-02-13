"""
Prompt templates for LLM-based evaluation of integrated table quality (Table Search vs Model Search).
Metrics: Relevance, Coverage, Diversity (IR-backed); total score = composition of sub-scores.
"""


def get_diversity_evaluation_prompt(
    query: str,
    table1_serialized: str,
    table2_serialized: str,
    table1_source: str = "Table Search Integration",
    table2_source: str = "Model Search Integration"
) -> str:
    """
    Generate prompt for quality comparison of two integrated tables given the user question.
    Output: total_quality_score, sub_scores (relevance, coverage, diversity with evidence),
    quality_analysis (strengths/weaknesses only), key_differences, evidence_for_differences.
    """
    prompt = f"""You are an expert data analyst evaluating the QUALITY of two integrated tables from a model search system, given the user's question.

## User question
{query}

## Table 1: {table1_source}
{table1_serialized}

## Table 2: {table2_source}
{table2_serialized}

## Task
Compare the quality of the two integrated tables with respect to the user question. Use information-retrieval style metrics so conclusions are evidence-based.

## Metrics (score each 0–100 per table; provide brief evidence)

1. **Relevance**: How well does the table content match the user question? (IR: relevance of retrieved items to query.)
2. **Coverage**: Breadth of information (distinct models, tasks, or aspects covered). (IR: recall-like.)
3. **Diversity**: Variety and non-redundancy of content. (IR: diversity / novelty.)

For each metric, give table_search and model_search scores and a short evidence sentence (e.g. counts, column names, examples).

## Output format (strict JSON only)
{{
    "total_quality_score": {{
        "table_search": <0-100>,
        "model_search": <0-100>,
        "winner": "<table_search or model_search>"
    }},
    "sub_scores": [
        {{
            "name": "Relevance",
            "table_search": <0-100>,
            "model_search": <0-100>,
            "evidence": "<one short sentence with numbers or examples>"
        }},
        {{
            "name": "Coverage",
            "table_search": <0-100>,
            "model_search": <0-100>,
            "evidence": "<one short sentence>"
        }},
        {{
            "name": "Diversity",
            "table_search": <0-100>,
            "model_search": <0-100>,
            "evidence": "<one short sentence>"
        }}
    ],
    "quality_analysis": {{
        "table_search": {{
            "strengths": ["<strength 1>", "<strength 2>"],
            "weaknesses": ["<weakness 1>", "<weakness 2>"]
        }},
        "model_search": {{
            "strengths": ["<strength 1>", "<strength 2>"],
            "weaknesses": ["<weakness 1>", "<weakness 2>"]
        }}
    }},
    "key_differences": [
        "<difference 1>",
        "<difference 2>"
    ],
    "evidence_for_differences": "<Short paragraph or bullets with evidence (counts, column names, examples) that support the key differences.>"
}}

## Instructions
- Score each metric 0–100 for both tables. Total quality can be the average of the three sub-scores or your overall judgment; keep total_quality_score.table_search and total_quality_score.model_search consistent with sub_scores.
- Provide brief, concrete evidence for each sub_score and for evidence_for_differences.
- In quality_analysis give only strengths and weaknesses (no long summary).
- Do not include recommendation, comparison_summary, or quality_difference.

Output valid JSON only, no markdown."""

    return prompt
