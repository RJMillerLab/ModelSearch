"""
Prompt templates for QA (Question Answering) based on integrated tables
"""

def get_qa_prompt(
    query: str,
    table_serialized: str,
    table_source: str = "Integrated Table"
) -> str:
    """
    Generate prompt for answering questions based on integrated table.
    
    Args:
        query: Original search query or question
        table_serialized: Serialized representation of the integrated table (CSV format or description)
        table_source: Description of where the table came from
    
    Returns:
        Formatted prompt string for LLM
    """
    prompt = f"""You are an expert data analyst helping users understand and analyze data from integrated tables.

## Task
Answer the user's question based on the provided integrated table data. Provide a clear, comprehensive, and data-driven answer.

## User Query/Question
{query}

## Integrated Table: {table_source}
{table_serialized}

## Instructions
1. **Analyze the table structure**: Understand the columns, data types, and relationships
2. **Answer the question**: Use the data in the table to provide a direct answer to the user's query
3. **Provide insights**: If relevant, include:
   - Key statistics or patterns found in the data
   - Notable observations or trends
   - Data quality considerations (if applicable)
   - Limitations or caveats (if any)
4. **Be specific**: Reference actual values, counts, or patterns from the table when possible
5. **Be concise**: Provide a clear and focused answer without unnecessary verbosity

## Output Format
Please provide your answer in the following JSON format:
{{
    "answer": "<your comprehensive answer to the user's question based on the table data>",
    "key_findings": [
        "<finding 1>",
        "<finding 2>",
        ...
    ],
    "data_summary": {{
        "total_rows": <number>,
        "key_columns": ["<column1>", "<column2>", ...],
        "notable_statistics": "<brief summary of key statistics or patterns>"
    }},
    "confidence": "<high/medium/low>",
    "limitations": [
        "<limitation 1 if any>",
        "<limitation 2 if any>",
        ...
    ]
}}

## Important Notes
- Base your answer ONLY on the data provided in the table
- If the table doesn't contain enough information to fully answer the question, state this clearly
- Use specific numbers and values from the table when possible
- If you notice data quality issues, mention them in limitations
- Be honest about what can and cannot be determined from the data

Please provide your answer now:"""

    return prompt

