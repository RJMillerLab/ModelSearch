"""
Unified LLM integration for evaluation and QA.
Uses llm_citationlake; fix that import if it fails.
"""
import os
import json
import re
from typing import Dict, Any, Optional, List
import pandas as pd

from .llm_citationlake import LLM_response, setup_openai


def serialize_table_for_prompt(
    df: pd.DataFrame,
    max_rows: int = 50,
    max_cols: int = 10,
    data_label: str = "Sample Data (first few rows):",
    include_extra_cols: bool = False,
) -> str:
    """
    Serialize DataFrame to a string format suitable for LLM prompt.

    Args:
        df: DataFrame to serialize
        max_rows: Maximum number of rows to include
        max_cols: Maximum number of columns to include
        data_label: Label before the table (e.g. "Sample Data (first few rows):" or "Data:")
        include_extra_cols: If True, append "... and N more columns" when truncated

    Returns:
        Serialized table as string
    """
    if df is None or df.empty:
        return "Empty table"

    cols_to_show = list(df.columns[:max_cols])
    df_subset = df[cols_to_show].head(max_rows)

    lines = [f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"]
    lines.append(f"Columns: {', '.join(df.columns.tolist())}")
    lines.append("")
    lines.append(data_label)
    lines.append(df_subset.to_csv(index=False))

    if df.shape[0] > max_rows:
        lines.append(f"\n... and {df.shape[0] - max_rows} more rows")
    if include_extra_cols and df.shape[1] > max_cols:
        lines.append(f"\n... and {df.shape[1] - max_cols} more columns")

    return "\n".join(lines)


def call_llm_api(
    prompt: str,
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    system_message: str = "You are an expert data analyst. Provide responses in valid JSON format.",
) -> str:
    """
    Call LLM API via llm_citationlake.

    Args:
        prompt: The prompt to send
        model: Model name (default: gpt-4)
        api_key: API key (if None, uses OPENAI_API_KEY from environment)
        system_message: System message for the chat

    Returns:
        LLM response text
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    if model.startswith("gpt-4") or model == "gpt-4":
        llm_model = "gpt-4-turbo"
    elif model.startswith("gpt-3.5") or model == "gpt-3.5-turbo":
        llm_model = "gpt-3.5-turbo-0125"
    else:
        llm_model = "gpt-3.5-turbo-0125"

    full_prompt = f"{system_message}\n\n{prompt}"
    setup_openai("", mode="openai")
    response, _ = LLM_response(
        chat_prompt=full_prompt,
        llm_model=llm_model,
        history=[],
        kwargs={"temperature": 0.3},
        max_tokens=2000,
    )
    return response


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM response for evaluation: extract JSON and return eval-style dict.
    """
    text = response_text.strip()
    candidates = []
    if text.startswith("{"):
        candidates.append(text)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        candidates.append(m.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for s in candidates:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            continue
    return {
        "error": "Failed to parse LLM response",
        "raw_response": response_text[:500],
    }


def parse_llm_response_qa(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM response for QA: extract JSON and return answer-style dict.
    """
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return {
                "answer": response_text,
                "model_ranking": [],
                "summary": {},
                "confidence": "medium",
                "limitations": ["Could not parse structured response from LLM"],
            }
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {
            "answer": response_text,
            "model_ranking": [],
            "summary": {},
            "confidence": "medium",
            "limitations": ["JSON parsing error"],
        }


def evaluate_diversity_with_llm(
    query: str,
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table1_source: str = "Table Search Integration",
    table2_source: str = "Model Search Integration",
) -> Dict[str, Any]:
    """
    Evaluate diversity between two tables using LLM.
    """
    table1_str = serialize_table_for_prompt(table1)
    table2_str = serialize_table_for_prompt(table2)
    from .prompt import get_diversity_evaluation_prompt

    prompt = get_diversity_evaluation_prompt(
        query=query,
        table1_serialized=table1_str,
        table2_serialized=table2_str,
        table1_source=table1_source,
        table2_source=table2_source,
    )
    response_text = call_llm_api(
        prompt,
        system_message="You are an expert data analyst. Provide evaluations in valid JSON format.",
    )
    result = parse_llm_response(response_text)
    result["success"] = True
    return result



if __name__ == "__main__":
    table1 = pd.DataFrame({"model_id": ["a", "b"], "steps": [500000, 300000]})
    table2 = pd.DataFrame({"model_id": ["c", "d"], "type": ["gen", "encoder"]})
    s1 = serialize_table_for_prompt(table1)
    s2 = serialize_table_for_prompt(table2)
    print("Serialize length", len(s1), len(s2))
    result = evaluate_diversity_with_llm(query="code gen", table1=table1, table2=table2)
    print("Eval result keys:", list(result.keys()))
