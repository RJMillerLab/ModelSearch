"""
QA entry: answer_question_with_llm. All LLM logic lives in evaluation.llm.
"""
import os
from typing import Dict, Any, Optional, List
import pandas as pd

from evaluation.llm import (
    call_llm_api,
    serialize_table_for_prompt,
    parse_llm_response_qa,
)
from .prompt import get_qa_prompt


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
        raise ValueError("OPENAI_API_KEY not found. Please set OPENAI_API_KEY in your environment or use fake response mode.")

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
    model_card_info = {}
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
