"""
Evaluation module for ModelSearch.

Currently provides:
- LLM-based evaluation of integrated tables (relevance / coverage / diversity).
- Metric-based (non-LLM) diversity / novelty utilities for model-card rankings
  and integrated tables.
"""

from .prompt import get_diversity_evaluation_prompt
from .llm import evaluate_diversity_with_llm  # LLM-based evaluation
from .llm_qa import answer_question_with_llm, get_qa_prompt
from .metrics import (  # type: ignore F401
    compare_model_lists_diversity,
    compare_tables_diversity,
    evaluate_model_list_diversity,
    evaluate_table_diversity,
    load_card2card_embeddings,
)

__all__ = [
    "get_diversity_evaluation_prompt",
    "evaluate_diversity_with_llm",
    "get_qa_prompt",
    "answer_question_with_llm",
    "load_card2card_embeddings",
    "evaluate_model_list_diversity",
    "compare_model_lists_diversity",
    "evaluate_table_diversity",
    "compare_tables_diversity",
]

