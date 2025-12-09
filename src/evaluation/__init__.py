"""
Evaluation module for ModelSearch

Provides LLM-based evaluation of table integration results, focusing on diversity scoring.
"""

from .prompt import get_diversity_evaluation_prompt
from .llm import evaluate_diversity_with_llm, load_fake_response

__all__ = [
    'get_diversity_evaluation_prompt',
    'evaluate_diversity_with_llm',
    'load_fake_response'
]

