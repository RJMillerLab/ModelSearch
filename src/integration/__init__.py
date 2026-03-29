"""
Table Integration Module

This module provides functionality to integrate/merge multiple tables retrieved from search results.
Uses Blend_internal's Union and Intersection combiners for table integration.
"""

__all__ = ["TableIntegrater", "BaseKeywordRecognizer", "KeywordRecognizer"]


def __getattr__(name: str):
    if name == "TableIntegrater":
        from .table_integration import TableIntegrater
        return TableIntegrater
    if name == "BaseKeywordRecognizer":
        from .quick_aug_recognition import BaseKeywordRecognizer
        return BaseKeywordRecognizer
    if name == "KeywordRecognizer":
        from .quick_aug_recognition import KeywordRecognizer
        return KeywordRecognizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
