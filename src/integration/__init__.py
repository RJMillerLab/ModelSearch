"""
Table Integration Module

This module provides functionality to integrate/merge multiple tables retrieved from search results.
Uses Blend_internal's Union and Intersection combiners for table integration.
"""

__all__ = ["TableIntegrater", "GroupTableIntegrater", "BaseKeywordRecognizer", "KeywordRecognizer", "KeywordRecognizerForAlite"]


def __getattr__(name: str):
    if name == "TableIntegrater":
        from .table_integration import TableIntegrater
        return TableIntegrater
    if name == "GroupTableIntegrater":
        from .table_integration import GroupTableIntegrater
        return GroupTableIntegrater
    if name == "BaseKeywordRecognizer":
        from .quick_aug_recognition import BaseKeywordRecognizer
        return BaseKeywordRecognizer
    if name == "KeywordRecognizer":
        from .quick_aug_recognition import KeywordRecognizer
        return KeywordRecognizer
    if name == "KeywordRecognizerForAlite":
        from .quick_aug_recognition import KeywordRecognizerForAlite
        return KeywordRecognizerForAlite
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
