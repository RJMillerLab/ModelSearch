"""
Table Integration Module

This module provides functionality to integrate/merge multiple tables retrieved from search results.
Uses Blend_internal's Union and Intersection combiners for table integration.
"""

from .table_integration import integrate_tables, integrate_tables_from_card2tab2card, integrate_tables_from_card2card

__all__ = ['integrate_tables', 'integrate_tables_from_card2tab2card', 'integrate_tables_from_card2card']

