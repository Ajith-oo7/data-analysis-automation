"""
AI Insights Module

This module generates insights from data using AI models and extracts use cases.
"""

from .insight_generator import (
    generate_insights,
    extract_use_cases,
    InsightGenerator
)

__all__ = [
    'generate_insights',
    'extract_use_cases',
    'InsightGenerator'
] 