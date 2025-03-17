"""
Reporting Module

This module handles report generation using Tableau/Power BI.
"""

from .report_generator import (
    generate_report,
    ReportGenerator
)

__all__ = [
    'generate_report',
    'ReportGenerator'
] 