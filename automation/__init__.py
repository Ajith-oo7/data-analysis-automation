"""
Automation Module

This module handles batch processing and on-demand execution.
"""

from .api_service import APIService
from .scheduler import Scheduler

__all__ = [
    'APIService',
    'Scheduler'
] 