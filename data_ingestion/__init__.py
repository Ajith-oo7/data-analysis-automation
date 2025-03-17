"""
Data Ingestion Module

This module handles file ingestion, validation, and preprocessing tasks.
"""

from .file_handler import (
    detect_file_type,
    is_file_type_supported,
    load_file,
    validate_dataframe_structure,
    FileWatcher,
    SUPPORTED_FILE_TYPES
)

__all__ = [
    'detect_file_type',
    'is_file_type_supported',
    'load_file',
    'validate_dataframe_structure',
    'FileWatcher',
    'SUPPORTED_FILE_TYPES'
] 