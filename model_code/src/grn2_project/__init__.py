"""Shared project layout for code, data, results, documentation and deployment."""

from .paths import (
    ARCHIVE_ROOT,
    DATA_ROOT,
    PROCESSED_DATA_ROOT,
    PROJECT_ROOT,
    RAW_DATA_ROOT,
    RESULT_DATA_ROOT,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "RAW_DATA_ROOT",
    "PROCESSED_DATA_ROOT",
    "RESULT_DATA_ROOT",
    "ARCHIVE_ROOT",
]
