"""
Reader facade for Bloomberg Excel format.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from modules.ingestion.readers import BaseFileReader, BloombergFileReader


class FileReader:
    """Facade for Bloomberg Excel reader."""

    def __init__(self, reader: Optional[BaseFileReader] = None) -> None:
        self._reader = reader or BloombergFileReader()

    def read(self, filepath: str, reader_name: str = "bloomberg") -> pd.DataFrame:
        return self._reader.read(filepath)

    def extract_period(self, filepath: str, reader_name: str = "bloomberg") -> str:
        return self._reader.extract_period(filepath)

    def extract_index_code(self, filepath: str, reader_name: str = "bloomberg"):
        return self._reader.extract_index_code(filepath)
