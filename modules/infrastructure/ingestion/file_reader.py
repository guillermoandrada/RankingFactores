"""Reader facade for supported ingestion formats."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from modules.infrastructure.ingestion.readers import (
    BaseFileReader,
    BqlFileReader,
    BloombergFileReader,
    ReutersMetricsFileReader,
)


class FileReader:
    """Facade that resolves the concrete ingestion reader."""

    def __init__(self, reader: Optional[BaseFileReader] = None) -> None:
        if reader is not None:
            self._readers: dict[str, BaseFileReader] = {"custom": reader}
            self._default_reader_name = "custom"
        else:
            self._readers = {
                "bql": BqlFileReader(),
                "bloomberg": BloombergFileReader(),
                "reuters_metrics": ReutersMetricsFileReader(),
            }
            self._default_reader_name = "bloomberg"

    def read(self, filepath: str, reader_name: str = "bloomberg") -> pd.DataFrame:
        return self._resolve_reader(filepath, reader_name).read(filepath)

    def extract_period(self, filepath: str, reader_name: str = "bloomberg") -> str:
        return self._resolve_reader(filepath, reader_name).extract_period(filepath)

    def extract_index_code(self, filepath: str, reader_name: str = "bloomberg"):
        return self._resolve_reader(filepath, reader_name).extract_index_code(filepath)

    def _resolve_reader(self, filepath: str, reader_name: str) -> BaseFileReader:
        requested = (reader_name or self._default_reader_name).strip().lower()
        if requested == "auto":
            return self._auto_detect_reader(filepath)

        reader = self._readers.get(requested)
        if reader is None:
            supported = ", ".join(sorted(self._readers))
            raise ValueError(f"Unsupported reader '{reader_name}'. Supported readers: {supported}")
        return reader

    def _auto_detect_reader(self, filepath: str) -> BaseFileReader:
        for name in ("reuters_metrics", "bql", self._default_reader_name):
            reader = self._readers.get(name)
            if reader and reader.can_read(filepath):
                return reader

        for name, reader in self._readers.items():
            if reader.can_read(filepath):
                return reader

        raise ValueError(f"Could not detect a reader for file: {filepath}")
