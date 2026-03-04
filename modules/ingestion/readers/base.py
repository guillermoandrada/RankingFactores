from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class BaseFileReader(ABC):
    """Abstract file reader strategy for ingestion."""

    @abstractmethod
    def can_read(self, filepath: str) -> bool:
        """Return True if this reader supports the filepath."""

    @abstractmethod
    def read(self, filepath: str) -> pd.DataFrame:
        """Read the main table and return a DataFrame."""

    @abstractmethod
    def extract_period(self, filepath: str) -> str:
        """Extract period label (e.g. '2022/01/01')."""

    @abstractmethod
    def extract_index_code(self, filepath: str) -> Optional[str]:
        """Extract index code (e.g. 'B500') if available."""
