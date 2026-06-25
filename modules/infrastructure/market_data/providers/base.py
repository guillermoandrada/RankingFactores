"""Base interfaces for market-data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PriceMatrixResult:
    """Normalized market-data result for multiple identifiers."""

    prices: pd.DataFrame
    resolved_identifiers: dict[str, str] = field(default_factory=dict)
    missing_identifiers: list[str] = field(default_factory=list)


class BasePriceProvider(ABC):
    """Read-only interface for historical price providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the user-facing provider name."""

    @abstractmethod
    def fetch_price_matrix(
        self,
        identifiers: list[str],
        *,
        start_date: str,
        end_date: str,
        frequency: str = "daily",
    ) -> PriceMatrixResult:
        """Return a price matrix keyed by original identifier."""
