"""Base interfaces for market-data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

MONTHLY_FREQUENCY = "monthly"
_MONTH_END_RULE = "ME"


@dataclass
class PriceMatrixResult:
    """
    Normalized market-data result for multiple identifiers.

    Every requested identifier appears in exactly one of: a populated column of
    ``prices``, ``missing_identifiers``, or — when it was priced but not across
    the whole requested window — ``partial_coverage``. Columns are keyed by the
    identifier the caller supplied; ``resolved_identifiers`` records the symbol
    the data actually came from.
    """

    prices: pd.DataFrame
    resolved_identifiers: dict[str, str] = field(default_factory=dict)
    missing_identifiers: list[str] = field(default_factory=list)
    partial_coverage: list[str] = field(default_factory=list)


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


def to_period_end(data: pd.DataFrame | pd.Series, frequency: str):
    """
    Force monthly data onto a month-end index.

    Providers disagree on how monthly bars are labelled — Yahoo Finance stamps
    them with the first day of the month while database series resample to the
    last. Both are routed through here so a matrix assembled from several sources
    shares one index instead of interleaving month-start and month-end rows.
    """
    if frequency != MONTHLY_FREQUENCY or data.empty:
        return data
    resampled = data.resample(_MONTH_END_RULE).last()
    if isinstance(resampled, pd.Series):
        return resampled.dropna()
    return resampled.dropna(axis=0, how="all")
