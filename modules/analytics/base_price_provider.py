"""
Abstract price provider interface for market data backends (e.g. Yahoo Finance).

Kept in modules/analytics/ (flat layout) — no time_series subpackage required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple


class BasePriceProvider(ABC):
    """Minimal contract for historical prices and returns by entity identifier."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider id (e.g. 'yfinance')."""

    @abstractmethod
    def validate_identifier(self, identifier: str) -> bool:
        """Syntax-level validation (no network)."""

    @abstractmethod
    def resolve_identifier(self, entity_identifier: str) -> List[str]:
        """Return ordered ticker candidates for the given identifier."""

    @abstractmethod
    def fetch_historical_prices(
        self,
        entity_identifier: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "daily",
        period: Optional[str] = None,
        interval: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[str, float]]:
        """Return (date_str, price) points."""

    @abstractmethod
    def fetch_historical_returns(
        self,
        entity_identifier: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "daily",
        period: Optional[str] = None,
        interval: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[str, float]]:
        """Return (date_str, return) points."""
