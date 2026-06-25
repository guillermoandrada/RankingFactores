"""Portfolio domain models used by construction strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SecurityCandidate:
    """Ranked security available for portfolio construction."""

    ticker: str
    score: float
    name: str = ""
    sector: str = ""
    industry: str = ""
    ethical_allowed: bool = True
    extra_filters_allowed: bool = True

    @property
    def is_allowed(self) -> bool:
        return self.ethical_allowed and self.extra_filters_allowed


@dataclass
class HoldingPosition:
    """Existing holding used for rebalance mode."""

    ticker: str
    quantity: float = 0.0
    price: Optional[float] = None
    market_value: Optional[float] = None
    name: str = ""
    sector: str = ""
    industry: str = ""
    score: Optional[float] = None

    def amount(self) -> float:
        if self.market_value is not None:
            return float(self.market_value)
        if self.price is None:
            raise ValueError(
                f"Holding '{self.ticker}' requires either market_value or price."
            )
        return float(self.quantity) * float(self.price)


@dataclass
class TargetPosition:
    """Target portfolio position with weight and optional amount."""

    ticker: str
    target_weight: float
    score: float
    name: str = ""
    sector: str = ""
    industry: str = ""
    target_amount: Optional[float] = None


@dataclass
class PortfolioTrade:
    """Trade recommendation between current and target state."""

    action: str
    ticker: str
    weight_delta: float
    current_weight: float
    target_weight: float
    reason: str
    quantity_delta: Optional[float] = None
    price: Optional[float] = None
    sector: str = ""
    industry: str = ""


@dataclass
class PortfolioDiagnostics:
    """Diagnostics emitted by strategies."""

    excluded: list[dict] = field(default_factory=list)
    constraints: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

