"""Domain models for backtesting workflows."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PortfolioWeight:
    """Single portfolio weight used in a backtest."""

    ticker: str
    weight: float
    name: str = ""
    sector: str = ""
    industry: str = ""


@dataclass
class BacktestWindow:
    """Single historical interval to simulate."""

    period: str
    start_date: str
    end_date: str


@dataclass
class BacktestInterval:
    """Per-window result metadata for stitched strategy simulations."""

    period: str
    start_date: str
    end_date: str
    starting_value: float
    ending_value: float
    position_count: int
    warnings: list[str] = field(default_factory=list)
