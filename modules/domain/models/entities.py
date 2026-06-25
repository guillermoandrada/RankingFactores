"""
Domain entities and data transfer objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ImportResult:
    """Result of a data import operation."""

    period: str
    companies_count: int
    metrics_count: int
    records_count: int
    index_code: Optional[str] = None


@dataclass
class RankingParams:
    """Parameters for computing a security ranking."""

    period: str
    metric_ids: list[int]
    index_name: Optional[str] = None
    industry_name: Optional[str] = None
    weights: dict[str, float] = field(default_factory=dict)
    method: str = "linear"  # "linear" or "softplus"


@dataclass
class RankingResult:
    """Result of a ranking computation."""

    df: "pd.DataFrame"
    direction_map: dict[str, bool]
    params: RankingParams
