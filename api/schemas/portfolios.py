"""Request schemas for portfolio construction."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class HoldingRow(BaseModel):
    ticker: str
    quantity: float = 0.0
    price: Optional[float] = None
    market_value: Optional[float] = None
    name: str = ""


class EthicalFilterRow(BaseModel):
    ticker: str
    ethical_evaluation: str = Field(
        default="",
        description="Use values like 'No' to exclude a security.",
    )


class PortfolioBuildBody(BaseModel):
    scoring_profile: str = Field(..., description="Scoring profile to use.")
    industry: str = ""
    sector: str = ""
    index: str = ""
    strategy: Literal["legacy_rebalance", "smart_beta", "long_short"] = "legacy_rebalance"
    construction_mode: Literal["new_portfolio", "rebalance_existing"] = "new_portfolio"
    capital_base: float = Field(
        default=1.0,
        description="Portfolio capital base when building a fresh portfolio.",
    )

    current_holdings: list[HoldingRow] = Field(default_factory=list)
    ethical_filter_rows: list[EthicalFilterRow] = Field(default_factory=list)

    sector_targets: dict[str, float] = Field(default_factory=dict)
    industry_targets: dict[str, float] = Field(default_factory=dict)

    max_position: float = 0.05
    neutral_position: float = 0.03
    score_quantile_cutoff: float = 0.5
    min_trade_weight: float = 0.0

    top_n: Optional[int] = None
    smart_beta_max_weight: float = 0.10

    bucket_count: int = 10
    long_bucket_count: int = 1
    short_bucket_count: int = 1
    long_short_weighting: Literal["equal", "score"] = "equal"
    gross_exposure: float = 1.0
    net_exposure: float = 0.0

