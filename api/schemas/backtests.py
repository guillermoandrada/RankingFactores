"""Request schemas for portfolio and strategy backtests."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from api.schemas.portfolios import PortfolioBuildBody


class PortfolioBacktestRow(BaseModel):
    ticker: str
    target_weight: float = 0.0
    name: str = ""
    sector: str = ""
    industry: str = ""


class PortfolioBacktestBody(BaseModel):
    portfolio: list[PortfolioBacktestRow] = Field(default_factory=list)
    start_date: date
    end_date: date
    methodology: Literal["fixed_weights", "drifting_weights"] = "drifting_weights"
    benchmark_ticker: str = ""
    frequency: Literal["daily", "monthly"] = "daily"
    capital_base: float = 1.0

    @model_validator(mode="after")
    def validate_request(self) -> "PortfolioBacktestBody":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date.")
        if not self.portfolio:
            raise ValueError("portfolio must include at least one row.")
        if self.capital_base <= 0:
            raise ValueError("capital_base must be positive.")
        return self


class StrategyBacktestWindow(BaseModel):
    period: str
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_window(self) -> "StrategyBacktestWindow":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date.")
        return self


class StrategyBacktestBody(BaseModel):
    portfolio_request: PortfolioBuildBody
    schedule: list[StrategyBacktestWindow] = Field(default_factory=list)
    methodology: Literal["fixed_weights", "drifting_weights"] = "drifting_weights"
    benchmark_ticker: str = ""
    frequency: Literal["daily", "monthly"] = "daily"

    @model_validator(mode="after")
    def validate_request(self) -> "StrategyBacktestBody":
        if not self.schedule:
            raise ValueError("schedule must include at least one row.")
        if self.portfolio_request.construction_mode != "new_portfolio":
            raise ValueError("strategy backtests only support construction_mode='new_portfolio'.")

        ordered = sorted(self.schedule, key=lambda item: (item.start_date, item.end_date, item.period))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_date <= previous.end_date:
                raise ValueError("schedule rows must not overlap.")
        return self
