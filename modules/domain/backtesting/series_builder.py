"""Helpers for transforming price data into backtest series."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from modules.domain.backtesting.models import PortfolioWeight


CASH_TICKER = "CASH_USD"
_WEIGHT_EPSILON = 1e-12


def extract_weights(rows: Iterable[dict]) -> list[PortfolioWeight]:
    """Convert API payload rows into normalized portfolio weights."""
    weights: list[PortfolioWeight] = []
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        try:
            weight = float(row.get("target_weight") or 0.0)
        except (TypeError, ValueError):
            continue
        if abs(weight) <= _WEIGHT_EPSILON:
            continue
        weights.append(
            PortfolioWeight(
                ticker=ticker,
                weight=weight,
                name=str(row.get("name") or ""),
                sector=str(row.get("sector") or ""),
                industry=str(row.get("industry") or ""),
            )
        )
    return weights


def build_portfolio_time_series(
    prices: pd.DataFrame,
    *,
    weights: list[PortfolioWeight],
    methodology: str,
    initial_value: float = 1.0,
) -> pd.DataFrame:
    """Build a cumulative portfolio series from aligned prices."""
    if initial_value <= 0:
        raise ValueError("initial_value must be positive.")
    if methodology not in {"fixed_weights", "drifting_weights"}:
        raise ValueError(f"Unsupported methodology '{methodology}'.")

    if prices.empty or not weights:
        index = prices.index if not prices.empty else pd.DatetimeIndex([])
        return _constant_series(index=index, initial_value=initial_value)

    security_weights = {
        item.ticker: float(item.weight)
        for item in weights
        if item.ticker != CASH_TICKER
    }
    cash_weight = 1.0 - sum(security_weights.values())
    active_prices = prices.loc[:, [ticker for ticker in security_weights if ticker in prices.columns]].copy()

    if active_prices.empty:
        return _constant_series(index=prices.index, initial_value=initial_value)

    if methodology == "fixed_weights":
        returns = active_prices.pct_change().fillna(0.0)
        fixed_weights = pd.Series(security_weights, dtype=float)
        portfolio_returns = returns.mul(fixed_weights, axis=1).sum(axis=1)
        portfolio_values = (1.0 + portfolio_returns).cumprod() * initial_value
    else:
        rebased_prices = active_prices.divide(active_prices.iloc[0]).ffill().fillna(1.0)
        position_values = rebased_prices.mul(pd.Series(security_weights, dtype=float), axis=1)
        portfolio_values = (cash_weight + position_values.sum(axis=1)) * initial_value
        portfolio_returns = portfolio_values.pct_change().fillna(0.0)

    result = pd.DataFrame(
        {
            "portfolio_value": portfolio_values,
            "portfolio_return": portfolio_returns,
            "portfolio_cumulative": portfolio_values / initial_value,
        }
    )
    return result


def join_benchmark_series(
    portfolio_series: pd.DataFrame,
    benchmark_prices: pd.Series | None,
    *,
    initial_value: float,
) -> pd.DataFrame:
    """Append normalized benchmark columns to a portfolio series."""
    if benchmark_prices is None or benchmark_prices.empty:
        return portfolio_series

    benchmark_values = benchmark_prices / benchmark_prices.iloc[0] * initial_value
    benchmark_returns = benchmark_values.pct_change().fillna(0.0)
    joined = portfolio_series.copy()
    joined["benchmark_value"] = benchmark_values.reindex(joined.index).ffill()
    joined["benchmark_return"] = benchmark_returns.reindex(joined.index).fillna(0.0)
    joined["benchmark_cumulative"] = joined["benchmark_value"] / initial_value
    joined["excess_return"] = joined["portfolio_return"] - joined["benchmark_return"]
    joined["relative_cumulative"] = joined["portfolio_cumulative"] / joined["benchmark_cumulative"] - 1.0
    return joined


def serialize_series(frame: pd.DataFrame) -> list[dict[str, float | str | None]]:
    """Convert a time series frame into JSON-friendly records."""
    if frame.empty:
        return []

    serializable = frame.copy()
    serializable.index = pd.to_datetime(serializable.index).strftime("%Y-%m-%d")
    serializable = serializable.reset_index().rename(columns={"index": "date"})
    serializable = serializable.where(pd.notnull(serializable), None)
    return serializable.to_dict(orient="records")


def _constant_series(
    *,
    index: pd.DatetimeIndex,
    initial_value: float,
) -> pd.DataFrame:
    if len(index) == 0:
        return pd.DataFrame(
            columns=["portfolio_value", "portfolio_return", "portfolio_cumulative"],
        )
    return pd.DataFrame(
        {
            "portfolio_value": [initial_value] * len(index),
            "portfolio_return": [0.0] * len(index),
            "portfolio_cumulative": [1.0] * len(index),
        },
        index=index,
    )
