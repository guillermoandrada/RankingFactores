"""Performance metric helpers for backtests."""

from __future__ import annotations

import math

import pandas as pd


def annualization_factor(frequency: str) -> int:
    if frequency == "monthly":
        return 12
    return 252


def compute_max_drawdown(cumulative: pd.Series) -> float:
    if cumulative.empty:
        return 0.0
    running_peak = cumulative.cummax()
    drawdown = cumulative / running_peak - 1.0
    return float(drawdown.min())


def compute_summary_metrics(
    *,
    portfolio_values: pd.Series,
    portfolio_returns: pd.Series,
    frequency: str,
    benchmark_values: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, float | None]:
    if portfolio_values.empty:
        return {
            "starting_value": None,
            "ending_value": None,
            "total_return": None,
            "annualized_return": None,
            "volatility": None,
            "max_drawdown": None,
            "benchmark_total_return": None,
            "benchmark_annualized_return": None,
            "tracking_error": None,
            "excess_total_return": None,
            "relative_total_return": None,
        }

    annualizer = annualization_factor(frequency)
    periods = max(len(portfolio_returns), 1)
    starting_value = float(portfolio_values.iloc[0])
    ending_value = float(portfolio_values.iloc[-1])
    total_return = ending_value / starting_value - 1.0 if starting_value else None
    annualized_return = _annualized_return(
        starting_value=starting_value,
        ending_value=ending_value,
        periods=periods,
        annualizer=annualizer,
    )
    volatility = _annualized_volatility(portfolio_returns, annualizer)
    max_drawdown = compute_max_drawdown(portfolio_values / starting_value if starting_value else portfolio_values)

    summary: dict[str, float | None] = {
        "starting_value": starting_value,
        "ending_value": ending_value,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "benchmark_total_return": None,
        "benchmark_annualized_return": None,
        "tracking_error": None,
        "excess_total_return": None,
        "relative_total_return": None,
    }

    if benchmark_values is None or benchmark_values.empty:
        return summary

    benchmark_start = float(benchmark_values.iloc[0])
    benchmark_end = float(benchmark_values.iloc[-1])
    benchmark_total_return = benchmark_end / benchmark_start - 1.0 if benchmark_start else None
    benchmark_return_series = (
        benchmark_returns
        if benchmark_returns is not None
        else pd.Series(dtype=float)
    )
    benchmark_annualized_return = _annualized_return(
        starting_value=benchmark_start,
        ending_value=benchmark_end,
        periods=max(len(benchmark_return_series), 1),
        annualizer=annualizer,
    )
    excess_returns = _aligned_excess_returns(portfolio_returns, benchmark_return_series)
    summary.update(
        {
            "benchmark_total_return": benchmark_total_return,
            "benchmark_annualized_return": benchmark_annualized_return,
            "tracking_error": _annualized_volatility(excess_returns, annualizer),
            "excess_total_return": (
                total_return - benchmark_total_return
                if total_return is not None and benchmark_total_return is not None
                else None
            ),
            "relative_total_return": (
                (ending_value / benchmark_end) - 1.0
                if benchmark_end
                else None
            ),
        }
    )
    return summary


def _annualized_return(
    *,
    starting_value: float,
    ending_value: float,
    periods: int,
    annualizer: int,
) -> float | None:
    if starting_value <= 0 or ending_value <= 0 or periods <= 0:
        return None
    years = periods / annualizer
    if years <= 0:
        return None
    return math.pow(ending_value / starting_value, 1.0 / years) - 1.0


def _annualized_volatility(returns: pd.Series, annualizer: int) -> float | None:
    clean = returns.dropna()
    if clean.empty:
        return None
    return float(clean.std(ddof=0) * math.sqrt(annualizer))


def _aligned_excess_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    if portfolio_returns.empty or benchmark_returns.empty:
        return pd.Series(dtype=float)
    combined = pd.concat(
        [portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if combined.empty:
        return pd.Series(dtype=float)
    return combined["portfolio"] - combined["benchmark"]
