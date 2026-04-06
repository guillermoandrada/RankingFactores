"""Backtest orchestration service."""

from __future__ import annotations

from typing import Any

import pandas as pd

from api.schemas.backtests import PortfolioBacktestBody, StrategyBacktestBody
from modules.backtesting import (
    BacktestInterval,
    build_portfolio_time_series,
    compute_summary_metrics,
    extract_weights,
    join_benchmark_series,
    serialize_series,
)
from modules.market_data import BasePriceProvider
from api.services.portfolio_service import PortfolioService


class BacktestService:
    """Run portfolio and strategy backtests from portfolio weights."""

    def __init__(
        self,
        *,
        portfolio_service: PortfolioService,
        price_provider: BasePriceProvider,
    ) -> None:
        self._portfolio_service = portfolio_service
        self._price_provider = price_provider

    def backtest_portfolio(self, request: PortfolioBacktestBody) -> dict[str, Any]:
        result, _ = self._run_single_backtest(
            portfolio_rows=[row.model_dump() for row in request.portfolio],
            start_date=request.start_date.isoformat(),
            end_date=request.end_date.isoformat(),
            methodology=request.methodology,
            frequency=request.frequency,
            benchmark_ticker=request.benchmark_ticker.strip(),
            initial_value=float(request.capital_base),
        )
        return result

    def backtest_strategy(self, request: StrategyBacktestBody) -> dict[str, Any]:
        windows = sorted(
            request.schedule,
            key=lambda item: (item.start_date, item.end_date, item.period),
        )
        interval_payloads: list[dict[str, Any]] = []
        combined_frames: list[pd.DataFrame] = []
        all_warnings: list[str] = []
        running_value = float(request.portfolio_request.capital_base)

        for window in windows:
            interval_starting_value = running_value
            portfolio_result = self._portfolio_service.construct_portfolio(
                window.period,
                request.portfolio_request,
            )
            interval_result, interval_frame = self._run_single_backtest(
                portfolio_rows=portfolio_result.get("portfolio", []),
                start_date=window.start_date.isoformat(),
                end_date=window.end_date.isoformat(),
                methodology=request.methodology,
                frequency=request.frequency,
                benchmark_ticker=request.benchmark_ticker.strip(),
                initial_value=running_value,
            )
            running_value = float(interval_result["summary"]["ending_value"] or running_value)
            combined_frames.append(interval_frame)
            all_warnings.extend(interval_result.get("warnings", []))

            interval = BacktestInterval(
                period=window.period,
                start_date=window.start_date.isoformat(),
                end_date=window.end_date.isoformat(),
                starting_value=float(
                    interval_result["summary"]["starting_value"] or interval_starting_value
                ),
                ending_value=float(interval_result["summary"]["ending_value"] or running_value),
                position_count=len(portfolio_result.get("portfolio", [])),
                warnings=list(interval_result.get("warnings", [])),
            )
            interval_payloads.append(
                {
                    "period": interval.period,
                    "start_date": interval.start_date,
                    "end_date": interval.end_date,
                    "starting_value": interval.starting_value,
                    "ending_value": interval.ending_value,
                    "position_count": interval.position_count,
                    "warnings": interval.warnings,
                    "portfolio_summary": portfolio_result.get("summary", {}),
                    "portfolio_notes": portfolio_result.get("notes", []),
                    "portfolio": portfolio_result.get("portfolio", []),
                    "components": interval_result.get("components", []),
                }
            )

        combined_frame = self._stitch_frames(combined_frames)
        summary = compute_summary_metrics(
            portfolio_values=combined_frame.get("portfolio_value", pd.Series(dtype=float)),
            portfolio_returns=combined_frame.get("portfolio_return", pd.Series(dtype=float)),
            benchmark_values=combined_frame.get("benchmark_value"),
            benchmark_returns=combined_frame.get("benchmark_return"),
            frequency=request.frequency,
        )
        return {
            "provider": self._price_provider.provider_name,
            "methodology": request.methodology,
            "frequency": request.frequency,
            "benchmark_ticker": request.benchmark_ticker.strip() or None,
            "summary": summary,
            "warnings": all_warnings,
            "intervals": interval_payloads,
            "series": serialize_series(combined_frame),
        }

    def _run_single_backtest(
        self,
        *,
        portfolio_rows: list[dict[str, Any]],
        start_date: str,
        end_date: str,
        methodology: str,
        frequency: str,
        benchmark_ticker: str,
        initial_value: float,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        weights = extract_weights(portfolio_rows)
        warnings: list[str] = []

        security_tickers = [item.ticker for item in weights if item.ticker != "CASH_USD"]
        price_result = self._price_provider.fetch_price_matrix(
            security_tickers,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
        ) if security_tickers else self._empty_price_result()

        if price_result.missing_identifiers:
            warnings.append(
                "Missing price data for: " + ", ".join(sorted(price_result.missing_identifiers))
            )

        aligned_prices, active_weights, dropped_tickers = self._prepare_backtest_inputs(
            price_result.prices,
            weights,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
        )
        if dropped_tickers:
            warnings.append(
                "Dropped positions without a starting price for the selected window: "
                + ", ".join(sorted(dropped_tickers))
            )

        benchmark_frame = pd.DataFrame()
        benchmark_used_ticker: str | None = None
        if benchmark_ticker:
            benchmark_result = self._price_provider.fetch_price_matrix(
                [benchmark_ticker],
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
            )
            benchmark_frame = benchmark_result.prices
            benchmark_used_ticker = benchmark_result.resolved_identifiers.get(benchmark_ticker)
            if benchmark_result.missing_identifiers or benchmark_frame.empty:
                warnings.append(f"Benchmark '{benchmark_ticker}' could not be priced and was omitted.")
                benchmark_frame = pd.DataFrame()
                benchmark_used_ticker = None

        index = self._build_index(
            aligned_prices,
            benchmark_frame,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
        )
        aligned_prices = aligned_prices.reindex(index).ffill()
        benchmark_series = None
        if not benchmark_frame.empty:
            benchmark_series = benchmark_frame.iloc[:, 0].reindex(index).ffill().dropna()

        series_frame = build_portfolio_time_series(
            aligned_prices,
            weights=active_weights,
            methodology=methodology,
            initial_value=initial_value,
        )
        series_frame = series_frame.reindex(index).ffill().fillna(
            {
                "portfolio_value": initial_value,
                "portfolio_return": 0.0,
                "portfolio_cumulative": 1.0,
            }
        )
        series_frame = join_benchmark_series(
            series_frame,
            benchmark_series,
            initial_value=initial_value,
        )
        summary = compute_summary_metrics(
            portfolio_values=series_frame.get("portfolio_value", pd.Series(dtype=float)),
            portfolio_returns=series_frame.get("portfolio_return", pd.Series(dtype=float)),
            benchmark_values=series_frame.get("benchmark_value"),
            benchmark_returns=series_frame.get("benchmark_return"),
            frequency=frequency,
        )
        result = {
            "provider": self._price_provider.provider_name,
            "methodology": methodology,
            "frequency": frequency,
            "start_date": start_date,
            "end_date": end_date,
            "benchmark_ticker": benchmark_ticker or None,
            "resolved_tickers": price_result.resolved_identifiers,
            "resolved_benchmark": benchmark_used_ticker,
            "warnings": warnings,
            "summary": summary,
            "components": self._build_component_results(
                portfolio_rows=portfolio_rows,
                active_weights=active_weights,
                aligned_prices=aligned_prices,
                resolved_tickers=price_result.resolved_identifiers,
                dropped_tickers=dropped_tickers,
            ),
            "series": serialize_series(series_frame),
        }
        return result, series_frame

    @staticmethod
    def _prepare_backtest_inputs(
        prices: pd.DataFrame,
        weights: list,
        *,
        start_date: str,
        end_date: str,
        frequency: str,
    ) -> tuple[pd.DataFrame, list, list[str]]:
        if prices.empty:
            index = BacktestService._date_index(start_date=start_date, end_date=end_date, frequency=frequency)
            cash_weights = [item for item in weights if item.ticker == "CASH_USD"]
            return pd.DataFrame(index=index), cash_weights, []

        active_prices = prices.copy()
        first_date = active_prices.index[0]
        kept_weights = []
        dropped_tickers: list[str] = []
        for item in weights:
            if item.ticker == "CASH_USD":
                kept_weights.append(item)
                continue
            if item.ticker not in active_prices.columns or pd.isna(active_prices.loc[first_date, item.ticker]):
                dropped_tickers.append(item.ticker)
                continue
            kept_weights.append(item)
        kept_tickers = [item.ticker for item in kept_weights if item.ticker != "CASH_USD"]
        return active_prices.reindex(columns=kept_tickers), kept_weights, dropped_tickers

    @staticmethod
    def _build_index(
        prices: pd.DataFrame,
        benchmark_frame: pd.DataFrame,
        *,
        start_date: str,
        end_date: str,
        frequency: str,
    ) -> pd.DatetimeIndex:
        frames = [frame.index for frame in (prices, benchmark_frame) if not frame.empty]
        if frames:
            combined = frames[0]
            for index in frames[1:]:
                combined = combined.union(index)
            return pd.DatetimeIndex(sorted(combined.unique()))
        return BacktestService._date_index(start_date=start_date, end_date=end_date, frequency=frequency)

    @staticmethod
    def _date_index(
        *,
        start_date: str,
        end_date: str,
        frequency: str,
    ) -> pd.DatetimeIndex:
        freq = "B" if frequency == "daily" else "ME"
        return pd.date_range(start=start_date, end=end_date, freq=freq)

    @staticmethod
    def _empty_price_result():
        class _EmptyResult:
            prices = pd.DataFrame()
            resolved_identifiers: dict[str, str] = {}
            missing_identifiers: list[str] = []

        return _EmptyResult()

    @staticmethod
    def _stitch_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        stitched = pd.concat(frames).sort_index()
        stitched = stitched[~stitched.index.duplicated(keep="last")]
        return stitched

    @staticmethod
    def _build_component_results(
        *,
        portfolio_rows: list[dict[str, Any]],
        active_weights: list,
        aligned_prices: pd.DataFrame,
        resolved_tickers: dict[str, str],
        dropped_tickers: list[str],
    ) -> list[dict[str, Any]]:
        row_by_ticker = {
            str(row.get("ticker") or "").strip().upper(): row
            for row in portfolio_rows
            if str(row.get("ticker") or "").strip()
        }
        dropped = set(dropped_tickers)
        components: list[dict[str, Any]] = []
        for weight in active_weights:
            row = row_by_ticker.get(weight.ticker, {})
            component = {
                "ticker": weight.ticker,
                "name": weight.name or str(row.get("name") or ""),
                "sector": weight.sector or str(row.get("sector") or ""),
                "industry": weight.industry or str(row.get("industry") or ""),
                "target_weight": float(weight.weight),
                "resolved_ticker": resolved_tickers.get(weight.ticker),
                "start_price": None,
                "end_price": None,
                "total_return": 0.0 if weight.ticker == "CASH_USD" else None,
                "status": "cash" if weight.ticker == "CASH_USD" else "ok",
            }
            if weight.ticker != "CASH_USD" and weight.ticker in aligned_prices.columns:
                series = aligned_prices[weight.ticker].dropna()
                if not series.empty:
                    start_price = float(series.iloc[0])
                    end_price = float(series.iloc[-1])
                    component["start_price"] = start_price
                    component["end_price"] = end_price
                    component["total_return"] = end_price / start_price - 1.0 if start_price else None
            components.append(component)

        for ticker in dropped:
            row = row_by_ticker.get(ticker, {})
            components.append(
                {
                    "ticker": ticker,
                    "name": str(row.get("name") or ""),
                    "sector": str(row.get("sector") or ""),
                    "industry": str(row.get("industry") or ""),
                    "target_weight": float(row.get("target_weight") or 0.0),
                    "resolved_ticker": resolved_tickers.get(ticker),
                    "start_price": None,
                    "end_price": None,
                    "total_return": None,
                    "status": "missing_start_price",
                }
            )
        return components
