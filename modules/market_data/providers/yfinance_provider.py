"""Yahoo Finance-backed historical pricing provider."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging

import pandas as pd
import yfinance as yf

from modules.market_data.providers.base import BasePriceProvider, PriceMatrixResult


class YFinancePriceProvider(BasePriceProvider):
    """Fetch adjusted historical prices from Yahoo Finance."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    @property
    def provider_name(self) -> str:
        return "yfinance"

    def fetch_price_matrix(
        self,
        identifiers: list[str],
        *,
        start_date: str,
        end_date: str,
        frequency: str = "daily",
    ) -> PriceMatrixResult:
        unique_identifiers = [
            identifier
            for identifier in dict.fromkeys(
                str(identifier or "").strip()
                for identifier in identifiers
                if str(identifier or "").strip()
            )
        ]
        if not unique_identifiers:
            return PriceMatrixResult(prices=pd.DataFrame())

        primary_candidates = {
            identifier: self._candidate_tickers(identifier)[0]
            for identifier in unique_identifiers
        }
        bulk_prices = self._download_candidate_matrix(
            sorted(set(primary_candidates.values())),
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
        )

        resolved_identifiers: dict[str, str] = {}
        series_by_identifier: dict[str, pd.Series] = {}
        missing_identifiers: list[str] = []

        for identifier in unique_identifiers:
            primary_candidate = primary_candidates[identifier]
            candidate_series = bulk_prices.get(primary_candidate)
            if candidate_series is not None and not candidate_series.dropna().empty:
                series_by_identifier[identifier] = candidate_series.rename(identifier)
                resolved_identifiers[identifier] = primary_candidate
                continue

            fallback_series, used_ticker = self._fetch_single_identifier(
                identifier,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
            )
            if fallback_series is None or fallback_series.dropna().empty or not used_ticker:
                missing_identifiers.append(identifier)
                continue

            series_by_identifier[identifier] = fallback_series.rename(identifier)
            resolved_identifiers[identifier] = used_ticker

        if not series_by_identifier:
            return PriceMatrixResult(
                prices=pd.DataFrame(),
                resolved_identifiers=resolved_identifiers,
                missing_identifiers=missing_identifiers,
            )

        price_matrix = pd.concat(series_by_identifier.values(), axis=1).sort_index()
        price_matrix = price_matrix.loc[:, ~price_matrix.columns.duplicated()]
        return PriceMatrixResult(
            prices=price_matrix,
            resolved_identifiers=resolved_identifiers,
            missing_identifiers=missing_identifiers,
        )

    @staticmethod
    def normalize_identifier(identifier: str) -> str:
        """Normalize a repo ticker into something Yahoo is more likely to accept."""
        return str(identifier or "").strip().upper().replace("/", "-").replace(" ", "")

    def _candidate_tickers(self, identifier: str) -> list[str]:
        normalized = self.normalize_identifier(identifier)
        candidates = [normalized]
        if "." in normalized:
            base_ticker = normalized.split(".", maxsplit=1)[0]
            if base_ticker and base_ticker not in candidates:
                candidates.append(base_ticker)
        return [candidate for candidate in candidates if candidate]

    def _download_candidate_matrix(
        self,
        tickers: list[str],
        *,
        start_date: str,
        end_date: str,
        frequency: str,
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        interval = self._interval_from_frequency(frequency)
        try:
            raw = yf.download(
                tickers=tickers,
                start=start_date,
                end=self._exclusive_end(end_date),
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._logger.warning("Yahoo bulk download failed for %s: %s", tickers, exc)
            return pd.DataFrame()

        close_prices = self._extract_close_frame(raw, tickers=tickers)
        if close_prices.empty:
            return pd.DataFrame()
        return self._normalize_price_frame(close_prices, start_date=start_date, end_date=end_date)

    def _fetch_single_identifier(
        self,
        identifier: str,
        *,
        start_date: str,
        end_date: str,
        frequency: str,
    ) -> tuple[pd.Series | None, str | None]:
        interval = self._interval_from_frequency(frequency)
        for candidate in self._candidate_tickers(identifier):
            try:
                history = yf.Ticker(candidate).history(
                    start=start_date,
                    end=self._exclusive_end(end_date),
                    interval=interval,
                    auto_adjust=True,
                    repair=False,
                    raise_errors=False,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self._logger.debug(
                    "Yahoo single download failed for '%s' via '%s': %s",
                    identifier,
                    candidate,
                    exc,
                )
                continue

            close_prices = self._extract_close_frame(history, tickers=[candidate])
            if close_prices.empty or candidate not in close_prices.columns:
                continue

            normalized = self._normalize_price_frame(
                close_prices[[candidate]],
                start_date=start_date,
                end_date=end_date,
            )
            series = normalized.get(candidate)
            if series is None or series.dropna().empty:
                continue
            return series, candidate
        return None, None

    def _extract_close_frame(
        self,
        raw: pd.DataFrame,
        *,
        tickers: list[str],
    ) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            first_level = list(raw.columns.get_level_values(0).unique())
            second_level = list(raw.columns.get_level_values(1).unique())
            if "Close" in first_level:
                close_frame = raw["Close"]
            elif "Adj Close" in first_level:
                close_frame = raw["Adj Close"]
            elif "Close" in second_level:
                close_frame = raw.xs("Close", axis=1, level=1)
            elif "Adj Close" in second_level:
                close_frame = raw.xs("Adj Close", axis=1, level=1)
            else:
                return pd.DataFrame()
            return pd.DataFrame(close_frame)

        close_column = "Close" if "Close" in raw.columns else "Adj Close" if "Adj Close" in raw.columns else None
        if not close_column:
            return pd.DataFrame()
        ticker = tickers[0] if tickers else "UNKNOWN"
        return raw[[close_column]].rename(columns={close_column: ticker})

    def _normalize_price_frame(
        self,
        frame: pd.DataFrame,
        *,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        normalized = frame.copy()
        normalized.index = pd.to_datetime(normalized.index).tz_localize(None).normalize()
        normalized = normalized.sort_index()
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        normalized = normalized[(normalized.index >= start) & (normalized.index <= end)]
        normalized = normalized.apply(pd.to_numeric, errors="coerce")
        normalized = normalized.dropna(axis=0, how="all").dropna(axis=1, how="all")
        return normalized

    @staticmethod
    def _interval_from_frequency(frequency: str) -> str:
        if frequency == "monthly":
            return "1mo"
        return "1d"

    @staticmethod
    def _exclusive_end(end_date: str) -> str:
        end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        return end.strftime("%Y-%m-%d")
