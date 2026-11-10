"""Forward returns are fetched once per window, not once per metric per period."""

from __future__ import annotations

import logging

import pandas as pd

from modules.domain.analytics.ic_analyzer import ICAnalyzer


class _CountingAnalyzer(ICAnalyzer):
    """Counts provider fetches without touching a database or a price provider."""

    def __init__(self, available: dict[str, float]) -> None:
        self._available = available
        self._logger = logging.getLogger("stub")
        self._forward_returns = {}
        self._forward_returns_attempted = {}
        self.fetches: list[list[str]] = []

    def _fetch_forward_returns(self, *, tickers, start_date, end_date) -> pd.DataFrame:
        self.fetches.append(list(tickers))
        rows = [
            {"ticker": ticker, "forward_return": self._available[ticker]}
            for ticker in tickers
            if ticker in self._available
        ]
        return pd.DataFrame(rows, columns=["ticker", "forward_return"])


_WINDOW = {"start_date": "2025-05-14", "end_date": "2025-08-14"}


def test_a_repeated_window_is_served_from_the_first_fetch() -> None:
    """This is the cost that made a wide IC run time out: one fetch per metric."""
    analyzer = _CountingAnalyzer({"AAA": 0.1, "BBB": 0.2})

    first = analyzer._compute_forward_returns(tickers=["AAA", "BBB"], **_WINDOW)
    second = analyzer._compute_forward_returns(tickers=["AAA", "BBB"], **_WINDOW)

    assert len(analyzer.fetches) == 1
    assert second.equals(first)


def test_only_tickers_not_seen_yet_are_fetched() -> None:
    analyzer = _CountingAnalyzer({"AAA": 0.1, "BBB": 0.2})

    analyzer._compute_forward_returns(tickers=["AAA"], **_WINDOW)
    result = analyzer._compute_forward_returns(tickers=["AAA", "BBB"], **_WINDOW)

    assert analyzer.fetches == [["AAA"], ["BBB"]]
    assert sorted(result["ticker"]) == ["AAA", "BBB"]


def test_a_ticker_without_prices_is_not_fetched_again() -> None:
    """A missing ticker costs a full provider retry, so it must be asked for once."""
    analyzer = _CountingAnalyzer({"AAA": 0.1})

    analyzer._compute_forward_returns(tickers=["AAA", "GONE"], **_WINDOW)
    result = analyzer._compute_forward_returns(tickers=["AAA", "GONE"], **_WINDOW)

    assert len(analyzer.fetches) == 1
    assert result["ticker"].tolist() == ["AAA"]


def test_a_different_window_is_fetched_separately() -> None:
    analyzer = _CountingAnalyzer({"AAA": 0.1})

    analyzer._compute_forward_returns(tickers=["AAA"], **_WINDOW)
    analyzer._compute_forward_returns(
        tickers=["AAA"], start_date="2025-05-14", end_date="2025-11-14"
    )

    assert len(analyzer.fetches) == 2


def test_only_requested_tickers_come_back() -> None:
    analyzer = _CountingAnalyzer({"AAA": 0.1, "BBB": 0.2})

    analyzer._compute_forward_returns(tickers=["AAA", "BBB"], **_WINDOW)
    result = analyzer._compute_forward_returns(tickers=["BBB"], **_WINDOW)

    assert result["ticker"].tolist() == ["BBB"]


def test_clearing_the_caches_forces_a_refetch() -> None:
    """Uploaded prices must reach the next run, gaps included."""
    analyzer = _CountingAnalyzer({"AAA": 0.1})

    analyzer._compute_forward_returns(tickers=["AAA"], **_WINDOW)
    analyzer.clear_caches()
    analyzer._compute_forward_returns(tickers=["AAA"], **_WINDOW)

    assert len(analyzer.fetches) == 2
