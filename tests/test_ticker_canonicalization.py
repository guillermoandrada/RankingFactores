"""Tests for the shared vendor-identifier and canonical ticker forms."""

from __future__ import annotations

import pytest

from modules.shared.tickers import (
    canonical_ticker,
    canonical_ticker_map,
    ticker_from_bloomberg_id,
    ticker_from_ric,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AAPL US Equity", "AAPL"),
        ("A UN Equity", "A"),
        ("  AAPL US Equity  ", "AAPL"),
        ("AAPL\xa0US\xa0Equity", "AAPL"),
        ("2677689D US Equity", "2677689D"),
        ("BRK/B US Equity", "BRK/B"),
        ("AAPL", "AAPL"),
    ],
)
def test_ticker_from_bloomberg_id_takes_the_leading_symbol(raw: str, expected: str) -> None:
    assert ticker_from_bloomberg_id(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BFb US Equity", "BFb"),
        ("BRKa US Equity", "BRKa"),
        ("GOOG UW Equity", "GOOG"),
    ],
)
def test_ticker_from_bloomberg_id_keeps_the_vendor_spelling(raw: str, expected: str) -> None:
    """Upper-casing or aliasing here would write onto the wrong security."""
    assert ticker_from_bloomberg_id(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AAPL.O", "AAPL"),
        ("VOD.L", "VOD"),
        ("BRKa.N", "BRKa"),
        ("  AAPL.O  ", "AAPL"),
        ("AAPL", "AAPL"),
    ],
)
def test_ticker_from_ric_cuts_at_the_exchange_suffix(raw: str, expected: str) -> None:
    assert ticker_from_ric(raw) == expected


@pytest.mark.parametrize("extract", [ticker_from_bloomberg_id, ticker_from_ric])
@pytest.mark.parametrize("raw", ["", "   ", None, float("nan"), "nan", "None", "NaT"])
def test_vendor_extractors_reject_empty_values(extract, raw) -> None:
    """A blank cell must not become a security literally named 'nan'."""
    assert extract(raw) == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AAPL", "AAPL"),
        ("  aapl  ", "AAPL"),
        ("AAPL US Equity", "AAPL"),
        ("AAPL UN Equity", "AAPL"),
        ("AAPL US", "AAPL"),
        ("BRK/B US Equity", "BRK/B"),
        (" brk/b ", "BRK/B"),
        ("SPX Index", "SPX"),
        ("EURUSD Curncy", "EURUSD"),
        ("AAPL\xa0US\xa0Equity", "AAPL"),
    ],
)
def test_canonical_ticker_normalizes_vendor_forms(raw: str, expected: str) -> None:
    assert canonical_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None, float("nan"), "nan", "None"])
def test_canonical_ticker_rejects_empty_values(raw) -> None:
    assert canonical_ticker(raw) == ""


def test_canonical_ticker_applies_known_aliases() -> None:
    assert canonical_ticker("GOOG") == "GOOGL"
    assert canonical_ticker("goog us equity") == "GOOGL"
    assert canonical_ticker("NXP") == "NXPI"


def test_canonical_ticker_preserves_slash_share_classes() -> None:
    """The Yahoo '/' -> '-' mapping belongs to the provider, not the canonical form."""
    assert canonical_ticker("BF/B") == "BF/B"
    assert canonical_ticker("HEI/A US Equity") == "HEI/A"


def test_canonical_ticker_map_preserves_order_and_drops_blanks() -> None:
    mapping = canonical_ticker_map(["AAPL US Equity", "", None, "brk/b", "AAPL US Equity"])

    assert mapping == {"AAPL US Equity": "AAPL", "brk/b": "BRK/B"}
    assert list(mapping) == ["AAPL US Equity", "brk/b"]
