"""Helpers for parsing holdings and filter inputs."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from modules.domain.portfolio.models import HoldingPosition

_KNOWN_DUPLICATES = {"GOOG": "GOOGL", "NXP": "NXPI"}


def normalize_ticker(value: Any) -> str:
    ticker = str(value or "").strip().upper()
    return _KNOWN_DUPLICATES.get(ticker, ticker)


def parse_holdings_rows(rows: list[dict[str, Any]]) -> tuple[list[HoldingPosition], float]:
    """Parse holdings rows from JSON-friendly input."""
    positions: list[HoldingPosition] = []
    cash = 0.0
    for row in rows:
        ticker = normalize_ticker(row.get("ticker"))
        if not ticker:
            continue
        if ticker == "CASH_USD":
            cash += float(row.get("market_value") or row.get("cash") or row.get("quantity") or 0.0)
            continue
        quantity = float(row.get("quantity") or 0.0)
        price = row.get("price")
        market_value = row.get("market_value")
        positions.append(
            HoldingPosition(
                ticker=ticker,
                quantity=quantity,
                price=float(price) if price not in (None, "") else None,
                market_value=float(market_value) if market_value not in (None, "") else None,
                name=str(row.get("name") or ""),
            )
        )
    return positions, cash


def parse_holdings_excel(file_content: bytes) -> tuple[list[HoldingPosition], float]:
    """
    Parse legacy-style holdings workbook.

    Expected first column = ticker. Second column = quantity or market value.
    Optional columns: price, market_value.
    """
    df = pd.read_excel(io.BytesIO(file_content))
    rows: list[dict[str, Any]] = []
    columns = list(df.columns)
    for _, row in df.iterrows():
        ticker = row.iloc[0] if len(columns) >= 1 else ""
        second_value = row.iloc[1] if len(columns) >= 2 else 0.0
        price = row.iloc[2] if len(columns) >= 3 else None
        market_value = row.iloc[3] if len(columns) >= 4 else None
        rows.append({
            "ticker": ticker,
            "quantity": second_value,
            "price": price,
            "market_value": market_value,
        })
    return parse_holdings_rows(rows)


def parse_ethical_filter_rows(rows: list[dict[str, Any]]) -> set[str]:
    """Return blocked tickers from normalized ethical-filter rows."""
    blocked: set[str] = set()
    for row in rows:
        ticker = normalize_ticker(row.get("ticker"))
        evaluation = str(row.get("ethical_evaluation") or row.get("value") or "").strip().lower()
        if ticker and evaluation == "no":
            blocked.add(ticker.replace(".", "/"))
    return blocked


def parse_ethical_filter_excel(file_content: bytes) -> set[str]:
    """Parse the legacy ethical filter workbook into blocked tickers."""
    df = pd.read_excel(
        io.BytesIO(file_content),
        sheet_name="STANDARD&POOR'S500",
        skiprows=3,
    )
    rows = [
        {
            "ticker": row.get("Ticker"),
            "ethical_evaluation": row.get("Ethical Evaluation"),
        }
        for _, row in df.iterrows()
    ]
    return parse_ethical_filter_rows(rows)

