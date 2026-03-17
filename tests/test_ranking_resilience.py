from __future__ import annotations

import pandas as pd

from modules.analytics.zscore import ZScoreCalculator
from modules.db.repository import FinancialDatabase


def test_zscore_calculator_warns_and_zeroes_completely_missing_metric(tmp_path) -> None:
    db_path = tmp_path / "ranking_resilience.db"
    db = FinancialDatabase(db_url=f"sqlite:///{db_path}")

    df = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Long Name": ["AAA Corp", "BBB Corp"],
            "GICS Sector Name": ["Tech", "Tech"],
            "GICS Industry Group Name": ["Software", "Software"],
            "Market Cap (USD)": [1000.0, 2000.0],
            "Current FCF/Net Income": [None, None],
            "Current ROA": [1.0, 2.0],
        }
    )
    db.save_fundamentals(df, "2024 Q1", mode="replace")

    calculator = ZScoreCalculator(engine=db.engine)
    result, _ = calculator.compute(
        period="2024 Q1",
        metric_names=["Current FCF/Net Income", "Current ROA"],
        industry_name="Software",
        out_suffix="_zscore",
    )

    assert result["Current FCF/Net Income_zscore"].tolist() == [0.0, 0.0]
    assert "warnings" in result.attrs
    assert any(
        "Current FCF/Net Income" in warning
        for warning in result.attrs["warnings"]
    )


def test_repository_normalizes_and_dedupes_industry_labels(tmp_path) -> None:
    db_path = tmp_path / "normalized_labels.db"
    db = FinancialDatabase(db_url=f"sqlite:///{db_path}")

    first = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "Long Name": ["AAA Corp"],
            "GICS Sector Name": ["Industrials"],
            "GICS Industry Group Name": ["Commercial & Professional Services"],
            "Market Cap (USD)": [1000.0],
            "Metric A": [1.0],
        }
    )
    second = pd.DataFrame(
        {
            "Ticker": ["BBB"],
            "Long Name": ["BBB Corp"],
            "GICS Sector Name": ["Industrials"],
            "GICS Industry Group Name": ["Commercial  &\xa0Professional   Services "],
            "Market Cap (USD)": [1000.0],
            "Metric A": [2.0],
        }
    )

    db.save_fundamentals(first, "2024 Q1", mode="replace")
    db.save_fundamentals(second, "2024 Q2", mode="replace")

    assert db.list_industries() == ["Commercial & Professional Services"]


def test_missing_metric_tracking_does_not_zero_valid_derived_metric(tmp_path) -> None:
    db_path = tmp_path / "derived_metric_presence.db"
    db = FinancialDatabase(db_url=f"sqlite:///{db_path}")

    df = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Long Name": ["AAA Corp", "BBB Corp"],
            "GICS Sector Name": ["Financials", "Financials"],
            "GICS Industry Group Name": ["Banks", "Banks"],
            "Market Cap (USD)": [1000.0, 2000.0],
            "Current Debt/Assets": [10.0, 12.0],
            "5Y Average Debt/Assets": [8.0, 9.5],
        }
    )
    db.save_fundamentals(df, "2024 Q1", mode="replace")

    calculator = ZScoreCalculator(engine=db.engine)
    result, _ = calculator.compute(
        period="2024 Q1",
        metric_names=["Past Debt/Assets"],
        industry_name="Banks",
        out_suffix="_zscore",
    )

    assert all(value > 0 for value in result["Past Debt/Assets_winsor"].tolist())
    assert result["Past Debt/Assets_zscore"].tolist() != [0.0, 0.0]
    assert not any(
        "Past Debt/Assets" in warning
        for warning in result.attrs.get("warnings", [])
    )
