from __future__ import annotations

import pandas as pd

from modules.db.repository import FinancialDatabase


def test_get_period_content_includes_period_scoped_name_sector_and_industry(tmp_path) -> None:
    db_path = tmp_path / "period_content.db"
    db = FinancialDatabase(db_url=f"sqlite:///{db_path}")

    df_period_1 = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "Long Name": ["AAA Corp"],
            "GICS Sector Name": ["Tech"],
            "GICS Industry Group Name": ["Software"],
            "Market Cap (USD)": [1000.0],
            "Metric A": [1.0],
        }
    )
    df_period_2 = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "Long Name": ["AAA Corp"],
            "GICS Sector Name": ["Financials"],
            "GICS Industry Group Name": ["Banks"],
            "Market Cap (USD)": [1000.0],
            "Metric A": [2.0],
        }
    )

    db.save_fundamentals(df_period_1, "2024 Q1", mode="replace")
    db.save_fundamentals(df_period_2, "2024 Q2", mode="replace")

    period_1 = db.get_period_content("2024 Q1")
    period_2 = db.get_period_content("2024 Q2")

    assert period_1["metrics"] == ["Metric A"]
    assert period_1["data"][0]["ticker"] == "AAA"
    assert period_1["data"][0]["name"] == "AAA Corp"
    assert period_1["data"][0]["sector"] == "Tech"
    assert period_1["data"][0]["industry"] == "Software"

    assert period_2["data"][0]["ticker"] == "AAA"
    assert period_2["data"][0]["name"] == "AAA Corp"
    assert period_2["data"][0]["sector"] == "Financials"
    assert period_2["data"][0]["industry"] == "Banks"
