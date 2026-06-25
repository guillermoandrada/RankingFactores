"""RankingFactores — navigation entry point."""

from __future__ import annotations

import streamlit as st

from streamlit_app.client.api_client import ApiError
from streamlit_app.ui import get_api_client, inject_custom_css, render_page_header

st.set_page_config(page_title="RankingFactores", layout="wide")
inject_custom_css()


def _home() -> None:
    render_page_header("RankingFactores", "Financial data ranking and scoring profile builder.")

    client = get_api_client("home")

    try:
        stats = client.get_stats()

        period_count = stats.get("period_count", 0)
        total_db_metrics = stats.get("total_db_metrics", 0)
        total_derived = stats.get("total_derived_metrics", 0)
        total_profiles = stats.get("total_scoring_profiles", 0)
        coverage = stats.get("coverage", [])

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Periods", period_count)
        col2.metric("Securities (latest)", coverage[0]["securities"] if coverage else 0)
        col3.metric("DB Metrics", total_db_metrics)
        col4.metric("Derived Metrics", total_derived)
        col5.metric("Scoring Profiles", total_profiles)

        if coverage:
            st.subheader("Period Coverage")
            st.dataframe(
                coverage,
                column_config={
                    "period": st.column_config.TextColumn("Period"),
                    "securities": st.column_config.NumberColumn("Securities"),
                    "metrics": st.column_config.NumberColumn("Metrics in Period"),
                },
                width="stretch",
                hide_index=True,
            )

    except ApiError:
        st.warning("API is not reachable. Start the backend with `uvicorn api.main:app --reload`.")

    st.divider()
    st.subheader("Workflow")
    st.markdown("""
1. **Periods** — Upload Excel data, create periods, view and edit content.
2. **Derived Metrics** — Create derived metrics (e.g. Debt/Assets) from existing metrics.
3. **Metric Selection (IC)** — Rank IC vs forward returns and inter-factor correlation.
4. **Scoring Profile Wizard** — Build scoring methodologies step-by-step.
5. **Scoring Profiles** — Edit and manage saved scoring profiles.
6. **Ranking** — Run rankings for a period and profile, export results.
7. **Portfolio Construction** — Build or rebalance portfolios from ranking output.
8. **Backtest Strategy** — Rebuild portfolios across manual date windows and compare performance versus a benchmark.
""")


pg = st.navigation(
    {
        "": [
            st.Page(_home, title="Home", default=True),
            st.Page("pages/1_Periods.py", title="Periods"),
        ],
        "Metrics": [
            st.Page("pages/2_Metrics.py", title="Derived Metrics"),
            st.Page("pages/6_Metric_Selection.py", title="Metric Selection"),
            st.Page("pages/8_Metric_Diagnostics.py", title="Metric Diagnostics"),
        ],
        "Scoring": [
            st.Page("pages/3_Scoring_Profile_Wizard.py", title="Scoring Profile Wizard"),
            st.Page("pages/4_Scoring_Profiles.py", title="Scoring Profiles"),
        ],
        "Strategy development and testing": [
            st.Page("pages/5_Ranking.py", title="Ranking"),
            st.Page("pages/6_Portfolio_Construction.py", title="Portfolio Construction"),
            st.Page("pages/7_Backtest_Strategy.py", title="Backtest Strategy"),
            st.Page("pages/9_Price_Data.py", title="Price Data"),
        ],
    }
)
pg.run()
