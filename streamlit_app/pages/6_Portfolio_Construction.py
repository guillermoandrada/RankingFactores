"""Portfolio Construction - build or rebalance portfolios from scoring output."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from modules.portfolio import parse_ethical_filter_excel, parse_holdings_excel
from streamlit_app.api_client import ApiError
from streamlit_app.ui import (
    get_api_client,
    inject_custom_css,
    render_page_header,
    render_sidebar_api_test,
)


def _targets_from_editor(df: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    if df.empty:
        return result
    for _, row in df.iterrows():
        group = str(row.get("group") or "").strip()
        if not group:
            continue
        try:
            weight = float(row.get("weight") or 0.0)
        except (TypeError, ValueError):
            continue
        result[group] = weight
    return result


def _positions_to_rows(positions) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in positions:
        rows.append({
            "ticker": position.ticker,
            "quantity": position.quantity,
            "price": position.price,
            "market_value": position.market_value,
            "name": position.name,
        })
    return rows


st.set_page_config(page_title="Portfolio Construction", layout="wide")
inject_custom_css()
render_page_header(
    "Portfolio Construction",
    "Build fresh portfolios or rebalance an existing portfolio using scoring output.",
)

client = get_api_client("portfolio")
render_sidebar_api_test(client, "portfolio_test_api")

try:
    periods = client.list_periods()
    profiles = client.list_scoring_profiles()
    profile_names = sorted(profiles.keys())
    sectors = client.list_sectors()
    industries = client.list_industries()
    indices = client.list_indices()
except ApiError as exc:
    st.error(f"Cannot load data: {exc}")
    periods = []
    profile_names = []
    sectors = []
    industries = []
    indices = []

if not periods:
    st.info("No periods found. Upload data via the Periods page.")
    st.stop()
if not profile_names:
    st.info("No scoring profiles found. Create one via the Scoring Profile Wizard.")
    st.stop()

st.divider()
with st.container(border=True):
    st.markdown("**Portfolio inputs**")
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        period = st.selectbox("Period", periods, key="portfolio_period")
    with row1_col2:
        scoring_profile = st.selectbox(
            "Scoring profile",
            profile_names,
            key="portfolio_profile",
        )
    with row1_col3:
        index_options = ["(All indices)"] + sorted(indices)
        index_label = st.selectbox("Index", index_options, key="portfolio_index")
        index_choice = "" if index_label == "(All indices)" else index_label

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        sector_options = ["(All sectors)"] + sorted(sectors)
        sector_label = st.selectbox("Sector", sector_options, key="portfolio_sector")
        sector_choice = "" if sector_label == "(All sectors)" else sector_label
    with row2_col2:
        industry_options = ["(All industries)"] + sorted(industries)
        industry_label = st.selectbox("Industry", industry_options, key="portfolio_industry")
        industry_choice = "" if industry_label == "(All industries)" else industry_label
    with row2_col3:
        strategy = st.selectbox(
            "Strategy",
            ["legacy_rebalance", "smart_beta", "long_short"],
            key="portfolio_strategy",
        )

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        construction_mode = st.selectbox(
            "Construction mode",
            ["new_portfolio", "rebalance_existing"],
            key="portfolio_mode",
        )
    with row3_col2:
        capital_base = st.number_input(
            "Capital base",
            min_value=0.0,
            value=1.0,
            step=1.0,
            key="portfolio_capital_base",
            help="Used for new portfolios. Rebalance mode uses uploaded holdings value plus cash.",
        )

st.divider()

holdings_rows: list[dict[str, Any]] = []
if construction_mode == "rebalance_existing":
    with st.container(border=True):
        st.markdown("**Current portfolio input**")
        st.caption(
            "Upload the legacy-style holdings workbook. If price or market value is missing, the rebalance API may reject the request."
        )
        holdings_file = st.file_uploader(
            "Current holdings workbook",
            type=["xlsx", "xls"],
            key="portfolio_holdings_file",
        )
        if holdings_file:
            try:
                parsed_positions, _cash = parse_holdings_excel(holdings_file.read())
                holdings_rows = _positions_to_rows(parsed_positions)
                st.dataframe(pd.DataFrame(holdings_rows), use_container_width=True)
                if holdings_rows and all(row.get("price") in (None, "") and row.get("market_value") in (None, "") for row in holdings_rows):
                    st.warning(
                        "Uploaded holdings do not include price or market value. Rebalance mode needs valuation data to compute current weights."
                    )
            except (ValueError, KeyError, OSError) as exc:
                st.error(f"Could not parse holdings workbook: {exc}")

ethical_filter_rows: list[dict[str, Any]] = []
with st.container(border=True):
    st.markdown("**External filters**")
    ethical_file = st.file_uploader(
        "Ethical filter workbook",
        type=["xlsx", "xls"],
        key="portfolio_ethical_filter_file",
    )
    if ethical_file:
        try:
            blocked = sorted(parse_ethical_filter_excel(ethical_file.read()))
            ethical_filter_rows = [
                {"ticker": ticker, "ethical_evaluation": "No"}
                for ticker in blocked
            ]
            st.caption(f"Blocked tickers loaded: {len(ethical_filter_rows)}")
            if ethical_filter_rows:
                st.dataframe(pd.DataFrame(ethical_filter_rows), use_container_width=True)
        except (ValueError, KeyError, OSError) as exc:
            st.error(f"Could not parse ethical filter workbook: {exc}")

st.divider()
col_targets_1, col_targets_2 = st.columns(2)
with col_targets_1:
    st.markdown("**Sector targets**")
    sector_targets_df = st.data_editor(
        pd.DataFrame(columns=["group", "weight"]),
        num_rows="dynamic",
        key="portfolio_sector_targets",
        use_container_width=True,
    )
with col_targets_2:
    st.markdown("**Industry targets**")
    industry_targets_df = st.data_editor(
        pd.DataFrame(columns=["group", "weight"]),
        num_rows="dynamic",
        key="portfolio_industry_targets",
        use_container_width=True,
    )

sector_targets = _targets_from_editor(sector_targets_df)
industry_targets = _targets_from_editor(industry_targets_df)

st.divider()
with st.container(border=True):
    st.markdown("**Strategy parameters**")
    body: dict[str, Any] = {
        "scoring_profile": scoring_profile,
        "sector": sector_choice,
        "industry": industry_choice,
        "index": index_choice,
        "strategy": strategy,
        "construction_mode": construction_mode,
        "capital_base": float(capital_base),
        "current_holdings": holdings_rows,
        "ethical_filter_rows": ethical_filter_rows,
        "sector_targets": sector_targets,
        "industry_targets": industry_targets,
    }

    if strategy == "legacy_rebalance":
        c1, c2, c3, c4 = st.columns(4)
        body["max_position"] = c1.number_input("Max position", min_value=0.0, value=0.05, step=0.01)
        body["neutral_position"] = c2.number_input("Neutral position", min_value=0.0, value=0.03, step=0.01)
        body["score_quantile_cutoff"] = c3.number_input("Score quantile cutoff", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        body["min_trade_weight"] = c4.number_input("Min trade weight", min_value=0.0, value=0.0, step=0.001)
    elif strategy == "smart_beta":
        c1, c2 = st.columns(2)
        top_n = c1.number_input("Top N (0 = all)", min_value=0, value=0, step=1)
        body["top_n"] = int(top_n) or None
        body["smart_beta_max_weight"] = c2.number_input("Max weight", min_value=0.0, value=0.10, step=0.01)
    else:
        c1, c2, c3 = st.columns(3)
        body["bucket_count"] = int(c1.number_input("Bucket count", min_value=2, value=10, step=1))
        body["long_bucket_count"] = int(c2.number_input("Long bucket count", min_value=1, value=1, step=1))
        body["short_bucket_count"] = int(c3.number_input("Short bucket count", min_value=1, value=1, step=1))
        c4, c5, c6 = st.columns(3)
        body["long_short_weighting"] = c4.selectbox("Weighting", ["equal", "score"])
        body["gross_exposure"] = c5.number_input("Gross exposure", min_value=0.0, value=1.0, step=0.1)
        body["net_exposure"] = c6.number_input("Net exposure", value=0.0, step=0.1)

run_clicked = st.button("Build portfolio", type="primary", key="portfolio_run")
if run_clicked:
    try:
        run_result = client.construct_portfolio(period, body)
        st.session_state["portfolio_result"] = run_result
        st.rerun()
    except ApiError as exc:
        st.error(str(exc))

portfolio_result = st.session_state.get("portfolio_result")
if portfolio_result:
    st.divider()
    st.subheader("Portfolio result")
    summary = portfolio_result.get("summary", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Capital", summary.get("total_capital", 0))
    m2.metric("Positions", summary.get("position_count", 0))
    m3.metric("Trades", summary.get("trade_count", 0))
    m4.metric("Excluded", summary.get("excluded_count", 0))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Portfolio", "Current", "Trades", "Diagnostics", "Raw JSON"]
    )
    with tab1:
        st.dataframe(pd.DataFrame(portfolio_result.get("portfolio", [])), use_container_width=True)
    with tab2:
        st.dataframe(pd.DataFrame(portfolio_result.get("current_portfolio", [])), use_container_width=True)
    with tab3:
        st.dataframe(pd.DataFrame(portfolio_result.get("trades", [])), use_container_width=True)
    with tab4:
        st.markdown("**Excluded**")
        st.dataframe(pd.DataFrame(portfolio_result.get("excluded", [])), use_container_width=True)
        st.markdown("**Constraint diagnostics**")
        sector_diag = portfolio_result.get("constraint_diagnostics", {}).get("sector", [])
        industry_diag = portfolio_result.get("constraint_diagnostics", {}).get("industry", [])
        if sector_diag:
            st.caption("Sector")
            st.dataframe(pd.DataFrame(sector_diag), use_container_width=True)
        if industry_diag:
            st.caption("Industry")
            st.dataframe(pd.DataFrame(industry_diag), use_container_width=True)
        notes = portfolio_result.get("notes", [])
        if notes:
            st.markdown("**Notes**")
            for note in notes:
                st.caption(note)
    with tab5:
        st.json(portfolio_result)

