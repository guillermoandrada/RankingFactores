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
        if not bool(row.get("enabled", False)):
            continue
        group = str(row.get("group") or "").strip()
        if not group:
            continue
        try:
            weight = float(row.get("weight") or 0.0) / 100.0
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


def _prefilled_targets(options: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"enabled": False, "group": option, "weight": 0.0} for option in sorted(options)],
        columns=["enabled", "group", "weight"],
    )


def _ensure_targets_state(state_key: str, options: list[str]) -> pd.DataFrame:
    expected = _prefilled_targets(options)
    current = st.session_state.get(state_key)
    if not isinstance(current, pd.DataFrame):
        st.session_state[state_key] = expected
        return expected

    current_map = {
        str(row.get("group") or "").strip(): row
        for _, row in current.iterrows()
        if str(row.get("group") or "").strip()
    }
    rows: list[dict[str, Any]] = []
    for group in expected["group"].tolist():
        existing = current_map.get(group, {})
        rows.append({
            "enabled": bool(existing.get("enabled", False)),
            "group": group,
            "weight": float(existing.get("weight", 0.0) or 0.0),
        })
    refreshed = pd.DataFrame(rows, columns=["enabled", "group", "weight"])
    st.session_state[state_key] = refreshed
    return refreshed


def _set_all_targets_enabled(state_key: str, enabled: bool) -> None:
    current = st.session_state.get(state_key)
    if not isinstance(current, pd.DataFrame) or current.empty:
        return
    updated = current.copy()
    updated["enabled"] = enabled
    st.session_state[state_key] = updated


def _percent_input(
    label: str,
    *,
    value: float,
    step: float = 0.01,
    min_value: float | None = 0.0,
    max_value: float | None = None,
    key: str | None = None,
    help_text: str | None = None,
) -> float:
    min_percent = None if min_value is None else min_value * 100.0
    max_percent = None if max_value is None else max_value * 100.0
    raw_value = st.number_input(
        f"{label} (%)",
        min_value=min_percent,
        max_value=max_percent,
        value=value * 100.0,
        step=step * 100.0,
        format="%.2f",
        key=key,
        help=help_text,
    )
    return float(raw_value) / 100.0


def _percentage_column_config(columns: list[str]) -> dict[str, Any]:
    return {
        column: st.column_config.NumberColumn(
            column,
            format="%.2f%%",
        )
        for column in columns
    }


def _display_percent_dataframe(
    df: pd.DataFrame,
    percent_columns: list[str],
) -> None:
    if df.empty:
        st.dataframe(df, use_container_width=True)
        return

    display_df = df.copy()
    active_columns = [column for column in percent_columns if column in display_df.columns]
    for column in active_columns:
        display_df[column] = pd.to_numeric(display_df[column], errors="coerce") * 100.0
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config=_percentage_column_config(active_columns),
    )


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
with st.container(border=True):
    st.markdown("**Constraint targets**")
    constraint_type = st.radio(
        "Constraint type",
        options=["none", "sector", "industry"],
        horizontal=True,
        key="portfolio_constraint_type",
        format_func=lambda value: {
            "none": "None",
            "sector": "Sector restrictions",
            "industry": "Industry restrictions",
        }[value],
    )
    st.caption(
        "Enable rows you want to constrain. Enabled rows with weight `0.0` mean no new buys for that group. "
        "Disabled rows are treated as unrestricted."
    )

    sector_targets_df = pd.DataFrame(columns=["enabled", "group", "weight"])
    industry_targets_df = pd.DataFrame(columns=["enabled", "group", "weight"])
    if constraint_type == "sector":
        st.markdown("**Sector targets**")
        sector_state_key = "portfolio_sector_targets_df"
        _ensure_targets_state(sector_state_key, sectors)
        sector_button_col1, sector_button_col2 = st.columns(2)
        with sector_button_col1:
            if st.button("Enable all sector restrictions", key="portfolio_sector_targets_enable_all"):
                _set_all_targets_enabled(sector_state_key, True)
                st.rerun()
        with sector_button_col2:
            if st.button("Disable all sector restrictions", key="portfolio_sector_targets_disable_all"):
                _set_all_targets_enabled(sector_state_key, False)
                st.rerun()
        sector_targets_df = st.data_editor(
            st.session_state[sector_state_key],
            num_rows="fixed",
            key="portfolio_sector_targets",
            use_container_width=True,
            disabled=["group"],
            column_config={
                "enabled": st.column_config.CheckboxColumn("enabled"),
                "group": st.column_config.TextColumn("group"),
                "weight": st.column_config.NumberColumn("weight (%)", format="%.2f"),
            },
        )
        st.session_state[sector_state_key] = sector_targets_df
    elif constraint_type == "industry":
        st.markdown("**Industry targets**")
        industry_state_key = "portfolio_industry_targets_df"
        _ensure_targets_state(industry_state_key, industries)
        industry_button_col1, industry_button_col2 = st.columns(2)
        with industry_button_col1:
            if st.button("Enable all industry restrictions", key="portfolio_industry_targets_enable_all"):
                _set_all_targets_enabled(industry_state_key, True)
                st.rerun()
        with industry_button_col2:
            if st.button("Disable all industry restrictions", key="portfolio_industry_targets_disable_all"):
                _set_all_targets_enabled(industry_state_key, False)
                st.rerun()
        industry_targets_df = st.data_editor(
            st.session_state[industry_state_key],
            num_rows="fixed",
            key="portfolio_industry_targets",
            use_container_width=True,
            disabled=["group"],
            column_config={
                "enabled": st.column_config.CheckboxColumn("enabled"),
                "group": st.column_config.TextColumn("group"),
                "weight": st.column_config.NumberColumn("weight (%)", format="%.2f"),
            },
        )
        st.session_state[industry_state_key] = industry_targets_df

sector_targets = _targets_from_editor(sector_targets_df) if constraint_type == "sector" else {}
industry_targets = _targets_from_editor(industry_targets_df) if constraint_type == "industry" else {}

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
        "constraint_type": constraint_type,
        "sector_targets": sector_targets,
        "industry_targets": industry_targets,
    }

    if strategy == "legacy_rebalance":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            body["max_position"] = _percent_input(
                "Max position",
                min_value=0.0,
                value=0.05,
                step=0.01,
                key="portfolio_max_position_pct",
            )
        with c2:
            body["neutral_position"] = _percent_input(
                "Neutral position",
                min_value=0.0,
                value=0.03,
                step=0.01,
                key="portfolio_neutral_position_pct",
            )
        with c3:
            body["score_quantile_cutoff"] = _percent_input(
                "Score quantile cutoff",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="portfolio_score_quantile_cutoff_pct",
            )
        with c4:
            body["min_trade_weight"] = _percent_input(
                "Min trade weight",
                min_value=0.0,
                value=0.0,
                step=0.001,
                key="portfolio_min_trade_weight_pct",
            )
    elif strategy == "smart_beta":
        c1, c2 = st.columns(2)
        top_n = c1.number_input("Top N (0 = all)", min_value=0, value=0, step=1)
        body["top_n"] = int(top_n) or None
        with c2:
            body["smart_beta_max_weight"] = _percent_input(
                "Max weight",
                min_value=0.0,
                value=0.10,
                step=0.01,
                key="portfolio_smart_beta_max_weight_pct",
            )
    else:
        c1, c2, c3 = st.columns(3)
        body["bucket_count"] = int(c1.number_input("Bucket count", min_value=2, value=10, step=1))
        body["long_bucket_count"] = int(c2.number_input("Long bucket count", min_value=1, value=1, step=1))
        body["short_bucket_count"] = int(c3.number_input("Short bucket count", min_value=1, value=1, step=1))
        c4, c5, c6 = st.columns(3)
        body["long_short_weighting"] = c4.selectbox("Weighting", ["equal", "score"])
        with c5:
            body["gross_exposure"] = _percent_input(
                "Gross exposure",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key="portfolio_gross_exposure_pct",
            )
        with c6:
            body["net_exposure"] = _percent_input(
                "Net exposure",
                min_value=None,
                value=0.0,
                step=0.1,
                key="portfolio_net_exposure_pct",
            )

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
        _display_percent_dataframe(
            pd.DataFrame(portfolio_result.get("portfolio", [])),
            ["current_weight", "target_weight"],
        )
    with tab2:
        _display_percent_dataframe(
            pd.DataFrame(portfolio_result.get("current_portfolio", [])),
            ["weight"],
        )
    with tab3:
        _display_percent_dataframe(
            pd.DataFrame(portfolio_result.get("trades", [])),
            ["weight_delta", "current_weight", "target_weight"],
        )
    with tab4:
        st.markdown("**Excluded**")
        st.dataframe(pd.DataFrame(portfolio_result.get("excluded", [])), use_container_width=True)
        st.markdown("**Constraint diagnostics**")
        sector_diag = portfolio_result.get("constraint_diagnostics", {}).get("sector", [])
        industry_diag = portfolio_result.get("constraint_diagnostics", {}).get("industry", [])
        if sector_diag:
            st.caption("Sector")
            _display_percent_dataframe(
                pd.DataFrame(sector_diag),
                ["target_weight", "actual_weight", "difference"],
            )
        if industry_diag:
            st.caption("Industry")
            _display_percent_dataframe(
                pd.DataFrame(industry_diag),
                ["target_weight", "actual_weight", "difference"],
            )
        notes = portfolio_result.get("notes", [])
        if notes:
            st.markdown("**Notes**")
            for note in notes:
                st.caption(note)
    with tab5:
        st.json(portfolio_result)

