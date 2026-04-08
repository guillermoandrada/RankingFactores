"""Portfolio Construction - build or rebalance portfolios from scoring output."""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from modules.market_data import fetch_latest_adjusted_closes
from modules.portfolio import parse_ethical_filter_excel, parse_holdings_excel
from modules.portfolio.input_parsers import normalize_ticker
from streamlit_app.api_client import ApiError
from streamlit_app.constraint_targets import (
    render_constraint_target_fields,
    set_all_targets_enabled,
    targets_from_dataframe,
)
from streamlit_app.ui import (
    get_api_client,
    inject_custom_css,
    render_page_header,
    render_sidebar_api_test,
)
from streamlit_app.ui.reference_data import (
    load_reference_data_bundle,
    render_reference_refresh_button,
)

_PORTFOLIO_HOLDINGS_CACHE_KEY = "portfolio_holdings_rows_cache"
_PORTFOLIO_HOLDINGS_SIG_KEY = "portfolio_holdings_content_sig"


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


def _merge_yfinance_prices_into_holdings_rows(
    rows: list[dict[str, Any]],
    closes: dict[str, float],
) -> list[dict[str, Any]]:
    """Apply Yahoo Finance closes; skip cash rows; recompute market_value as quantity * price."""
    merged: list[dict[str, Any]] = []
    for row in rows:
        ticker_key = normalize_ticker(row.get("ticker"))
        if ticker_key == "CASH_USD":
            merged.append(dict(row))
            continue
        if ticker_key in closes:
            quantity = float(row.get("quantity") or 0.0)
            price = closes[ticker_key]
            merged.append({**dict(row), "price": price, "market_value": quantity * price})
        else:
            merged.append(dict(row))
    return merged


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


def _value_column_config(columns: list[str]) -> dict[str, Any]:
    return {
        column: st.column_config.NumberColumn(column, format="%.1f")
        for column in columns
    }


def _display_percent_dataframe(
    df: pd.DataFrame,
    percent_columns: list[str],
    *,
    value_columns: list[str] | None = None,
) -> None:
    if df.empty:
        st.dataframe(df, width="stretch")
        return

    display_df = df.copy()
    active_percent = [c for c in percent_columns if c in display_df.columns]
    active_value = [c for c in (value_columns or []) if c in display_df.columns]
    for column in active_percent:
        display_df[column] = pd.to_numeric(display_df[column], errors="coerce") * 100.0

    column_config = {**_percentage_column_config(active_percent), **_value_column_config(active_value)}
    st.dataframe(
        display_df,
        width="stretch",
        column_config=column_config or None,
    )


def _format_metric_value(value: Any, *, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.2%}" if percent else f"{numeric:,.1f}"


def _render_component_returns(components: list[dict[str, Any]]) -> None:
    if not components:
        return
    st.markdown("**Component total returns**")
    component_df = pd.DataFrame(components)
    preferred_columns = [
        "ticker",
        "name",
        "sector",
        "industry",
        "target_weight",
        "resolved_ticker",
        "start_price",
        "end_price",
        "total_return",
        "status",
    ]
    available_columns = [column for column in preferred_columns if column in component_df.columns]
    _display_percent_dataframe(
        component_df[available_columns],
        ["target_weight", "total_return"],
        value_columns=["start_price", "end_price"],
    )


def _render_backtest_chart(series_df: pd.DataFrame) -> None:
    if series_df.empty or "date" not in series_df.columns:
        return
    value_columns = [
        column
        for column in ["portfolio_value", "benchmark_value"]
        if column in series_df.columns
    ]
    if not value_columns:
        return

    chart_df = series_df[["date", *value_columns]].melt(
        id_vars="date",
        value_vars=value_columns,
        var_name="series",
        value_name="value",
    )
    chart_df["series"] = chart_df["series"].map(
        {
            "portfolio_value": "Portfolio value",
            "benchmark_value": "Benchmark value",
        }
    ).fillna(chart_df["series"])

    chart = (
        alt.Chart(chart_df)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Value", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title="Series"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title="Value", format=",.2f"),
            ],
        )
    )
    st.altair_chart(chart, width="stretch")


def _render_backtest_result(result: dict[str, Any]) -> None:
    backtest_summary = result.get("summary", {})
    metrics_cols = st.columns(4)
    metrics_cols[0].metric(
        "Total return",
        _format_metric_value(backtest_summary.get("total_return"), percent=True),
    )
    metrics_cols[1].metric(
        "Annualized return",
        _format_metric_value(backtest_summary.get("annualized_return"), percent=True),
    )
    metrics_cols[2].metric(
        "Volatility",
        _format_metric_value(backtest_summary.get("volatility"), percent=True),
    )
    metrics_cols[3].metric(
        "Max drawdown",
        _format_metric_value(backtest_summary.get("max_drawdown"), percent=True),
    )

    series_df = pd.DataFrame(result.get("series", []))
    if not series_df.empty and "date" in series_df.columns:
        series_df["date"] = pd.to_datetime(series_df["date"])
        _render_backtest_chart(series_df)
        series_df = series_df.set_index("date")
        display_columns = [
            column
            for column in [
                "portfolio_value",
                "portfolio_return",
                "portfolio_cumulative",
                "benchmark_value",
                "benchmark_return",
                "benchmark_cumulative",
                "excess_return",
                "relative_cumulative",
            ]
            if column in series_df.columns
        ]
        _display_percent_dataframe(
            series_df.reset_index()[["date", *display_columns]],
            [
                "portfolio_return",
                "portfolio_cumulative",
                "benchmark_return",
                "benchmark_cumulative",
                "excess_return",
                "relative_cumulative",
            ],
            value_columns=["portfolio_value", "benchmark_value"],
        )

    warnings = result.get("warnings", [])
    if warnings:
        st.markdown("**Warnings**")
        for warning in warnings:
            st.warning(warning)

    _render_component_returns(result.get("components", []))


st.set_page_config(page_title="Portfolio Construction", layout="wide")
inject_custom_css()
render_page_header(
    "Portfolio Construction",
    "Build fresh portfolios or rebalance an existing portfolio using scoring output.",
)

client = get_api_client("portfolio")
render_sidebar_api_test(client, "portfolio_test_api")
render_reference_refresh_button("portfolio")

try:
    periods, profiles, sectors, industries, indices = load_reference_data_bundle(
        client,
        cache_key="portfolio",
    )
    profile_names = sorted(profiles.keys())
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
            "Upload the legacy-style holdings workbook. If price or market value is missing, use "
            "**Fetch latest prices (Yahoo Finance)** or the rebalance API may reject the request."
        )
        holdings_file = st.file_uploader(
            "Current holdings workbook",
            type=["xlsx", "xls"],
            key="portfolio_holdings_file",
        )
        if holdings_file:
            try:
                file_bytes = holdings_file.read()
                content_sig = hashlib.sha256(file_bytes).hexdigest()
                if st.session_state.get(_PORTFOLIO_HOLDINGS_SIG_KEY) != content_sig:
                    parsed_positions, _cash = parse_holdings_excel(file_bytes)
                    holdings_rows = _positions_to_rows(parsed_positions)
                    st.session_state[_PORTFOLIO_HOLDINGS_CACHE_KEY] = holdings_rows
                    st.session_state[_PORTFOLIO_HOLDINGS_SIG_KEY] = content_sig
                else:
                    holdings_rows = list(st.session_state.get(_PORTFOLIO_HOLDINGS_CACHE_KEY, []))

                st.dataframe(pd.DataFrame(holdings_rows), width="stretch")

                fetch_disabled = not holdings_rows
                if st.button(
                    "Fetch latest prices (Yahoo Finance)",
                    key="portfolio_fetch_yf_prices",
                    disabled=fetch_disabled,
                    help="Loads the latest adjusted closes from Yahoo Finance and fills price and market value.",
                ):
                    tickers = [
                        normalize_ticker(row.get("ticker"))
                        for row in holdings_rows
                        if normalize_ticker(row.get("ticker")) not in ("", "CASH_USD")
                    ]
                    if not tickers:
                        st.warning("No equity tickers to price (only cash or empty rows).")
                    else:
                        try:
                            price_result = fetch_latest_adjusted_closes(tickers)
                            updated = _merge_yfinance_prices_into_holdings_rows(
                                holdings_rows,
                                price_result.closes,
                            )
                            st.session_state[_PORTFOLIO_HOLDINGS_CACHE_KEY] = updated
                            holdings_rows = updated
                            if price_result.missing_identifiers:
                                st.warning(
                                    "No Yahoo Finance close for: "
                                    + ", ".join(sorted(price_result.missing_identifiers))
                                )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not fetch prices: {exc}")

                equity_holdings = [
                    row
                    for row in holdings_rows
                    if normalize_ticker(row.get("ticker")) not in ("", "CASH_USD")
                ]
                if equity_holdings and all(
                    row.get("price") in (None, "") and row.get("market_value") in (None, "")
                    for row in equity_holdings
                ):
                    st.warning(
                        "Equity rows still lack price or market value. Fetch prices above or add them in the workbook."
                    )
            except (ValueError, KeyError, OSError) as exc:
                st.error(f"Could not parse holdings workbook: {exc}")
        else:
            st.session_state.pop(_PORTFOLIO_HOLDINGS_CACHE_KEY, None)
            st.session_state.pop(_PORTFOLIO_HOLDINGS_SIG_KEY, None)

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
                st.dataframe(pd.DataFrame(ethical_filter_rows), width="stretch")
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
        sector_button_col1, sector_button_col2 = st.columns(2)
        with sector_button_col1:
            if st.button("Enable all sector restrictions", key="portfolio_sector_targets_enable_all"):
                set_all_targets_enabled(sector_state_key, True)
                st.rerun()
        with sector_button_col2:
            if st.button("Disable all sector restrictions", key="portfolio_sector_targets_disable_all"):
                set_all_targets_enabled(sector_state_key, False)
                st.rerun()
        sector_targets_df = render_constraint_target_fields(
            sector_state_key,
            "portfolio_sector_target_row",
            sectors,
        )
    elif constraint_type == "industry":
        st.markdown("**Industry targets**")
        industry_state_key = "portfolio_industry_targets_df"
        industry_button_col1, industry_button_col2 = st.columns(2)
        with industry_button_col1:
            if st.button("Enable all industry restrictions", key="portfolio_industry_targets_enable_all"):
                set_all_targets_enabled(industry_state_key, True)
                st.rerun()
        with industry_button_col2:
            if st.button("Disable all industry restrictions", key="portfolio_industry_targets_disable_all"):
                set_all_targets_enabled(industry_state_key, False)
                st.rerun()
        industry_targets_df = render_constraint_target_fields(
            industry_state_key,
            "portfolio_industry_target_row",
            industries,
        )

sector_targets = targets_from_dataframe(sector_targets_df) if constraint_type == "sector" else {}
industry_targets = targets_from_dataframe(industry_targets_df) if constraint_type == "industry" else {}

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
        st.session_state.pop("portfolio_backtest_result", None)
        st.rerun()
    except ApiError as exc:
        st.error(str(exc))

portfolio_result = st.session_state.get("portfolio_result")
if portfolio_result:
    st.divider()
    st.subheader("Portfolio result")
    summary = portfolio_result.get("summary", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Capital", _format_metric_value(summary.get("total_capital", 0)))
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
            value_columns=["target_amount"],
        )
    with tab2:
        _display_percent_dataframe(
            pd.DataFrame(portfolio_result.get("current_portfolio", [])),
            ["weight"],
            value_columns=["quantity", "price", "amount"],
        )
    with tab3:
        _display_percent_dataframe(
            pd.DataFrame(portfolio_result.get("trades", [])),
            ["weight_delta", "current_weight", "target_weight"],
            value_columns=["quantity_delta", "price"],
        )
    with tab4:
        st.markdown("**Excluded**")
        st.dataframe(pd.DataFrame(portfolio_result.get("excluded", [])), width="stretch")
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

    st.divider()
    with st.container(border=True):
        st.markdown("**Portfolio backtest**")
        today = date.today()
        default_start = today - timedelta(days=365)
        backtest_col1, backtest_col2 = st.columns(2)
        with backtest_col1:
            backtest_start = st.date_input(
                "Backtest start date",
                value=default_start,
                key="portfolio_backtest_start",
            )
        with backtest_col2:
            backtest_end = st.date_input(
                "Backtest end date",
                value=today,
                key="portfolio_backtest_end",
            )

        backtest_col3, backtest_col4 = st.columns(2)
        with backtest_col3:
            methodology = st.selectbox(
                "Backtest methodology",
                options=["fixed_weights", "drifting_weights"],
                key="portfolio_backtest_methodology",
                format_func=lambda value: {
                    "fixed_weights": "Fixed weights",
                    "drifting_weights": "Drifting weights",
                }[value],
            )
        with backtest_col4:
            benchmark_ticker = st.text_input(
                "Benchmark ticker (optional)",
                key="portfolio_backtest_benchmark",
                placeholder="SPY",
            ).strip().upper()

        if st.button("Run portfolio backtest", key="portfolio_run_backtest"):
            if backtest_end < backtest_start:
                st.error("Backtest end date must be on or after start date.")
            else:
                try:
                    backtest_result = client.run_portfolio_backtest(
                        {
                            "portfolio": portfolio_result.get("portfolio", []),
                            "capital_base": float(summary.get("total_capital") or 1.0),
                            "start_date": backtest_start.isoformat(),
                            "end_date": backtest_end.isoformat(),
                            "methodology": methodology,
                            "benchmark_ticker": benchmark_ticker,
                        }
                    )
                    st.session_state["portfolio_backtest_result"] = backtest_result
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))

        portfolio_backtest_result = st.session_state.get("portfolio_backtest_result")
        if portfolio_backtest_result:
            _render_backtest_result(portfolio_backtest_result)
            with st.expander("Backtest raw JSON"):
                st.json(portfolio_backtest_result)

