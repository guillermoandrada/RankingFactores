"""Strategy Backtest - historical simulation across manual period windows."""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from modules.portfolio import parse_ethical_filter_excel
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

STRATEGY_SCHEDULE_ROWS_KEY = "strategy_backtest_schedule_rows"
_LEGACY_SCHEDULE_DF_KEY = "strategy_backtest_schedule_df"


def _percent_input(
    label: str,
    *,
    value: float,
    step: float = 0.01,
    min_value: float | None = 0.0,
    max_value: float | None = None,
    key: str | None = None,
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
                alt.Tooltip("value:Q", title="Value", format=",.1f"),
            ],
        )
    )
    st.altair_chart(chart, width="stretch")


def _render_backtest_result(backtest_result: dict[str, Any]) -> None:
    summary = backtest_result.get("summary", {})
    metrics_cols = st.columns(4)
    metrics_cols[0].metric(
        "Total return",
        _format_metric_value(summary.get("total_return"), percent=True),
    )
    metrics_cols[1].metric(
        "Annualized return",
        _format_metric_value(summary.get("annualized_return"), percent=True),
    )
    metrics_cols[2].metric(
        "Volatility",
        _format_metric_value(summary.get("volatility"), percent=True),
    )
    metrics_cols[3].metric(
        "Max drawdown",
        _format_metric_value(summary.get("max_drawdown"), percent=True),
    )

    series_df = pd.DataFrame(backtest_result.get("series", []))
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

    if backtest_result.get("components"):
        st.markdown("**Component total returns**")
        _render_component_returns(backtest_result.get("components", []))


def _default_schedule_rows(available_periods: list[str]) -> list[dict[str, Any]]:
    today = date.today()
    return [
        {
            "id": str(uuid.uuid4()),
            "period": available_periods[0] if available_periods else "",
            "start_date": today - timedelta(days=365),
            "end_date": today,
        }
    ]


def _migrate_legacy_schedule_df_to_rows(available_periods: list[str]) -> None:
    if STRATEGY_SCHEDULE_ROWS_KEY in st.session_state or _LEGACY_SCHEDULE_DF_KEY not in st.session_state:
        return
    legacy = st.session_state.pop(_LEGACY_SCHEDULE_DF_KEY)
    if not isinstance(legacy, pd.DataFrame) or legacy.empty:
        return
    rows_out: list[dict[str, Any]] = []
    fallback_period = available_periods[0] if available_periods else ""
    for _, r in legacy.iterrows():
        p = str(r.get("period") or "").strip() or fallback_period
        if available_periods and p not in available_periods:
            p = fallback_period
        try:
            sv = r.get("start_date")
            ev = r.get("end_date")
            sd = pd.Timestamp(sv).date() if pd.notna(sv) else date.today() - timedelta(days=365)
            ed = pd.Timestamp(ev).date() if pd.notna(ev) else date.today()
        except (ValueError, TypeError, OSError):
            sd = date.today() - timedelta(days=365)
            ed = date.today()
        rows_out.append(
            {"id": str(uuid.uuid4()), "period": p, "start_date": sd, "end_date": ed}
        )
    if rows_out:
        st.session_state[STRATEGY_SCHEDULE_ROWS_KEY] = rows_out


def _render_strategy_schedule_windows(available_periods: list[str]) -> None:
    st.markdown("**Schedule windows**")
    st.caption("Pick a ranking period and calendar start/end dates for each backtest window.")

    _migrate_legacy_schedule_df_to_rows(available_periods)
    if STRATEGY_SCHEDULE_ROWS_KEY not in st.session_state:
        st.session_state[STRATEGY_SCHEDULE_ROWS_KEY] = _default_schedule_rows(available_periods)

    sched_rows: list[dict[str, Any]] = list(st.session_state[STRATEGY_SCHEDULE_ROWS_KEY])
    for row in sched_rows:
        if "id" not in row:
            row["id"] = str(uuid.uuid4())

    h1, _, _, h4 = st.columns([4, 2, 2, 1])
    h1.caption("Period")
    h4.caption("")

    updated: list[dict[str, Any]] = []
    for row in sched_rows:
        rid = str(row["id"])
        period_val = str(row.get("period") or "")
        if available_periods and period_val not in available_periods:
            period_val = available_periods[0]
        start_val = row.get("start_date")
        end_val = row.get("end_date")
        if not isinstance(start_val, date):
            start_val = date.today() - timedelta(days=365)
        if not isinstance(end_val, date):
            end_val = date.today()

        c1, c2, c3, c4 = st.columns([4, 2, 2, 1])
        with c1:
            p_idx = available_periods.index(period_val) if period_val in available_periods else 0
            period_chosen = st.selectbox(
                "Period",
                options=available_periods,
                index=p_idx,
                key=f"strategy_bt_sched_period_{rid}",
                label_visibility="collapsed",
            )
        with c2:
            start_d = st.date_input(
                "Start date",
                value=start_val,
                key=f"strategy_bt_sched_start_{rid}",
            )
        with c3:
            end_d = st.date_input(
                "End date",
                value=end_val,
                key=f"strategy_bt_sched_end_{rid}",
            )
        with c4:
            if st.button("Remove", key=f"strategy_bt_sched_remove_{rid}"):
                st.session_state[STRATEGY_SCHEDULE_ROWS_KEY] = [
                    item for item in sched_rows if str(item["id"]) != rid
                ]
                st.rerun()
        updated.append(
            {"id": rid, "period": period_chosen, "start_date": start_d, "end_date": end_d}
        )

    st.session_state[STRATEGY_SCHEDULE_ROWS_KEY] = updated

    if st.button("Add window", key="strategy_bt_sched_add"):
        st.session_state[STRATEGY_SCHEDULE_ROWS_KEY].append(
            {
                "id": str(uuid.uuid4()),
                "period": available_periods[0] if available_periods else "",
                "start_date": date.today() - timedelta(days=365),
                "end_date": date.today(),
            }
        )
        st.rerun()


st.set_page_config(page_title="Strategy Backtest", layout="wide")
inject_custom_css()
render_page_header(
    "Strategy Backtest",
    "Run a historical simulation by rebuilding the portfolio across manual period windows.",
)

client = get_api_client("strategy_backtest")
render_sidebar_api_test(client, "strategy_backtest_api")
render_reference_refresh_button("strategy_backtest")

try:
    periods, profiles, sectors, industries, indices = load_reference_data_bundle(
        client,
        cache_key="strategy_backtest",
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
    st.markdown("**Shared portfolio inputs**")
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        scoring_profile = st.selectbox(
            "Scoring profile",
            profile_names,
            key="strategy_backtest_profile",
        )
    with row1_col2:
        index_options = ["(All indices)"] + sorted(indices)
        index_label = st.selectbox("Index", index_options, key="strategy_backtest_index")
        index_choice = "" if index_label == "(All indices)" else index_label
    with row1_col3:
        strategy = st.selectbox(
            "Strategy",
            ["legacy_rebalance", "smart_beta", "long_short"],
            key="strategy_backtest_strategy",
        )

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        sector_options = ["(All sectors)"] + sorted(sectors)
        sector_label = st.selectbox("Sector", sector_options, key="strategy_backtest_sector")
        sector_choice = "" if sector_label == "(All sectors)" else sector_label
    with row2_col2:
        industry_options = ["(All industries)"] + sorted(industries)
        industry_label = st.selectbox("Industry", industry_options, key="strategy_backtest_industry")
        industry_choice = "" if industry_label == "(All industries)" else industry_label
    with row2_col3:
        capital_base = st.number_input(
            "Capital base",
            min_value=0.0,
            value=1.0,
            step=1.0,
            key="strategy_backtest_capital_base",
        )

ethical_filter_rows: list[dict[str, Any]] = []
with st.container(border=True):
    st.markdown("**External filters**")
    ethical_file = st.file_uploader(
        "Ethical filter workbook",
        type=["xlsx", "xls"],
        key="strategy_backtest_ethical_filter_file",
    )
    if ethical_file:
        try:
            blocked = sorted(parse_ethical_filter_excel(ethical_file.read()))
            ethical_filter_rows = [
                {"ticker": ticker, "ethical_evaluation": "No"}
                for ticker in blocked
            ]
            st.caption(f"Blocked tickers loaded: {len(ethical_filter_rows)}")
        except (ValueError, KeyError, OSError) as exc:
            st.error(f"Could not parse ethical filter workbook: {exc}")

st.divider()
with st.container(border=True):
    st.markdown("**Constraint targets**")
    constraint_type = st.radio(
        "Constraint type",
        options=["none", "sector", "industry"],
        horizontal=True,
        key="strategy_backtest_constraint_type",
        format_func=lambda value: {
            "none": "None",
            "sector": "Sector restrictions",
            "industry": "Industry restrictions",
        }[value],
    )
    st.caption(
        "Enabled rows with weight `0.0` are treated as explicit no-buy constraints and are shared across all schedule rows."
    )

    sector_targets_df = pd.DataFrame(columns=["enabled", "group", "weight"])
    industry_targets_df = pd.DataFrame(columns=["enabled", "group", "weight"])
    if constraint_type == "sector":
        st.markdown("**Sector targets**")
        sector_state_key = "strategy_backtest_sector_targets_df"
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Enable all sector restrictions", key="strategy_backtest_sector_enable"):
                set_all_targets_enabled(sector_state_key, True)
                st.rerun()
        with action_col2:
            if st.button("Disable all sector restrictions", key="strategy_backtest_sector_disable"):
                set_all_targets_enabled(sector_state_key, False)
                st.rerun()
        sector_targets_df = render_constraint_target_fields(
            sector_state_key,
            "strategy_backtest_sector_row",
            sectors,
        )
    elif constraint_type == "industry":
        st.markdown("**Industry targets**")
        industry_state_key = "strategy_backtest_industry_targets_df"
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Enable all industry restrictions", key="strategy_backtest_industry_enable"):
                set_all_targets_enabled(industry_state_key, True)
                st.rerun()
        with action_col2:
            if st.button("Disable all industry restrictions", key="strategy_backtest_industry_disable"):
                set_all_targets_enabled(industry_state_key, False)
                st.rerun()
        industry_targets_df = render_constraint_target_fields(
            industry_state_key,
            "strategy_backtest_industry_row",
            industries,
        )

sector_targets = targets_from_dataframe(sector_targets_df) if constraint_type == "sector" else {}
industry_targets = targets_from_dataframe(industry_targets_df) if constraint_type == "industry" else {}

portfolio_request: dict[str, Any] = {
    "scoring_profile": scoring_profile,
    "sector": sector_choice,
    "industry": industry_choice,
    "index": index_choice,
    "strategy": strategy,
    "construction_mode": "new_portfolio",
    "capital_base": float(capital_base),
    "ethical_filter_rows": ethical_filter_rows,
    "constraint_type": constraint_type,
    "sector_targets": sector_targets,
    "industry_targets": industry_targets,
}

st.divider()
with st.container(border=True):
    st.markdown("**Strategy parameters**")
    if strategy == "legacy_rebalance":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            portfolio_request["max_position"] = _percent_input(
                "Max position",
                min_value=0.0,
                value=0.05,
                step=0.01,
                key="strategy_backtest_max_position",
            )
        with c2:
            portfolio_request["neutral_position"] = _percent_input(
                "Neutral position",
                min_value=0.0,
                value=0.03,
                step=0.01,
                key="strategy_backtest_neutral_position",
            )
        with c3:
            portfolio_request["score_quantile_cutoff"] = _percent_input(
                "Score quantile cutoff",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="strategy_backtest_score_quantile_cutoff",
            )
        with c4:
            portfolio_request["min_trade_weight"] = 0.0
    elif strategy == "smart_beta":
        c1, c2 = st.columns(2)
        portfolio_request["top_n"] = int(c1.number_input("Top N (0 = all)", min_value=0, value=0, step=1)) or None
        with c2:
            portfolio_request["smart_beta_max_weight"] = _percent_input(
                "Max weight",
                min_value=0.0,
                value=0.10,
                step=0.01,
                key="strategy_backtest_smart_beta_max_weight",
            )
    else:
        c1, c2, c3 = st.columns(3)
        portfolio_request["bucket_count"] = int(c1.number_input("Bucket count", min_value=2, value=10, step=1))
        portfolio_request["long_bucket_count"] = int(c2.number_input("Long bucket count", min_value=1, value=1, step=1))
        portfolio_request["short_bucket_count"] = int(c3.number_input("Short bucket count", min_value=1, value=1, step=1))
        c4, c5, c6 = st.columns(3)
        portfolio_request["long_short_weighting"] = c4.selectbox("Weighting", ["equal", "score"])
        with c5:
            portfolio_request["gross_exposure"] = _percent_input(
                "Gross exposure",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key="strategy_backtest_gross_exposure",
            )
        with c6:
            portfolio_request["net_exposure"] = _percent_input(
                "Net exposure",
                min_value=None,
                value=0.0,
                step=0.1,
                key="strategy_backtest_net_exposure",
            )

st.divider()
with st.container(border=True):
    st.markdown("**Backtest setup**")
    methodology_col, benchmark_col = st.columns(2)
    with methodology_col:
        methodology = st.selectbox(
            "Backtest methodology",
            ["fixed_weights", "drifting_weights"],
            key="strategy_backtest_methodology",
            format_func=lambda value: {
                "fixed_weights": "Fixed weights",
                "drifting_weights": "Drifting weights",
            }[value],
        )
    with benchmark_col:
        benchmark_ticker = st.text_input(
            "Benchmark ticker (optional)",
            key="strategy_backtest_benchmark",
            placeholder="SPY",
        ).strip().upper()

    _render_strategy_schedule_windows(periods)

run_clicked = st.button("Run strategy backtest", type="primary", key="strategy_backtest_run")
if run_clicked:
    schedule_rows: list[dict[str, Any]] = []
    for sched in st.session_state.get(STRATEGY_SCHEDULE_ROWS_KEY, []):
        period = str(sched.get("period") or "").strip()
        start_d = sched.get("start_date")
        end_d = sched.get("end_date")
        if not period or not isinstance(start_d, date) or not isinstance(end_d, date):
            continue
        schedule_rows.append(
            {
                "period": period,
                "start_date": start_d.isoformat(),
                "end_date": end_d.isoformat(),
            }
        )

    if not schedule_rows:
        st.error("Add at least one backtest window with period, start date, and end date.")
    else:
        try:
            result = client.run_strategy_backtest(
                {
                    "portfolio_request": portfolio_request,
                    "schedule": schedule_rows,
                    "methodology": methodology,
                    "benchmark_ticker": benchmark_ticker,
                }
            )
            st.session_state["strategy_backtest_result"] = result
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))

strategy_backtest_result = st.session_state.get("strategy_backtest_result")
if strategy_backtest_result:
    st.divider()
    st.subheader("Strategy backtest result")
    _render_backtest_result(strategy_backtest_result)

    intervals_df = pd.DataFrame(strategy_backtest_result.get("intervals", []))
    if not intervals_df.empty:
        st.markdown("**Intervals**")
        st.dataframe(intervals_df.drop(columns=["portfolio"], errors="ignore"), width="stretch")
        for interval in strategy_backtest_result.get("intervals", []):
            interval_label = (
                f"{interval.get('period', '')} | "
                f"{interval.get('start_date', '')} to {interval.get('end_date', '')}"
            )
            with st.expander(f"Component returns - {interval_label}"):
                _render_component_returns(interval.get("components", []))

    warnings = strategy_backtest_result.get("warnings", [])
    if warnings:
        st.markdown("**Warnings**")
        for warning in warnings:
            st.warning(warning)

    with st.expander("Backtest raw JSON"):
        st.json(strategy_backtest_result)
