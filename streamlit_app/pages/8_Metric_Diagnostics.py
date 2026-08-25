"""Metric Diagnostics — distribution shape and outlier detection per period."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import and_, case, func, select

from modules.infrastructure.db import FinancialDatabase
from streamlit_app.ui import (
    read_result,
    render_page_header,
    render_result_caption,
    store_result,
)

_DIAGNOSTICS_RESULT_KEY = "metric_diagnostics_result"


# ---------- Data loading ----------

def _metric_id_by_name(db: FinancialDatabase) -> dict[str, int]:
    metrics = db.list_metrics()
    out: dict[str, int] = {}
    for m in metrics:
        name = (m.get("metric_name") or "").strip()
        mid = m.get("metric_id")
        if name and isinstance(mid, int):
            out[name] = mid
    return out


def _load_period_counts(db: FinancialDatabase, *, metric_id: int) -> pd.DataFrame:
    meta = db._metadata
    tbl = meta.tables["fundamental_values"]
    na_count = func.sum(case((tbl.c.value.is_(None), 1), else_=0)).label("na_count")
    total = func.count(tbl.c.id).label("total")
    q = (
        select(tbl.c.period.label("period"), total, na_count)
        .where(tbl.c.metric_id == int(metric_id))
        .group_by(tbl.c.period)
        .order_by(tbl.c.period)
    )
    with db.engine.connect() as conn:
        df = pd.read_sql_query(q, con=conn)
    if df.empty:
        return df
    df["period"] = df["period"].astype(str)
    df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0).astype(int)
    df["na_count"] = pd.to_numeric(df["na_count"], errors="coerce").fillna(0).astype(int)
    return df


def _load_metric_values(db: FinancialDatabase, *, metric_id: int) -> pd.DataFrame:
    meta = db._metadata
    tbl = meta.tables["fundamental_values"]
    q = (
        select(tbl.c.period.label("period"), tbl.c.value.label("value"))
        .where(and_(tbl.c.metric_id == int(metric_id), tbl.c.value.is_not(None)))
        .order_by(tbl.c.period)
    )
    with db.engine.connect() as conn:
        df = pd.read_sql_query(q, con=conn)
    if df.empty:
        return df
    df["period"] = df["period"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["period", "value"])
    return df


def _load_metric_overall_mean(db: FinancialDatabase, *, metric_id: int) -> float | None:
    meta = db._metadata
    tbl = meta.tables["fundamental_values"]
    q = select(func.avg(tbl.c.value)).where(tbl.c.metric_id == int(metric_id))
    with db.engine.connect() as conn:
        val = conn.execute(q).scalar()
    return float(val) if val is not None else None


def _load_metric_values_with_group(
    db: FinancialDatabase, *, metric_id: int, group_by: str
) -> pd.DataFrame:
    if group_by not in {"sector", "industry"}:
        raise ValueError("group_by must be 'sector' or 'industry'.")

    meta = db._metadata
    tbl_fund = meta.tables["fundamental_values"]
    tbl_sec = meta.tables["securities"]
    tbl_class = meta.tables.get("security_classification")
    tbl_sector = meta.tables.get("sectors")
    tbl_industry = meta.tables.get("industries")
    if tbl_class is None or tbl_sector is None or tbl_industry is None:
        raise RuntimeError("Missing classification tables in database metadata.")

    if group_by == "sector":
        label_col = tbl_sector.c.sector_name
        join_label = tbl_sector.c.sector_id == func.coalesce(
            tbl_class.c.sector_id, tbl_sec.c.sector_id
        )
    else:
        label_col = tbl_industry.c.industry_name
        join_label = tbl_industry.c.industry_id == func.coalesce(
            tbl_class.c.industry_id, tbl_sec.c.industry_id
        )

    q = (
        select(
            tbl_fund.c.period.label("period"),
            label_col.label("group"),
            tbl_fund.c.value.label("value"),
        )
        .select_from(tbl_fund)
        .join(tbl_sec, tbl_sec.c.id == tbl_fund.c.security_id)
        .outerjoin(
            tbl_class,
            and_(
                tbl_class.c.security_id == tbl_fund.c.security_id,
                tbl_class.c.period == tbl_fund.c.period,
            ),
        )
        .outerjoin(
            tbl_sector if group_by == "sector" else tbl_industry,
            join_label,
        )
        .where(tbl_fund.c.metric_id == int(metric_id))
        .order_by(tbl_fund.c.period)
    )
    with db.engine.connect() as conn:
        df = pd.read_sql_query(q, con=conn)
    if df.empty:
        return df
    df["period"] = df["period"].astype(str)
    df["group"] = df["group"].astype("string").fillna("Unknown").astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


# ---------- Stats ----------

_QUANTILE_LEVELS = [0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99]

_GROUP_BY_AGGREGATION = {"By sector": "sector", "By industry": "industry"}


def _stats_for_series(v: pd.Series) -> dict:
    v = v.dropna()
    n = int(v.size)
    if n == 0:
        return {}
    median = float(v.median())
    mean = float(v.mean())
    std = float(v.std(ddof=0)) if n >= 2 else 0.0
    mad = float((v - median).abs().median())
    std_mad = std / mad if mad > 0 else np.nan
    skew_idx = (mean - median) / mad if mad > 0 else np.nan

    q = v.quantile(_QUANTILE_LEVELS, interpolation="linear").to_dict()
    iqr = q[0.75] - q[0.25]
    if iqr > 0:
        lower = q[0.25] - 1.5 * iqr
        upper = q[0.75] + 1.5 * iqr
        n_above = int((v > upper).sum())
        n_below = int((v < lower).sum())
        pct_above = n_above / n * 100.0
        pct_below = n_below / n * 100.0
        pct_total = (n_above + n_below) / n * 100.0
    else:
        pct_above = pct_below = pct_total = 0.0

    return {
        "n": n,
        "min": float(v.min()),
        "max": float(v.max()),
        "median": median,
        "mean": mean,
        "std": std,
        "mad": mad,
        "std_mad": std_mad,
        "skew_idx": skew_idx,
        "pct_outliers": pct_total,
        "pct_outliers_above": pct_above,
        "pct_outliers_below": pct_below,
        "p01": q[0.01], "p05": q[0.05], "p10": q[0.10],
        "p90": q[0.90], "p95": q[0.95], "p99": q[0.99],
    }


def _compute_period_stats(values: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, sub in values.groupby("period", sort=True):
        s = _stats_for_series(sub["value"])
        if not s:
            continue
        s["period"] = period
        rows.append(s)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if not counts.empty:
        df = df.merge(counts, on="period", how="left")
        df["na_pct"] = (df["na_count"] / df["total"].replace(0, pd.NA)) * 100.0
    else:
        df["total"] = df["n"]
        df["na_count"] = 0
        df["na_pct"] = 0.0
    return df


def _compute_group_period_stats(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, period), sub in values.groupby(["group", "period"], sort=True):
        s = _stats_for_series(sub["value"])
        if not s:
            continue
        s["group"] = group
        s["period"] = period
        rows.append(s)
    return pd.DataFrame(rows)


def _compute_diagnostics(
    db: FinancialDatabase, *, metric_id: int, aggregation: str
) -> dict[str, Any]:
    """Run every query this page needs, so rendering never touches the database again."""
    counts = _load_period_counts(db, metric_id=metric_id)
    values = _load_metric_values(db, metric_id=metric_id)
    payload: dict[str, Any] = {
        "stats": _compute_period_stats(values, counts),
        "overall_mean": _load_metric_overall_mean(db, metric_id=metric_id),
        "group_stats": None,
        "group_label": "",
    }
    group_by = _GROUP_BY_AGGREGATION.get(aggregation)
    if group_by:
        grouped = _load_metric_values_with_group(db, metric_id=metric_id, group_by=group_by)
        payload["group_stats"] = (
            pd.DataFrame() if grouped.empty else _compute_group_period_stats(grouped)
        )
        payload["group_label"] = group_by.capitalize()
    return payload


# ---------- Color rules ----------

_COLOR_GREEN = "background-color: #c6efce; color: #006100"
_COLOR_YELLOW = "background-color: #ffeb9c; color: #9c5700"
_COLOR_RED = "background-color: #ffc7ce; color: #9c0006"

_SKEW_STRONG_POS = "background-color: #d62728; color: white"
_SKEW_LIGHT_POS = "background-color: #f7b6a8"
_SKEW_NEUTRAL = "background-color: #f5f5f5"
_SKEW_LIGHT_NEG = "background-color: #aec7e8"
_SKEW_STRONG_NEG = "background-color: #1f77b4; color: white"


def _color_skew(v):
    if pd.isna(v):
        return ""
    a = abs(v)
    if a >= 1.0:
        return _SKEW_STRONG_POS if v > 0 else _SKEW_STRONG_NEG
    if a >= 0.3:
        return _SKEW_LIGHT_POS if v > 0 else _SKEW_LIGHT_NEG
    return _SKEW_NEUTRAL


def _traffic(v, green_max: float, yellow_max: float):
    if pd.isna(v):
        return ""
    if v <= green_max:
        return _COLOR_GREEN
    if v <= yellow_max:
        return _COLOR_YELLOW
    return _COLOR_RED


def _color_na(v):       return _traffic(v, 5.0, 25.0)
def _color_std_mad(v):  return _traffic(v, 2.0, 4.0)
def _color_outliers(v): return _traffic(v, 5.0, 15.0)


# ---------- Rendering ----------


def _render_headline(stats: pd.DataFrame, overall_mean: float | None) -> None:
    h1, h2, h3 = st.columns(3)
    with h1:
        st.metric("Periods", int(stats["period"].nunique()))
    with h2:
        st.metric("Overall mean", "—" if overall_mean is None else f"{overall_mean:.6g}")
    with h3:
        total_rows = float(stats["total"].sum())
        weighted_na = (
            float(stats["na_count"].sum()) / total_rows * 100.0 if total_rows > 0 else 0.0
        )
        st.metric("Weighted % N/A", f"{weighted_na:.2f}%")


def _render_period_table(stats: pd.DataFrame) -> None:
    st.markdown("### Per-period distribution")
    st.caption(
        "**Skew** = (Mean − Median) / MAD — 🔵 left tail · ⚪ symmetric · 🔴 right tail "
        "(|value| ≥ 1 = strong). "
        "**Std/MAD** & **% Outliers** (Tukey 1.5·IQR) — 🟢 clean · 🟡 moderate · 🔴 contaminated."
    )

    main = stats[[
        "period", "total", "na_pct", "min", "max",
        "skew_idx", "std", "std_mad", "pct_outliers",
    ]].rename(columns={
        "period": "Period",
        "total": "Rows",
        "na_pct": "% N/A",
        "min": "Min",
        "max": "Max",
        "skew_idx": "Skew",
        "std": "Std",
        "std_mad": "Std/MAD",
        "pct_outliers": "% Outliers",
    })

    styler = (
        main.style
        .format({
            "% N/A": "{:.2f}%",
            "Min": "{:.4g}",
            "Max": "{:.4g}",
            "Skew": "{:+.2f}",
            "Std": "{:.4g}",
            "Std/MAD": "{:.2f}",
            "% Outliers": "{:.2f}%",
        }, na_rep="—")
        .map(_color_na, subset=["% N/A"])
        .map(_color_skew, subset=["Skew"])
        .map(_color_std_mad, subset=["Std/MAD"])
        .map(_color_outliers, subset=["% Outliers"])
    )
    st.dataframe(styler, width="stretch", hide_index=True)

    with st.expander("Show mean, median, percentiles & split outliers"):
        detail = stats[[
            "period", "median", "mean", "mad",
            "pct_outliers_above", "pct_outliers_below",
            "p01", "p05", "p10", "p90", "p95", "p99",
        ]].rename(columns={
            "period": "Period",
            "median": "Median",
            "mean": "Mean",
            "mad": "MAD",
            "pct_outliers_above": "% Out ↑",
            "pct_outliers_below": "% Out ↓",
            "p01": "P01", "p05": "P05", "p10": "P10",
            "p90": "P90", "p95": "P95", "p99": "P99",
        })
        st.dataframe(detail, width="stretch", hide_index=True)


def _render_group_table(group_stats: pd.DataFrame, group_label: str) -> None:
    st.divider()
    st.markdown(f"### Breakdown by {group_label.lower()}")
    st.caption(
        "Same color rules as the main table. "
        "Pick a period to compare groups against each other."
    )

    periods_avail = sorted(group_stats["period"].astype(str).unique().tolist())
    # Safe to rerender: the result lives in session state, so changing this cannot clear it.
    sel_period = st.selectbox(
        "Period",
        options=periods_avail,
        index=len(periods_avail) - 1,
        key="diag_breakdown_period",
    )
    one = group_stats.loc[group_stats["period"].astype(str) == str(sel_period)]
    one = one[[
        "group", "n", "min", "max",
        "skew_idx", "std", "std_mad", "pct_outliers",
    ]].rename(columns={
        "group": group_label,
        "n": "Rows",
        "min": "Min",
        "max": "Max",
        "skew_idx": "Skew",
        "std": "Std",
        "std_mad": "Std/MAD",
        "pct_outliers": "% Outliers",
    }).sort_values(group_label)

    g_styler = (
        one.style
        .format({
            "Min": "{:.4g}",
            "Max": "{:.4g}",
            "Skew": "{:+.2f}",
            "Std": "{:.4g}",
            "Std/MAD": "{:.2f}",
            "% Outliers": "{:.2f}%",
        }, na_rep="—")
        .map(_color_skew, subset=["Skew"])
        .map(_color_std_mad, subset=["Std/MAD"])
        .map(_color_outliers, subset=["% Outliers"])
    )
    st.dataframe(g_styler, width="stretch", hide_index=True)


# ---------- Page ----------

render_page_header(
    "Metric Diagnostics",
    "Distribution shape and outlier detection per period (no normality assumed).",
)

db = FinancialDatabase()
name_to_id = _metric_id_by_name(db)
metric_names = sorted(name_to_id.keys())

with st.container(border=True):
    st.markdown("**Inputs**")
    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        selected_metric = st.selectbox(
            "Metric",
            options=metric_names,
            index=0 if metric_names else None,
            disabled=not bool(metric_names),
            help="Select the metric to analyze across uploaded periods.",
        )
    with c2:
        aggregation = st.selectbox(
            "Breakdown",
            options=["Total", "By sector", "By industry"],
            index=0,
            disabled=not bool(metric_names),
        )
    with c3:
        run = st.button("Compute", type="primary", disabled=not bool(metric_names))

if not metric_names:
    st.info("No metrics found. Upload period data first.")
    st.stop()

current_inputs = {"Metric": selected_metric, "Breakdown": aggregation}

if run:
    with st.spinner("Computing diagnostics..."):
        store_result(
            _DIAGNOSTICS_RESULT_KEY,
            _compute_diagnostics(
                db, metric_id=name_to_id[selected_metric], aggregation=aggregation
            ),
            inputs=current_inputs,
        )

# Rendered from session state, so the breakdown selector below cannot clear the tables.
stored_result = read_result(_DIAGNOSTICS_RESULT_KEY)
if stored_result is None:
    st.info("Pick a metric and press Compute to see diagnostics here.")
    st.stop()

st.divider()
render_result_caption(stored_result, current_inputs)

payload = stored_result.payload
period_stats = payload["stats"]
if period_stats.empty:
    st.warning("No rows found for this metric.")
    st.stop()

_render_headline(period_stats, payload["overall_mean"])
st.divider()
_render_period_table(period_stats)

group_stats = payload["group_stats"]
if group_stats is not None:
    if group_stats.empty:
        st.info("No sector/industry breakdown data available.")
    else:
        _render_group_table(group_stats, payload["group_label"])
