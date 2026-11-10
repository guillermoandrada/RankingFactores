"""Metric Diagnostics — distribution shape and outlier detection per period."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import select

from modules.config.derived_metrics import DerivedMetricStore
from modules.domain.analytics.metric_loader import (
    fetch_metric_matrix,
    resolve_metric_dependencies,
)
from modules.domain.analytics.security_groups import select_security_groups
from modules.infrastructure.db import FinancialDatabase
from streamlit_app.ui import (
    is_derived_metric,
    metric_label,
    read_result,
    render_page_header,
    render_result_caption,
    store_result,
)

_DIAGNOSTICS_RESULT_KEY = "metric_diagnostics_result"


# ---------- Data loading ----------
#
# Every metric, DB or derived, is loaded as the same long frame of raw rows
# (period, security_id, value), NaN kept, so the stats below never branch on kind.

_ROW_COLUMNS = ["period", "security_id", "value"]


def _metric_index(
    db: FinancialDatabase, derived_store: DerivedMetricStore
) -> dict[str, dict[str, Any]]:
    """DB metrics and derived formulas by name, as GET /db/metrics/available lists them."""
    index: dict[str, dict[str, Any]] = {}
    for metric in db.list_metrics():
        name = (metric.get("metric_name") or "").strip()
        if name and isinstance(metric.get("metric_id"), int):
            index[name] = metric
    for name, formula in derived_store.list_formulas().items():
        index.setdefault(name, {**formula, "metric_name": name, "derived": True})
    return index


def _load_db_metric_rows(db: FinancialDatabase, *, metric_id: int) -> pd.DataFrame:
    tbl = db._metadata.tables["fundamental_values"]
    q = (
        select(
            tbl.c.period.label("period"),
            tbl.c.security_id.label("security_id"),
            tbl.c.value.label("value"),
        )
        .where(tbl.c.metric_id == int(metric_id))
        .order_by(tbl.c.period)
    )
    with db.engine.connect() as conn:
        return pd.read_sql_query(q, con=conn)


def _select_metric_universe(db: FinancialDatabase, *, metric_ids: list[int] | None = None):
    """Distinct (period, security_id) with a fundamental_values row, NULL values included."""
    tbl_fund = db._metadata.tables["fundamental_values"]
    q = select(tbl_fund.c.period, tbl_fund.c.security_id).distinct()
    if metric_ids is not None:
        q = q.where(tbl_fund.c.metric_id.in_(metric_ids))
    return q


def _load_metric_universe(db: FinancialDatabase, *, metric_ids: list[int]) -> pd.DataFrame:
    with db.engine.connect() as conn:
        universe = pd.read_sql_query(
            _select_metric_universe(db, metric_ids=metric_ids), con=conn
        )
    universe["period"] = universe["period"].astype(str)
    return universe


def _base_metric_ids(
    db: FinancialDatabase, *, metric_name: str, derived_store: DerivedMetricStore
) -> list[int]:
    """Ids of the DB metrics a derived metric is built from; formula errors propagate."""
    db_ids = {
        metric["metric_name"]: metric["metric_id"]
        for metric in db.list_metrics()
        if isinstance(metric.get("metric_id"), int)
    }
    dependencies = resolve_metric_dependencies(
        metric_name, derived_store.list_formulas(), set(db_ids)
    )
    return [db_ids[name] for name in dependencies if name in db_ids]


def _load_derived_metric_rows(
    db: FinancialDatabase, *, metric_name: str, derived_store: DerivedMetricStore
) -> pd.DataFrame:
    """
    Compute a derived metric per period from its raw base metrics.

    The row universe is every security with a row for any base metric, NULL values
    included, so missing data counts as N/A exactly as it does for DB metrics. N/A
    fills are skipped. Periods where the formula resolves to nothing (a base metric
    has no data) are left out, matching where the metric is usable in rankings and IC.
    """
    base_ids = _base_metric_ids(db, metric_name=metric_name, derived_store=derived_store)
    universe = _load_metric_universe(db, metric_ids=base_ids)
    frames: list[pd.DataFrame] = []
    for period, period_universe in universe.groupby("period", sort=True):
        wide, _ = fetch_metric_matrix(
            db.engine,
            period,
            [metric_name],
            derived_store,
            apply_na_handling=False,
        )
        if metric_name in wide.attrs.get("missing_metrics", []):
            continue
        security_ids = period_universe["security_id"].to_numpy()
        values = wide[metric_name].droplevel(["ticker", "long_name"])
        frames.append(pd.DataFrame({
            "period": period,
            "security_id": security_ids,
            "value": values.reindex(security_ids).to_numpy(),
        }))
    if not frames:
        return pd.DataFrame(columns=_ROW_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _normalise_rows(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.reindex(columns=_ROW_COLUMNS).copy()
    rows["period"] = rows["period"].astype(str)
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    return rows


def _load_metric_rows(
    db: FinancialDatabase,
    *,
    metric: dict[str, Any],
    derived_store: DerivedMetricStore,
) -> pd.DataFrame:
    if is_derived_metric(metric):
        rows = _load_derived_metric_rows(
            db, metric_name=metric["metric_name"], derived_store=derived_store
        )
    else:
        rows = _load_db_metric_rows(db, metric_id=metric["metric_id"])
    return _normalise_rows(rows)


def _load_security_groups(db: FinancialDatabase, *, group_by: str) -> pd.DataFrame:
    """Sector or industry per (period, security): period classification, then security."""
    universe = _select_metric_universe(db).subquery()
    q = select_security_groups(db._metadata, universe, group_by=group_by)
    with db.engine.connect() as conn:
        df = pd.read_sql_query(q, con=conn)
    df["period"] = df["period"].astype(str)
    return df[["period", "security_id", "group"]]


def _attach_groups(rows: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.merge(groups, on=["period", "security_id"], how="left")
    grouped["group"] = grouped["group"].astype("string").fillna("Unknown").astype(str)
    return grouped


def _period_counts(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["period", "total", "na_count"])
    return (
        rows.assign(is_na=rows["value"].isna())
        .groupby("period", sort=True)
        .agg(total=("value", "size"), na_count=("is_na", "sum"))
        .astype(int)
        .reset_index()
    )


def _overall_mean(rows: pd.DataFrame) -> float | None:
    mean = rows["value"].mean()
    return None if pd.isna(mean) else float(mean)


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
    rows: pd.DataFrame, *, groups: pd.DataFrame | None, group_label: str
) -> dict[str, Any]:
    """Build every table this page renders from the metric's raw rows."""
    values = rows.dropna(subset=["value"])
    return {
        "stats": _compute_period_stats(values, _period_counts(rows)),
        "overall_mean": _overall_mean(rows),
        "group_stats": (
            None if groups is None
            else _compute_group_period_stats(_attach_groups(rows, groups))
        ),
        "group_label": group_label,
    }


def _run_diagnostics(
    db: FinancialDatabase,
    *,
    metric: dict[str, Any],
    derived_store: DerivedMetricStore,
    aggregation: str,
) -> dict[str, Any]:
    """Run every query this page needs, so rendering never touches the database again."""
    rows = _load_metric_rows(db, metric=metric, derived_store=derived_store)
    group_by = _GROUP_BY_AGGREGATION.get(aggregation)
    groups = _load_security_groups(db, group_by=group_by) if group_by else None
    return _compute_diagnostics(
        rows, groups=groups, group_label=group_by.capitalize() if group_by else ""
    )


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
derived_store = DerivedMetricStore()
metric_index = _metric_index(db, derived_store)
metric_names = sorted(metric_index.keys())
derived_count = sum(1 for m in metric_index.values() if is_derived_metric(m))

with st.container(border=True):
    st.markdown("**Inputs**")
    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        selected_metric = st.selectbox(
            "Metric",
            options=metric_names,
            index=0 if metric_names else None,
            disabled=not bool(metric_names),
            format_func=lambda name: metric_label(metric_index, name),
            help=(
                "Select the metric to analyze across uploaded periods. Derived metrics "
                "are listed with their formula and computed per period from their raw "
                "base metrics, before any N/A treatment."
            ),
        )
        if derived_count:
            st.caption(f"{derived_count} derived metric(s) available.")
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
        try:
            diagnostics = _run_diagnostics(
                db,
                metric=metric_index[selected_metric],
                derived_store=derived_store,
                aggregation=aggregation,
            )
        except ValueError as exc:
            st.error(f"Cannot compute diagnostics for '{selected_metric}': {exc}")
            st.stop()
        store_result(_DIAGNOSTICS_RESULT_KEY, diagnostics, inputs=current_inputs)

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
