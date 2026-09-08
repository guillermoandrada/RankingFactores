"""Metric Selection — multivariate IC (predictive + inter-factor colinearity)."""

from __future__ import annotations

import io
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from streamlit_app.client.api_client import ApiError
from streamlit_app.ui import (
    get_api_client,
    read_result,
    render_page_header,
    render_result_caption,
    render_sidebar_api_status,
    store_result,
)

_IC_RESULT_KEY = "ic_analysis_result"
_ALL_PERIODS_LABEL = "every period available per metric"
# Below this absolute mean Rank IC the sign is treated as noise, not as evidence
# that the stored direction flag is wrong.
_DIRECTION_IC_THRESHOLD = 0.02
_HEATMAP_CELL_PX = 46
_HEATMAP_MIN_PX = 260
_HEATMAP_MAX_PX = 900


def _predictive_dataframe(predictive: list[dict[str, Any]]) -> pd.DataFrame:
    """Section A table, shared by the on-screen view and the export."""
    return pd.DataFrame(
        [
            {
                "Metric": item.get("metric_name", ""),
                "Mean Rank IC": item.get("mean_rank_ic"),
                "IC Standard Deviation": item.get("ic_std"),
                "Information Ratio": item.get("information_ratio"),
                "Periods (T)": item.get("n_periods", 0),
            }
            for item in predictive
        ]
    )


def _correlation_dataframe(inter: dict[str, Any]) -> pd.DataFrame:
    """Square Spearman matrix, empty when the analysis produced none."""
    labels = inter.get("labels") or []
    matrix = inter.get("matrix") or []
    if not labels or not matrix:
        return pd.DataFrame()
    return pd.DataFrame(matrix, index=labels, columns=labels)


def _periods_dataframe(periods_info: dict[str, Any]) -> pd.DataFrame:
    """Per-metric period availability, shared by the on-screen view and the export."""
    return pd.DataFrame(
        [
            {
                "Metric": item.get("metric_name", ""),
                "Available periods": len(item.get("available_periods") or []),
                "Used periods": len(item.get("used_periods") or []),
                "Used period list": ", ".join(map(str, item.get("used_periods") or [])),
            }
            for item in (periods_info.get("per_metric") or [])
        ]
    )


@st.cache_data(show_spinner=False)
def _ic_export_bytes(result: dict[str, Any]) -> bytes:
    """One workbook holding the summary, the correlation matrix, and the periods used."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        _predictive_dataframe(result.get("predictive", [])).to_excel(
            writer, index=False, sheet_name="Predictive"
        )
        correlation = _correlation_dataframe(result.get("inter_factor_correlation", {}))
        if not correlation.empty:
            correlation.to_excel(writer, sheet_name="Correlation")
        periods = _periods_dataframe(result.get("periods", {}))
        if not periods.empty:
            periods.to_excel(writer, index=False, sheet_name="Periods")
    return buffer.getvalue()


def _render_periods_used(periods_info: dict[str, Any]) -> None:
    """Show, per metric, how many periods were available and how many were usable."""
    with st.expander("Periods used in this IC analysis", expanded=True):
        per_metric = _periods_dataframe(periods_info)
        if per_metric.empty:
            st.info("No period availability information returned.")
        else:
            st.dataframe(per_metric, width="stretch", hide_index=True)

        shared_available = (periods_info.get("inter_factor") or {}).get("available_periods") or []
        if shared_available:
            st.caption(
                "Shared periods available for inter-factor correlation (intersection): "
                f"{len(shared_available)}"
            )
            st.code(", ".join(map(str, shared_available)))


def _render_predictive_section(predictive: list[dict[str, Any]]) -> None:
    """Section A: mean Rank IC, IC volatility, and Information Ratio per factor."""
    st.markdown("### Section A — Summary (predictive power)")
    st.caption(
        "Cross-sectional Spearman correlation (Rank IC) between each factor and forward returns, "
        "then time-series mean, volatility, and Information Ratio (Mean IC / σ_IC)."
    )
    if not predictive:
        st.info("No predictive IC results (check fundamentals and price coverage).")
        return
    st.dataframe(_predictive_dataframe(predictive), width="stretch", hide_index=True)


def _render_correlation_heatmap(correlation: pd.DataFrame) -> None:
    """
    Interactive Spearman heatmap.

    Altair rather than a static image, so long metric names stay readable through
    tooltips and the chart matches every other chart in the app.
    """
    labels = [str(label) for label in correlation.columns]
    records = [
        {"row": str(row_label), "column": str(column_label), "correlation": float(value)}
        for row_label in correlation.index
        for column_label, value in correlation.loc[row_label].items()
        if pd.notna(value)
    ]
    if not records:
        st.info("No inter-factor matrix available.")
        return

    size = min(_HEATMAP_MAX_PX, max(_HEATMAP_MIN_PX, _HEATMAP_CELL_PX * len(labels)))
    base = alt.Chart(pd.DataFrame(records)).encode(
        x=alt.X("column:N", title=None, sort=labels, axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("row:N", title=None, sort=labels),
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color(
            "correlation:Q",
            title="Spearman",
            scale=alt.Scale(scheme="redblue", domain=[-1.0, 1.0], reverse=True),
        ),
        tooltip=[
            alt.Tooltip("row:N", title="Factor"),
            alt.Tooltip("column:N", title="vs"),
            alt.Tooltip("correlation:Q", title="Spearman", format=".3f"),
        ],
    )
    annotations = base.mark_text(fontSize=11).encode(
        text=alt.Text("correlation:Q", format=".2f"),
        # Cells at either extreme are dark, so their labels need to flip to white.
        color=alt.condition(
            "abs(datum.correlation) > 0.55", alt.value("white"), alt.value("black")
        ),
    )
    st.altair_chart(
        (heatmap + annotations).properties(width=size, height=size),
        use_container_width=True,
    )


def _render_inter_factor_section(inter: dict[str, Any]) -> None:
    """Section B: mean Spearman correlation between the selected factors."""
    st.markdown("### Section B — Inter-factor correlation")
    st.caption(
        "Mean Spearman correlation matrix across periods on the shared cross-section "
        "(tickers with all selected metrics). "
        "**Absolute correlations above ~0.7 often indicate redundancy.**"
    )

    correlation = _correlation_dataframe(inter)
    if correlation.empty:
        st.info("No inter-factor matrix available.")
        return

    _render_correlation_heatmap(correlation)
    st.info(
        "**Note:** Absolute Spearman correlations above **0.7** between factors suggest strong "
        "overlap; consider keeping one representative factor to avoid redundant signals."
    )


def _direction_verdict(mean_ic: float | None, higher_is_better: bool) -> str:
    """
    Classify one metric's IC sign against its stored direction flag.

    Returns one of: "consistent", "contradiction", "inconclusive".
    """
    if mean_ic is None:
        return "inconclusive"
    if abs(mean_ic) < _DIRECTION_IC_THRESHOLD:
        return "inconclusive"
    ic_says_higher_is_better = mean_ic > 0
    return "consistent" if ic_says_higher_is_better == higher_is_better else "contradiction"


def _flip_db_metric_direction(client, metric_id: int, metric_name: str, new_value: bool) -> None:
    """Persist the inverted higher_is_better flag and refresh the page."""
    try:
        client.update_db_metric(metric_id, higher_is_better=new_value)
    except ApiError as exc:
        st.error(f"Could not update '{metric_name}': {exc}")
        return
    direction = "higher is better" if new_value else "lower is better"
    st.toast(f"'{metric_name}' flipped to {direction}.")
    st.rerun()


def _render_direction_check(
    predictive: list[dict[str, Any]],
    db_metric_index: dict[str, dict[str, Any]],
    client,
) -> None:
    """
    Section C: cross-check the IC sign against each metric's stored direction flag.

    A strongly negative Mean Rank IC on a metric marked higher_is_better=True (or the
    reverse) means the metric would score backwards in scoring profiles and backtests;
    offer a one-click flip of the flag.
    """
    st.markdown("### Section C — Direction check (IC sign vs higher_is_better)")
    st.caption(
        "Scoring profiles use each metric's **higher_is_better** flag to orient its "
        "z-score, while the Rank IC above is computed on raw values. When the IC sign "
        "contradicts the stored flag, the metric contributes with the wrong sign in "
        f"rankings and backtests. Contradictions require |Mean Rank IC| ≥ "
        f"{_DIRECTION_IC_THRESHOLD:.2f}; weaker ICs are treated as inconclusive."
    )
    if not predictive:
        st.info("No predictive IC results to check directions against.")
        return

    contradictions = 0
    for item in predictive:
        metric_name = item.get("metric_name", "")
        mean_ic = item.get("mean_rank_ic")
        db_metric = db_metric_index.get(metric_name)
        stored_flag = db_metric.get("higher_is_better") if db_metric else None
        # Scoring falls back to higher-is-better when the flag was never set.
        effective_flag = True if stored_flag is None else bool(stored_flag)
        verdict = _direction_verdict(mean_ic, effective_flag)

        flag_label = "higher is better" if effective_flag else "lower is better"
        if stored_flag is None:
            flag_label += " (not set, scoring default)"
        ic_label = "n/a" if mean_ic is None else f"{mean_ic:+.4f}"

        name_col, ic_col, flag_col, action_col = st.columns([3, 2, 3, 3])
        name_col.markdown(f"**{metric_name}**")
        ic_col.markdown(f"Mean IC: `{ic_label}`")
        flag_col.markdown(f"Flag: `{flag_label}`")

        if verdict == "contradiction" and db_metric and db_metric.get("metric_id") is not None:
            contradictions += 1
            new_value = not effective_flag
            suggested = "higher is better" if new_value else "lower is better"
            with action_col:
                if st.button(
                    f"Flip to '{suggested}'",
                    key=f"ic_flip_direction_{db_metric['metric_id']}",
                    help=(
                        f"'{metric_name}' is marked '{flag_label}' but its Mean Rank IC "
                        f"is {ic_label}: it is scoring backwards. Flip the flag so "
                        "scoring matches the observed IC direction."
                    ),
                ):
                    _flip_db_metric_direction(
                        client, int(db_metric["metric_id"]), metric_name, new_value
                    )
        elif verdict == "contradiction":
            action_col.markdown("⚠️ Contradiction (flag not editable here)")
        elif verdict == "consistent":
            action_col.markdown("✅ Consistent")
        else:
            action_col.markdown("➖ Inconclusive (weak IC)")

    if contradictions:
        st.warning(
            f"{contradictions} metric(s) have an IC sign that contradicts their stored "
            "higher_is_better flag. Until flipped, they contribute backwards to any "
            "scoring profile and backtest that uses them."
        )
    else:
        st.success("No direction contradictions between IC signs and stored flags.")


def _render_export(result: dict[str, Any]) -> None:
    st.divider()
    with st.container(border=True):
        st.markdown("**Export**")
        st.caption("Summary, correlation matrix, and periods used, in one workbook.")
        st.download_button(
            "Export to Excel",
            data=_ic_export_bytes(result),
            file_name="IC_Analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="ic_export_xlsx",
        )


def _render_ic_result(
    result: dict[str, Any],
    db_metric_index: dict[str, dict[str, Any]],
    client,
) -> None:
    for warning in result.get("warnings", []):
        st.warning(warning)

    st.divider()
    _render_periods_used(result.get("periods", {}))
    _render_predictive_section(result.get("predictive", []))
    st.divider()
    _render_inter_factor_section(result.get("inter_factor_correlation", {}))
    st.divider()
    _render_direction_check(result.get("predictive", []), db_metric_index, client)
    _render_export(result)


render_page_header(
    "Metric Selection (IC)",
    "Multivariate factor analysis: Rank IC vs forward returns (predictive power) and "
    "Spearman correlation between factors on shared cross-sections (colinearity).",
)

client = get_api_client()
render_sidebar_api_status(client)

try:
    db_metrics = client.list_db_metrics()
    metric_names = sorted([m["metric_name"] for m in db_metrics if m.get("metric_name")])
except ApiError as exc:
    st.error(f"Cannot load metrics: {exc}")
    db_metrics = []
    metric_names = []

db_metric_index = {m["metric_name"]: m for m in db_metrics if m.get("metric_name")}

try:
    available_periods = client.list_periods()
except ApiError:
    available_periods = []

with st.container(border=True):
    st.markdown("**Inputs**")
    metrics_column, horizon_column = st.columns([3, 2])
    with metrics_column:
        selected_metrics = st.multiselect(
            "Metrics",
            options=metric_names,
            default=[],
            key="ic_metrics_multiselect",
            help="Select at least two metrics for multivariate analysis.",
        )
    with horizon_column:
        forward_months = st.number_input(
            "Forward horizon (months)",
            min_value=1,
            max_value=60,
            value=3,
            step=1,
            key="ic_forward_months",
            help="Forward return horizon used for Rank IC.",
        )

    selected_periods = st.multiselect(
        "Periods",
        options=available_periods,
        default=[],
        key="ic_periods_multiselect",
        help=(
            "Restrict the analysis to these periods, for example to compare one regime "
            "against another. Leave empty to use every period available per metric."
        ),
    )
    if not available_periods:
        st.caption("No periods available to scope by.")

    run = st.button("Run IC analysis", type="primary", key="ic_run_btn")

if not metric_names:
    st.info("No metrics found. Upload period data first.")
    st.stop()

current_inputs = {
    "Metrics": sorted(selected_metrics),
    "Forward horizon": f"{int(forward_months)} month(s)",
    "Periods": sorted(selected_periods) if selected_periods else _ALL_PERIODS_LABEL,
}

if run:
    if len(selected_metrics) < 2:
        st.error("Select at least two metrics to run the analysis.")
    else:
        spinner_text = "Computing Rank IC and inter-factor correlations (price fetch may take a while)..."
        with st.spinner(spinner_text):
            try:
                store_result(
                    _IC_RESULT_KEY,
                    client.run_ic_analysis(
                        metric_names=selected_metrics,
                        forward_months=int(forward_months),
                        periods=sorted(selected_periods) or None,
                    ),
                    inputs=current_inputs,
                )
            except (ApiError, ValueError) as exc:
                st.error(str(exc))

# Rendered from session state, so the result survives every later widget interaction.
stored_result = read_result(_IC_RESULT_KEY)
if stored_result is None:
    st.info("Run the IC analysis to see results here.")
else:
    st.divider()
    render_result_caption(stored_result, current_inputs)
    _render_ic_result(stored_result.payload, db_metric_index, client)
