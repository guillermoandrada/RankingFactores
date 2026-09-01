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


def _render_ic_result(result: dict[str, Any]) -> None:
    for warning in result.get("warnings", []):
        st.warning(warning)

    st.divider()
    _render_periods_used(result.get("periods", {}))
    _render_predictive_section(result.get("predictive", []))
    st.divider()
    _render_inter_factor_section(result.get("inter_factor_correlation", {}))
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
    metric_names = []

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
    _render_ic_result(stored_result.payload)
