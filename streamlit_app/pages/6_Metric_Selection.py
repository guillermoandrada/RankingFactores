"""Metric Selection — multivariate IC (predictive + inter-factor colinearity)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


def _render_periods_used(periods_info: dict[str, Any]) -> None:
    """Show, per metric, how many periods were available and how many were usable."""
    with st.expander("Periods used in this IC analysis", expanded=True):
        per_metric = periods_info.get("per_metric", [])
        if not per_metric:
            st.info("No period availability information returned.")
        else:
            rows = [
                {
                    "Metric": item.get("metric_name", ""),
                    "Available periods": len(item.get("available_periods") or []),
                    "Used periods": len(item.get("used_periods") or []),
                    "Used period list": ", ".join(map(str, item.get("used_periods") or [])),
                }
                for item in per_metric
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

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

    summary_df = pd.DataFrame(
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
    st.dataframe(summary_df, width="stretch", hide_index=True)


def _render_inter_factor_section(inter: dict[str, Any]) -> None:
    """Section B: mean Spearman correlation heatmap between the selected factors."""
    st.markdown("### Section B — Inter-factor correlation")
    st.caption(
        "Mean Spearman correlation matrix across periods on the shared cross-section "
        "(tickers with all selected metrics). "
        "**Absolute correlations above ~0.7 often indicate redundancy.**"
    )

    labels = inter.get("labels", [])
    matrix = inter.get("matrix", [])
    if not labels or not matrix:
        st.info("No inter-factor matrix available.")
        return

    corr_df = pd.DataFrame(matrix, index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), max(5, len(labels) * 0.9)))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Spearman correlation between factors (mean across periods)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "**Note:** Absolute Spearman correlations above **0.7** between factors suggest strong "
        "overlap; consider keeping one representative factor to avoid redundant signals."
    )


def _render_ic_result(result: dict[str, Any]) -> None:
    for warning in result.get("warnings", []):
        st.warning(warning)

    st.divider()
    _render_periods_used(result.get("periods", {}))
    _render_predictive_section(result.get("predictive", []))
    st.divider()
    _render_inter_factor_section(result.get("inter_factor_correlation", {}))


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

with st.container(border=True):
    st.markdown("**Inputs**")
    c1, c2 = st.columns([3, 2])
    with c1:
        selected_metrics = st.multiselect(
            "Metrics",
            options=metric_names,
            default=[],
            key="ic_metrics_multiselect",
            help="Select at least two metrics for multivariate analysis.",
        )
    with c2:
        forward_months = st.selectbox(
            "Forward horizon (months)",
            options=[1, 3, 6],
            index=1,
            key="ic_forward_months",
            help="Forward return horizon used for Rank IC.",
        )

    run = st.button("Run IC analysis", type="primary", key="ic_run_btn")

if not metric_names:
    st.info("No metrics found. Upload period data first.")
    st.stop()

current_inputs = {
    "Metrics": sorted(selected_metrics),
    "Forward horizon": f"{forward_months} month(s)",
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
