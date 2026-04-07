"""Metric Selection — multivariate IC (predictive + inter-factor colinearity)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from modules.analytics.ic_analyzer import ICAnalyzer
from modules.db import FinancialDatabase


st.set_page_config(page_title="Metric Selection (IC)", layout="wide")
st.title("Metric Selection (IC)")
st.caption(
    "Multivariate factor analysis: Rank IC vs forward returns (predictive power) and "
    "Spearman correlation between factors on shared cross-sections (colinearity)."
)

db = FinancialDatabase()
analyzer = ICAnalyzer(db=db)

metrics = db.list_metrics()
metric_names = sorted([m["metric_name"] for m in metrics if m.get("metric_name")])

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
elif run:
    if len(selected_metrics) < 2:
        st.error("Select at least two metrics to run the analysis.")
    else:
        with st.spinner("Computing Rank IC and inter-factor correlations (price fetch may take a while)..."):
            try:
                result = analyzer.analyze_multivariate(
                    metric_names=selected_metrics,
                    forward_months=int(forward_months),
                )
            except ValueError as exc:
                st.error(str(exc))
                result = None

        if result is not None:
            for w in result.get("warnings", []):
                st.warning(w)

            predictive = result.get("predictive", [])
            inter = result.get("inter_factor_correlation", {})
            labels = inter.get("labels", [])
            matrix = inter.get("matrix", [])

            st.divider()
            st.markdown("### Section A — Summary (predictive power)")
            st.caption(
                "Cross-sectional Spearman correlation (Rank IC) between each factor and forward returns, "
                "then time-series mean, volatility, and Information Ratio (Mean IC / σ_IC)."
            )

            if not predictive:
                st.info("No predictive IC results (check fundamentals and price coverage).")
            else:
                rows = []
                for item in predictive:
                    rows.append(
                        {
                            "Metric": item.get("metric_name", ""),
                            "Mean Rank IC": item.get("mean_rank_ic"),
                            "IC Standard Deviation": item.get("ic_std"),
                            "Information Ratio": item.get("information_ratio"),
                            "Periods (T)": item.get("n_periods", 0),
                        }
                    )
                summary_df = pd.DataFrame(rows)
                st.dataframe(summary_df, width="stretch", hide_index=True)

            st.divider()
            st.markdown("### Section B — Inter-factor correlation")
            st.caption(
                "Mean Spearman correlation matrix across periods on the shared cross-section "
                "(tickers with all selected metrics). **Absolute correlations above ~0.7 often indicate redundancy.**"
            )

            if not labels or not matrix:
                st.info("No inter-factor matrix available.")
            else:
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
                    "**Note:** Absolute Spearman correlations above **0.7** between factors suggest strong overlap; "
                    "consider keeping one representative factor to avoid redundant signals."
                )
