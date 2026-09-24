"""Price Data — upload Bloomberg prices and manage cached price data."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_app.client.api_client import ApiError
from streamlit_app.ui import get_api_client, render_page_header, render_sidebar_api_status

render_page_header(
    "Price Data",
    "The price cache behind IC analysis and backtests. Series downloaded from Yahoo "
    "Finance are stored here automatically and served from the database on later "
    "runs; uploaded Bloomberg files take priority over them, date by date.",
)

_DOWNLOADED_SOURCE = "yfinance"

client = get_api_client()
render_sidebar_api_status(client)

upload_tab, manage_tab, total_returns_tab = st.tabs(["Upload", "Manage", "Total returns"])

# ── Upload ────────────────────────────────────────────────────────────────────
with upload_tab:
    st.markdown(
        "Upload a Bloomberg DAPI wide-format Excel file. "
        "Expected layout: **Row 1** = ticker headers (column B onwards); "
        "**Row 2+** = date (column A) + close prices."
    )
    uploaded = st.file_uploader(
        "Bloomberg price file",
        type=["xlsx", "xls"],
        key="price_upload",
    )
    if uploaded is not None:
        if st.button("Import prices", type="primary"):
            with st.spinner("Parsing and persisting prices…"):
                try:
                    result = client.upload_price_file(uploaded.getvalue(), uploaded.name)
                    tickers = result.get("tickers_imported", [])
                    rows = result.get("rows_written", 0)
                    dr = result.get("date_range", {})
                    st.success(
                        f"Imported **{rows:,}** rows for **{len(tickers)}** tickers "
                        f"({dr.get('min', '?')} → {dr.get('max', '?')})."
                    )
                    if tickers:
                        st.write("**Tickers imported:**", ", ".join(tickers))
                except ApiError as exc:
                    st.error(str(exc))

# ── Manage ────────────────────────────────────────────────────────────────────
with manage_tab:
    st.markdown("View, invalidate and delete cached price series.")

    if st.button("Reload list", key="price_refresh"):
        st.session_state.pop("cached_price_tickers", None)

    if "cached_price_tickers" not in st.session_state:
        with st.spinner("Loading cached tickers…"):
            try:
                st.session_state["cached_price_tickers"] = client.list_cached_price_tickers()
            except ApiError as exc:
                st.error(str(exc))
                st.session_state["cached_price_tickers"] = []

    rows_data = st.session_state.get("cached_price_tickers", [])

    if not rows_data:
        st.info(
            "No price data cached yet. Upload a Bloomberg file, or run an IC analysis "
            "or backtest — downloaded series are stored automatically."
        )
    else:
        df = pd.DataFrame(rows_data)
        if "sources" in df.columns:
            df["sources"] = df["sources"].apply(
                lambda values: ", ".join(values) if isinstance(values, list) else values
            )
        st.dataframe(df, use_container_width=True, hide_index=True)

        ticker_list = sorted(df["ticker"].tolist())
        selected = st.multiselect(
            "Select tickers",
            options=ticker_list,
            key="price_delete_select",
        )
        if selected:
            refresh_col, delete_col = st.columns(2)
            with refresh_col:
                if st.button(
                    f"Invalidate downloaded prices ({len(selected)})",
                    key="price_invalidate_btn",
                    help=(
                        "Drop only the rows downloaded from Yahoo Finance so the next "
                        "run refetches them. Uploaded Bloomberg prices are kept. Use "
                        "this after a split or a large dividend, which rescale the "
                        "whole adjusted history."
                    ),
                ):
                    try:
                        result = client.delete_cached_price_tickers(
                            selected, source=_DOWNLOADED_SOURCE
                        )
                        deleted = result.get("deleted_rows", 0)
                        st.success(
                            f"Invalidated **{deleted:,}** downloaded rows for: "
                            f"{', '.join(selected)}"
                        )
                        st.session_state.pop("cached_price_tickers", None)
                        st.rerun()
                    except ApiError as exc:
                        st.error(str(exc))
            with delete_col:
                if st.button(
                    f"Delete all prices ({len(selected)})",
                    type="primary",
                    key="price_delete_btn",
                    help="Remove every cached row, uploaded prices included.",
                ):
                    try:
                        result = client.delete_cached_price_tickers(selected)
                        deleted = result.get("deleted_rows", 0)
                        st.success(f"Deleted **{deleted:,}** rows for: {', '.join(selected)}")
                        st.session_state.pop("cached_price_tickers", None)
                        st.rerun()
                    except ApiError as exc:
                        st.error(str(exc))

# ── Total returns ─────────────────────────────────────────────────────────────
with total_returns_tab:
    st.markdown(
        "Upload a Bloomberg BQL total-return export. Each sheet holds side-by-side "
        "blocks of five columns: **Row 1** = benchmark, end date, start date; "
        "**Row 2** = `ID`, name, sector, industry, `total_return(...)`; "
        "**Row 3+** = Bloomberg identifier + total return. The horizon (3m, 6m, 12m…) "
        "is derived from each block's dates."
    )
    uploaded_returns = st.file_uploader(
        "Bloomberg total-return file",
        type=["xlsx", "xls"],
        key="total_return_upload",
    )
    if uploaded_returns is not None:
        if st.button("Import total returns", type="primary"):
            with st.spinner("Parsing and persisting total returns…"):
                try:
                    result = client.upload_total_return_file(
                        uploaded_returns.getvalue(), uploaded_returns.name
                    )
                    tickers = result.get("tickers_imported", [])
                    rows = result.get("rows_written", 0)
                    windows = result.get("windows_imported", 0)
                    horizons = ", ".join(f"{h}m" for h in result.get("horizons_months", []))
                    dr = result.get("date_range", {})
                    st.success(
                        f"Imported **{rows:,}** rows for **{len(tickers)}** tickers across "
                        f"**{windows}** windows ({horizons}; "
                        f"{dr.get('min', '?')} → {dr.get('max', '?')})."
                    )
                    skipped = result.get("tickers_skipped", [])
                    if skipped:
                        st.warning(f"Unrecognised identifiers skipped: {', '.join(skipped)}")
                    st.session_state.pop("total_return_windows", None)
                except ApiError as exc:
                    st.error(str(exc))

    if st.button("Reload list", key="total_return_refresh"):
        st.session_state.pop("total_return_windows", None)

    if "total_return_windows" not in st.session_state:
        try:
            st.session_state["total_return_windows"] = client.list_total_return_windows()
        except ApiError as exc:
            st.error(str(exc))
            st.session_state["total_return_windows"] = []

    windows_data = st.session_state.get("total_return_windows", [])
    if not windows_data:
        st.info("No total returns stored yet.")
    else:
        windows_df = pd.DataFrame(windows_data)
        st.dataframe(windows_df, use_container_width=True, hide_index=True)

        horizon_options = sorted(windows_df["horizon_months"].unique().tolist())
        selected_horizon = st.selectbox(
            "Horizon to delete",
            options=[None, *horizon_options],
            format_func=lambda h: "All horizons" if h is None else f"{h}m",
            key="total_return_delete_horizon",
        )
        if st.button("Delete total returns", type="primary", key="total_return_delete_btn"):
            try:
                result = client.delete_total_return_windows(selected_horizon)
                st.success(f"Deleted **{result.get('deleted_rows', 0):,}** rows.")
                st.session_state.pop("total_return_windows", None)
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))
