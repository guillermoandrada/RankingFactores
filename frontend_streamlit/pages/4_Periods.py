"""Periods - Create (upload), View & Edit (editable table + remove security/metric), Delete."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend_streamlit.api_client import ApiError, RankingApiClient


st.set_page_config(page_title="Periods", layout="wide")
st.title("Periods")
st.caption("Create periods from Excel/CSV, view and edit content, remove securities/metrics, delete period.")


def _client() -> RankingApiClient:
    api_url = st.sidebar.text_input(
        "API Base URL",
        value="http://127.0.0.1:8000",
        key="periods_api_url",
    )
    return RankingApiClient(api_url)


client = _client()

if st.sidebar.button("Test API connection", key="periods_test_api"):
    try:
        periods = client.list_periods()
        st.sidebar.success(f"Connected. Periods: {len(periods)}")
    except ApiError as exc:
        st.sidebar.error(str(exc))

tabs = st.tabs(["Create", "View & Edit", "Delete"])

# --- Create tab ---
with tabs[0]:
    st.subheader("Create period from file")
    st.caption("Upload Bloomberg Excel (.xlsx, .xls).")
    file = st.file_uploader(
        "Select file",
        type=["xlsx", "xls"],
        key="period_create_file",
    )
    if_period_exists = st.selectbox(
        "If period exists",
        options=["replace", "append"],
        index=0,
        key="period_if_exists",
        help="replace = overwrite; append = merge new metrics/securities",
    )
    if st.button("Create period", type="primary", key="period_create_btn"):
        if not file:
            st.error("Select a file first.")
        else:
            try:
                content = file.read()
                result = client.create_period(
                    file_content=content,
                    filename=file.name,
                    if_period_exists=if_period_exists,
                )
                st.success(
                    f"Period '{result.get('period', '')}' created. "
                    f"Companies: {result.get('companies_count', 0)}, "
                    f"Metrics: {result.get('metrics_count', 0)}, "
                    f"Records: {result.get('records_count', 0)}."
                )
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))

# --- View & Edit tab (merged Get + Edit) ---
with tabs[1]:
    st.subheader("View & Edit period")
    st.caption("Load a period, edit values in the table, then save. Or remove a security or metric from the period.")
    try:
        periods = client.list_periods()
    except ApiError as exc:
        st.error(f"Cannot load periods: {exc}")
        periods = []

    if not periods:
        st.info("No periods. Upload a file via the Create tab.")
    else:
        selected = st.selectbox(
            "Select period",
            options=periods,
            key="period_view_select",
        )
        if st.button("Load content", key="period_load_btn"):
            try:
                content = client.get_period_content(selected)
                st.session_state["period_content"] = content
                st.session_state["period_name"] = selected
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))

        content = st.session_state.get("period_content")
        period_name = st.session_state.get("period_name", selected)

        if content and content.get("data") and period_name == selected:
            df = pd.DataFrame(content["data"])
            metrics = content.get("metrics", [])
            metric_ids_map = {}
            try:
                db_metrics = client.list_db_metrics()
                metric_ids_map = {m["metric_name"]: m["metric_id"] for m in db_metrics if m.get("metric_id")}
            except ApiError:
                pass

            st.markdown("**Edit values in the table, then click Save changes.**")
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                key="period_data_editor",
                num_rows="fixed",
            )

            st.divider()
            st.markdown("**Commands**")

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Save changes", type="primary", key="period_save_btn"):
                    updates = []
                    for _, row in edited_df.iterrows():
                        ticker = row.get("ticker")
                        if not ticker:
                            continue
                        for m in metrics:
                            if m not in edited_df.columns:
                                continue
                            new_val = row.get(m)
                            orig_row = df[df["ticker"] == ticker]
                            if orig_row.empty:
                                continue
                            orig_val = orig_row[m].iloc[0]
                            if pd.isna(new_val) and pd.isna(orig_val):
                                continue
                            if pd.isna(new_val) != pd.isna(orig_val) or (
                                not pd.isna(new_val) and float(orig_val) != float(new_val)
                            ):
                                if not pd.isna(new_val):
                                    try:
                                        val = float(new_val)
                                        updates.append({
                                            "ticker": str(ticker),
                                            "metric_name": m,
                                            "value": val,
                                        })
                                    except (TypeError, ValueError):
                                        pass
                    if updates:
                        try:
                            result = client.edit_period(period_name, update_values=updates)
                            st.success(f"Updated {result.get('updated_values', 0)} values.")
                            st.session_state.pop("period_content", None)
                            st.rerun()
                        except ApiError as exc:
                            st.error(str(exc))
                    else:
                        st.info("No changes to save.")

            with c2:
                st.caption("Remove a security from this period")
                tickers = df["ticker"].dropna().unique().tolist()
                sec_to_remove = st.selectbox(
                    "Security",
                    options=["— Select —"] + sorted(tickers),
                    key="period_remove_sec_select",
                    label_visibility="collapsed",
                )
                if sec_to_remove != "— Select —" and st.button("Remove security", key="period_remove_sec_btn"):
                    try:
                        sec_ids = df[df["ticker"] == sec_to_remove]["security_id"].dropna().unique().tolist()
                        if sec_ids:
                            client.edit_period(
                                period_name,
                                remove_securities=[int(x) for x in sec_ids],
                            )
                            st.success(f"Removed {sec_to_remove}.")
                            st.session_state.pop("period_content", None)
                            st.rerun()
                        else:
                            st.error("Could not resolve security_id.")
                    except ApiError as exc:
                        st.error(str(exc))

            with c3:
                st.caption("Remove a metric from this period")
                metrics_to_remove = st.selectbox(
                    "Metric",
                    options=["— Select —"] + sorted(metrics),
                    key="period_remove_met_select",
                    label_visibility="collapsed",
                )
                if metrics_to_remove != "— Select —" and st.button("Remove metric", key="period_remove_met_btn"):
                    mid = metric_ids_map.get(metrics_to_remove)
                    if mid:
                        try:
                            client.edit_period(period_name, remove_metrics=[mid])
                            st.success(f"Removed {metrics_to_remove}.")
                            st.session_state.pop("period_content", None)
                            st.rerun()
                        except ApiError as exc:
                            st.error(str(exc))
                    else:
                        st.error("Could not resolve metric_id.")

# --- Delete tab ---
with tabs[2]:
    st.subheader("Delete period")
    try:
        periods = client.list_periods()
    except ApiError as exc:
        st.error(f"Cannot load periods: {exc}")
        periods = []

    if not periods:
        st.info("No periods to delete.")
    else:
        delete_period = st.selectbox(
            "Select period to delete",
            options=periods,
            key="period_delete_select",
        )
        st.warning(f"Delete **{delete_period}**? This removes all fundamental values and index membership.")
        if st.button("Delete period", type="primary", key="period_delete_btn"):
            try:
                client.delete_period(delete_period)
                st.success(f"Period '{delete_period}' deleted.")
                st.session_state.pop("period_content", None)
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))
