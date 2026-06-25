"""Periods - Create (upload), View & Edit (editable table + remove security/metric), Delete."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_app.client.api_client import ApiError
from streamlit_app.ui import (
    get_api_client,
    render_page_header,
    render_section,
    render_sidebar_api_test,
)

render_page_header("Periods", "Create periods from Excel/CSV, view and edit content, remove securities/metrics, delete period.")

client = get_api_client("periods")
render_sidebar_api_test(client, "periods_test_api")

st.divider()
tabs = st.tabs(["Create", "View & Edit", "Delete"])

# --- Create tab ---
with tabs[0]:
    render_section("Create period from file", "Upload Bloomberg, BQL, or Reuters Excel (.xlsx, .xls).")
    upload_success_message = st.session_state.pop("period_upload_success", None)
    if upload_success_message:
        st.success(upload_success_message)

    reader = st.selectbox(
        "Reader",
        options=["bloomberg", "bql", "reuters_metrics"],
        key="period_create_reader",
        format_func=lambda value: {
            "bloomberg": "Bloomberg",
            "bql": "BQL",
            "reuters_metrics": "Reuters Metrics",
        }[value],
    )

    target_period = None
    existing_periods_for_append: list[str] = []
    upload_behavior = "replace"

    if reader == "reuters_metrics":
        st.caption(
            "Reuters uploads require a manual period and store the metric as `Reuters Score`. "
            "The file only needs **Identifier** (or Identifier (RIC)) and **Earnings Quality Country Rank, Current**; "
            "other columns are optional."
        )
        import_mode = st.radio(
            "Import target",
            options=["create_or_replace", "append_existing"],
            key="period_reuters_import_mode",
            format_func=lambda value: (
                "Create or replace a named period"
                if value == "create_or_replace"
                else "Append to an existing period"
            ),
        )

        try:
            existing_periods_for_append = client.list_periods()
        except ApiError as exc:
            st.warning(f"Could not load existing periods: {exc}")

        if import_mode == "append_existing":
            upload_behavior = "append"
            if existing_periods_for_append:
                target_period = st.selectbox(
                    "Existing period",
                    options=existing_periods_for_append,
                    key="period_reuters_existing_period",
                    help="Reuters Score will be merged into this period by ticker.",
                )
                st.caption(
                    "Append merges the uploaded Reuters Score values into the selected period by ticker."
                )
            else:
                st.info("No existing periods available to append to yet.")
        else:
            upload_behavior = "replace"
            target_period = st.text_input(
                "Period",
                key="period_reuters_period",
                help="Required for Reuters uploads because the file does not encode the period.",
            )
            st.caption("This creates the period if it does not exist, or replaces it if it already exists.")
    elif reader == "bql":
        st.caption(
            "BQL uploads use `Name` for security names, `Classification` for GICS data, "
            "read factors from `Current`, `Past`, and `Estimated`, infer the period from "
            "`Config!B1`, and infer the universe from `Config!B2`."
        )
    else:
        st.caption("Bloomberg uploads infer the period directly from the file.")

    file = st.file_uploader(
        "Select file",
        type=["xlsx", "xls"],
        key="period_create_file",
    )
    if reader in ("bloomberg", "bql"):
        upload_behavior = st.selectbox(
            "If period exists",
            options=["replace", "append"],
            index=0,
            key="period_if_exists",
            help="replace = overwrite; append = merge new metrics/securities",
        )
    if st.button("Create period", type="primary", key="period_create_btn"):
        if not file:
            st.error("Select a file first.")
        elif reader == "reuters_metrics" and upload_behavior == "append" and not existing_periods_for_append:
            st.error("No existing periods are available for append.")
        elif reader == "reuters_metrics" and not str(target_period or "").strip():
            st.error("Enter or select a period for the Reuters upload.")
        else:
            try:
                content = file.read()
                result = client.create_period(
                    file_content=content,
                    filename=file.name,
                    if_period_exists=upload_behavior,
                    reader=reader,
                    period=target_period,
                )
                st.session_state["period_upload_success"] = (
                    f"Period '{result.get('period', '')}' uploaded successfully. "
                    f"Companies: {result.get('companies_count', 0)}, "
                    f"Metrics: {result.get('metrics_count', 0)}, "
                    f"Records: {result.get('records_count', 0)}."
                )
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))

# --- View & Edit tab (merged Get + Edit) ---
with tabs[1]:
    render_section("View & Edit period", "Load a period, edit values in the table, then save. Or remove a security or metric from the period.")
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
            preferred_columns = [
                column
                for column in ("ticker", "name", "sector", "industry", "security_id")
                if column in df.columns
            ]
            remaining_columns = [column for column in df.columns if column not in preferred_columns]
            if preferred_columns:
                df = df[preferred_columns + remaining_columns]
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
                width="stretch",
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

            st.divider()
            st.markdown("**Metric parameters (higher is better & N/A treatment)**")

            db_metrics_by_name = {m["metric_name"]: m for m in db_metrics}
            editable_metrics = [m for m in metrics if m in db_metrics_by_name]

            if not editable_metrics:
                st.info("No DB metrics with editable parameters in this period.")
            else:
                hib_options = [
                    ("Keep current", "keep"),
                    ("Higher is better", True),
                    ("Lower is better", False),
                    ("Unset (no preference)", None),
                ]
                na_options = [
                    ("Keep current", "keep"),
                    ("Replace with zero", "replace_with_zero"),
                    ("Replace with high", "replace_with_high"),
                    ("Replace with low", "replace_with_low"),
                    ("Eliminate rows with N/A", "eliminate"),
                    ("Unset (no special handling)", None),
                ]

                pending_updates: list[tuple[int, dict]] = []

                for metric_name in sorted(editable_metrics):
                    metric_info = db_metrics_by_name.get(metric_name, {})
                    metric_id = metric_ids_map.get(metric_name)
                    if not metric_id:
                        continue

                    current_hib = metric_info.get("higher_is_better")
                    current_na = metric_info.get("na_handling")

                    col_label, col_hib, col_na = st.columns([3, 2, 3])
                    with col_label:
                        st.markdown(f"**{metric_name}**")
                        hib_str = (
                            "Higher is better"
                            if current_hib is True
                            else "Lower is better"
                            if current_hib is False
                            else "Not set"
                        )
                        na_str = current_na or "Not set"
                        st.caption(f"Current: {hib_str} | N/A: {na_str}")

                    hib_labels = [x[0] for x in hib_options]
                    hib_values = [x[1] for x in hib_options]
                    na_labels = [x[0] for x in na_options]
                    na_values = [x[1] for x in na_options]

                    with col_hib:
                        hib_choice = st.selectbox(
                            f"Higher is better – {metric_name}",
                            options=hib_labels,
                            index=0,
                            key=f"metric_param_hib_{metric_name}",
                            help="Select how this metric should behave globally.",
                        )
                        hib_value = hib_values[hib_labels.index(hib_choice)]

                    with col_na:
                        na_choice = st.selectbox(
                            f"N/A treatment – {metric_name}",
                            options=na_labels,
                            index=0,
                            key=f"metric_param_na_{metric_name}",
                            help="Select how to treat missing values for this metric globally.",
                        )
                        na_value = na_values[na_labels.index(na_choice)]

                    update_payload: dict = {}
                    if hib_value != "keep":
                        update_payload["higher_is_better"] = hib_value
                    if na_value != "keep":
                        update_payload["na_handling"] = na_value

                    if update_payload:
                        pending_updates.append((metric_id, update_payload))

                if st.button("Save metric parameters", type="secondary", key="metric_params_save_btn"):
                    if not pending_updates:
                        st.info("No metric parameter changes to save.")
                    else:
                        updated_count = 0
                        for mid, payload in pending_updates:
                            try:
                                client.update_db_metric(
                                    mid,
                                    higher_is_better=payload.get("higher_is_better"),
                                    na_handling=payload.get("na_handling"),
                                )
                                updated_count += 1
                            except ApiError as exc:
                                st.error(f"Failed to update metric id {mid}: {exc}")

                        if updated_count:
                            st.success(f"Updated parameters for {updated_count} metric(s).")
                            st.rerun()

# --- Delete tab ---
with tabs[2]:
    render_section("Delete period", "Remove a period and all its data.")
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
