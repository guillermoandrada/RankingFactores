"""Periods - Create (upload), View & Edit (editable table + remove security/metric), Delete."""

from __future__ import annotations

import hashlib
import io
from typing import Any

import pandas as pd
import streamlit as st

from streamlit_app.client.api_client import ApiError
from streamlit_app.ui import (
    get_api_client,
    render_page_header,
    render_section,
    render_sidebar_api_status,
)

_VARIABLE_READER = "bloomberg_individual_variable"

_READER_LABELS = {
    "bloomberg": "Bloomberg",
    "bql": "BQL",
    "reuters_metrics": "Reuters Metrics",
    _VARIABLE_READER: "Bloomberg Individual Variable",
}

# Columns the editor shows but never saves, and the internal key it should not show at all.
_LOCKED_EDITOR_COLUMNS = ("ticker", "name", "sector", "industry")
_HIDDEN_EDITOR_COLUMNS = ("security_id",)

_REMOVE_SECURITY_STATE_KEY = "period_pending_remove_security"
_REMOVE_METRIC_STATE_KEY = "period_pending_remove_metric"


def _render_variable_upload_result(result: dict) -> None:
    """Report a single-variable upload, which spans several periods at once."""
    periods = result.get("periods", [])
    st.success(
        f"Variable **{result.get('variable', '')}** imported: "
        f"**{result.get('records_count', 0):,}** values for "
        f"**{result.get('securities_count', 0)}** securities "
        f"across **{len(periods)}** period(s)."
    )
    if result.get("creates_securities"):
        st.caption(
            "The file carried names and GICS data, so missing securities were created "
            "and each period's classification was refreshed."
        )
    else:
        st.caption(
            "Ticker and value only: values were appended to securities that already "
            "exist. No security was created and no classification was changed."
        )
    if periods:
        st.dataframe(pd.DataFrame(periods), width="stretch", hide_index=True)

    securities_skipped = result.get("securities_skipped", [])
    if securities_skipped:
        st.warning(
            f"{len(securities_skipped)} identifier(s) have no security in the database "
            "and were skipped. Add long name and GICS columns to create them, or import "
            "the period fundamentals first."
        )
        st.write(_truncated_list(securities_skipped))
    skipped_periods = result.get("periods_skipped", [])
    if skipped_periods:
        st.warning(f"Period blocks skipped: {', '.join(skipped_periods)}")
    rows_skipped = result.get("rows_skipped", 0)
    if rows_skipped:
        st.info(f"{rows_skipped:,} row(s) skipped for a missing ticker.")


_ACTION_NOTICES = {
    "create": ("success", "Creates a new period **{period}**."),
    "replace": (
        "warning",
        "Period **{period}** already exists. Importing **replaces** it: every existing "
        "value in that period is overwritten.",
    ),
    "append": (
        "info",
        "Period **{period}** already exists. Importing **merges** the file into it, "
        "keeping metrics and securities that the file does not mention.",
    ),
}


def _render_upload_preview(preview: dict) -> None:
    """Show what an import would do, so a replace is never a surprise."""
    period = preview.get("period")
    if not period:
        st.error(
            preview.get("period_error")
            or "No period could be determined from this file. Enter one manually if the "
            "reader supports it."
        )
    else:
        level, template = _ACTION_NOTICES.get(preview.get("action", "create"), _ACTION_NOTICES["create"])
        getattr(st, level)(template.format(period=period))
        if preview.get("period_source") == "manual":
            st.caption("Period taken from the field above, not from the file.")

    summary_columns = st.columns(4)
    summary_columns[0].metric("Rows", f"{preview.get('row_count', 0):,}")
    summary_columns[1].metric("Columns", preview.get("column_count", 0))
    summary_columns[2].metric("Index code", preview.get("index_code") or "—")
    summary_columns[3].metric("Reader", preview.get("reader", ""))

    if preview.get("missing_ticker_column"):
        st.error(
            "No **Ticker** column was found. The import will fail — check the reader "
            "matches this file."
        )

    sheet_names = preview.get("sheet_names") or []
    if sheet_names:
        st.caption(f"Sheets in the workbook: {_truncated_list([str(s) for s in sheet_names])}")

    sample_rows = preview.get("sample_rows") or []
    if sample_rows:
        st.caption("First rows as the reader parses them:")
        st.dataframe(pd.DataFrame(sample_rows), width="stretch", hide_index=True)


def _render_workbook_sheets(file_bytes: bytes) -> list[str]:
    """
    List the workbook's sheets locally.

    Sheet names are a property of the file, not of any reader, so reading them here
    duplicates no ingestion logic.
    """
    try:
        with pd.ExcelFile(io.BytesIO(file_bytes)) as workbook:
            return [str(name) for name in workbook.sheet_names]
    except (ValueError, OSError):
        return []


def _truncated_list(values: list[str], limit: int = 40) -> str:
    """Comma-separated preview, so a wide universe cannot flood the page."""
    shown = ", ".join(values[:limit])
    return shown if len(values) <= limit else f"{shown} … (+{len(values) - limit} more)"


def _values_match(new_value: Any, original_value: Any) -> bool:
    """True when an edited cell is unchanged, treating every missing form as equal."""
    new_missing = bool(pd.isna(new_value))
    original_missing = bool(pd.isna(original_value))
    if new_missing or original_missing:
        return new_missing and original_missing
    try:
        return float(new_value) == float(original_value)
    except (TypeError, ValueError):
        return str(new_value) == str(original_value)


def _collect_period_edits(
    original_df: pd.DataFrame,
    edited_df: pd.DataFrame,
    metric_names: list[str],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """
    Compare the edited table against the loaded one.

    Returns (updates, cleared_cells, invalid_cells). Cleared and invalid cells cannot be
    expressed by the API, so the caller reports them instead of dropping them silently.
    """
    if "ticker" not in edited_df.columns or "ticker" not in original_df.columns:
        return [], [], []
    editable = [
        name
        for name in metric_names
        if name in edited_df.columns and name in original_df.columns
    ]
    if not editable:
        return [], [], []

    baseline = original_df.drop_duplicates(subset="ticker", keep="first").set_index("ticker")

    updates: list[dict[str, Any]] = []
    cleared: list[str] = []
    invalid: list[str] = []
    for _, row in edited_df.iterrows():
        ticker = row.get("ticker")
        if not ticker or ticker not in baseline.index:
            continue
        original_row = baseline.loc[ticker]
        for metric_name in editable:
            new_value = row.get(metric_name)
            if _values_match(new_value, original_row.get(metric_name)):
                continue
            cell = f"{ticker} · {metric_name}"
            if pd.isna(new_value):
                cleared.append(cell)
                continue
            try:
                numeric_value = float(new_value)
            except (TypeError, ValueError):
                invalid.append(cell)
                continue
            updates.append(
                {"ticker": str(ticker), "metric_name": metric_name, "value": numeric_value}
            )
    return updates, cleared, invalid


def _metric_coverage(df: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    """Present/missing counts per metric, so the N/A settings below have context."""
    total = len(df)
    rows = []
    for metric_name in metric_names:
        if metric_name not in df.columns:
            continue
        missing = int(df[metric_name].isna().sum())
        rows.append(
            {
                "Metric": metric_name,
                "Present": total - missing,
                "Missing": missing,
                "% Missing": (missing / total * 100.0) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _filter_period_rows(
    df: pd.DataFrame,
    *,
    search: str,
    sector: str,
    only_missing: bool,
    metric_names: list[str],
) -> pd.DataFrame:
    """Narrow a 500-row period down to the securities the user is actually working on."""
    filtered = df
    text = search.strip().lower()
    if text:
        searchable = [column for column in ("ticker", "name") if column in filtered.columns]
        if searchable:
            matches = pd.Series(False, index=filtered.index)
            for column in searchable:
                matches |= (
                    filtered[column]
                    .astype("string")
                    .str.lower()
                    # regex=False: a ticker like "BRK.B" or a stray "(" must stay literal.
                    .str.contains(text, na=False, regex=False)
                )
            filtered = filtered[matches]
    if sector and "sector" in filtered.columns:
        filtered = filtered[filtered["sector"] == sector]
    if only_missing:
        present_metrics = [name for name in metric_names if name in filtered.columns]
        if present_metrics:
            filtered = filtered[filtered[present_metrics].isna().any(axis=1)]
    return filtered


def _editor_key_suffix(period_name: str, search: str, sector: str, only_missing: bool) -> str:
    """
    Give the editor a new identity whenever the visible rows change.

    st.data_editor tracks pending edits by row position, so reusing one key across two
    different filters would replay an edit onto whichever row now sits at that position.
    """
    raw = f"{period_name}|{search.strip().lower()}|{sector}|{int(only_missing)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


@st.cache_data(show_spinner=False)
def _period_export_bytes(df: pd.DataFrame) -> bytes:
    """Single-sheet xlsx of the rows currently on screen."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Period")
    return buffer.getvalue()


def _refresh_period_content(client, period_name: str) -> None:
    """
    Reload the table after an edit so the outcome is visible straight away.

    Dropping the cached content instead would blank the table and hide the report until
    the user pressed Load content again.
    """
    try:
        st.session_state["period_content"] = client.get_period_content(period_name)
    except ApiError:
        st.session_state.pop("period_content", None)


def _period_editor_column_config(columns: list[str]) -> dict[str, Any]:
    """
    Make the grid say what it does: only metric columns are writable.

    Identity columns are locked because the save path never sends them, and the internal
    security_id is hidden entirely.
    """
    config: dict[str, Any] = {
        column: None for column in _HIDDEN_EDITOR_COLUMNS if column in columns
    }
    for column in _LOCKED_EDITOR_COLUMNS:
        if column in columns:
            config[column] = st.column_config.Column(
                column,
                disabled=True,
                pinned=column == "ticker",
                help="Read-only. Only metric values can be edited here.",
            )
    return config


def _confirm_pending_action(
    state_key: str,
    prompt: str,
    confirm_label: str,
    key_prefix: str,
) -> bool:
    """
    Render the confirm/cancel pair for a pending destructive action.

    Returns True only when the user confirms. Cancelling clears the pending action and
    reruns, so the caller never sees it again.
    """
    st.warning(prompt)
    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        confirmed = st.button(confirm_label, type="primary", key=f"{key_prefix}_confirm")
    with cancel_col:
        if st.button("Cancel", key=f"{key_prefix}_cancel"):
            st.session_state.pop(state_key, None)
            st.rerun()
    return confirmed


def _render_save_report(report: dict) -> None:
    """
    Report the outcome of the last edit to this period.

    Covers saves, including the edits the API could not accept, and removals, which
    survive the rerun that reloads the table.
    """
    removed = report.get("removed")
    if removed:
        st.success(removed)
    saved = report.get("saved", 0)
    if saved:
        st.success(f"Updated {saved} values.")
    cleared = report.get("cleared", [])
    if cleared:
        st.warning(
            f"{len(cleared)} cell(s) were blanked. The API cannot store an empty value, so "
            f"they were left unchanged: {_truncated_list(cleared)}"
        )
    invalid = report.get("invalid", [])
    if invalid:
        st.error(
            f"{len(invalid)} cell(s) are not numeric and were not saved: "
            f"{_truncated_list(invalid)}"
        )


render_page_header("Periods", "Create periods from Excel/CSV, view and edit content, remove securities/metrics, delete period.")

client = get_api_client()
render_sidebar_api_status(client)

# Fetched once and shared: Streamlit renders every tab on each run, so fetching inside
# each tab issued the same request three times.
try:
    periods = client.list_periods()
    periods_error = ""
except ApiError as exc:
    periods = []
    periods_error = f"Cannot load periods: {exc}"

st.divider()
tabs = st.tabs(["Create", "View & Edit", "Delete"])

# --- Create tab ---
with tabs[0]:
    render_section(
        "Create period from file",
        "Upload Bloomberg, BQL, Reuters, or single-variable Excel (.xlsx, .xls).",
    )
    upload_success_message = st.session_state.pop("period_upload_success", None)
    if upload_success_message:
        st.success(upload_success_message)
    variable_upload_result = st.session_state.pop("period_variable_upload_result", None)
    if variable_upload_result:
        _render_variable_upload_result(variable_upload_result)

    reader = st.selectbox(
        "Reader",
        options=["bloomberg", "bql", "reuters_metrics", _VARIABLE_READER],
        key="period_create_reader",
        format_func=lambda value: _READER_LABELS[value],
    )

    target_period = None
    sheet_name = ""
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

        existing_periods_for_append = periods
        if periods_error:
            st.warning(periods_error)

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
    elif reader == _VARIABLE_READER:
        st.markdown(
            "One variable observed at several periods. **Row 1** = index name (A1) and period "
            "date (B1); **row 2** = field labels (ignored); **row 3+** = the data, in either "
            "layout:"
        )
        st.markdown(
            "- **5 columns per period** — ticker, long name, GICS sector, GICS industry "
            "group, value. Carries enough to **create** securities that are new to the "
            "database.\n"
            "- **2 columns per period** — ticker, value. Values are appended to securities "
            "that **already exist**; unknown tickers are reported and skipped."
        )
        st.caption(
            "The sheet name becomes the metric name. Every period in the file is imported in "
            "append mode: this variable replaces its own previous values and the period's other "
            "metrics are untouched. Periods that do not exist yet are created. The index name is "
            "reported back for checking only — index membership is never rewritten."
        )
    else:
        st.caption("Bloomberg uploads infer the period directly from the file.")

    file = st.file_uploader(
        "Select file",
        type=["xlsx", "xls"],
        key="period_create_file",
    )
    if reader == _VARIABLE_READER:
        sheet_name = st.text_input(
            "Sheet",
            key="period_variable_sheet",
            help="Leave empty to read the first sheet. The sheet name becomes the metric name.",
        )
    elif reader in ("bloomberg", "bql"):
        upload_behavior = st.selectbox(
            "If period exists",
            options=["replace", "append"],
            index=0,
            key="period_if_exists",
            help="replace = overwrite; append = merge new metrics/securities",
        )
    if file and reader == _VARIABLE_READER:
        workbook_sheets = _render_workbook_sheets(file.getvalue())
        if workbook_sheets:
            target_sheet = sheet_name.strip() or workbook_sheets[0]
            st.info(
                f"Sheets in this workbook: {_truncated_list(workbook_sheets)}. "
                f"Reading **{target_sheet}**, so the metric will be named **{target_sheet}**."
            )

    if file and reader != _VARIABLE_READER:
        preview_col, clear_col = st.columns([1, 3])
        with preview_col:
            if st.button("Preview import", key="period_preview_btn"):
                try:
                    with st.spinner("Parsing the file…"):
                        st.session_state["period_preview"] = client.preview_period_file(
                            file.getvalue(),
                            file.name,
                            reader=reader,
                            if_period_exists=upload_behavior,
                            period=target_period,
                        )
                except ApiError as exc:
                    st.session_state.pop("period_preview", None)
                    st.error(str(exc))
        with clear_col:
            if st.session_state.get("period_preview") and st.button(
                "Clear preview", key="period_preview_clear_btn"
            ):
                st.session_state.pop("period_preview", None)
                st.rerun()

        preview = st.session_state.get("period_preview")
        if preview:
            with st.container(border=True):
                _render_upload_preview(preview)

    button_label = "Upload variable" if reader == _VARIABLE_READER else "Create period"
    if st.button(button_label, type="primary", key="period_create_btn"):
        if not file:
            st.error("Select a file first.")
        elif reader == _VARIABLE_READER:
            try:
                with st.spinner("Importing every period in the file — around a second each…"):
                    st.session_state["period_variable_upload_result"] = client.upload_variable_file(
                        file.read(),
                        file.name,
                        sheet=sheet_name.strip() or None,
                    )
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))
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
                action_verbs = {
                    "create": "created",
                    "replace": "replaced (previous contents overwritten)",
                    "append": "merged into the existing period",
                }
                action = result.get("action", "create")
                st.session_state["period_upload_success"] = (
                    f"Period '{result.get('period', '')}' "
                    f"{action_verbs.get(action, 'uploaded')}. "
                    f"Companies: {result.get('companies_count', 0)}, "
                    f"Metrics: {result.get('metrics_count', 0)}, "
                    f"Records: {result.get('records_count', 0)}."
                )
                st.session_state.pop("period_preview", None)
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))

# --- View & Edit tab (merged Get + Edit) ---
with tabs[1]:
    render_section("View & Edit period", "Load a period, edit values in the table, then save. Or remove a security or metric from the period.")
    if periods_error:
        st.error(periods_error)

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

        save_report = st.session_state.pop("period_save_report", None)
        if save_report:
            _render_save_report(save_report)

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
            db_metrics: list[dict] = []
            metric_ids_map: dict[str, int] = {}
            try:
                db_metrics = client.list_db_metrics()
                metric_ids_map = {m["metric_name"]: m["metric_id"] for m in db_metrics if m.get("metric_id")}
            except ApiError as exc:
                st.warning(
                    f"Could not load metric definitions: {exc}. Removing a metric and editing "
                    "metric parameters are unavailable until the API responds."
                )

            st.markdown("**Edit metric values in the table, then click Save changes.**")
            st.caption(
                "Ticker, name, sector and industry are read-only: this page saves metric "
                "values only. Use the commands below to remove a security or a metric."
            )

            coverage_df = _metric_coverage(df, metrics)
            if not coverage_df.empty:
                total_missing = int(coverage_df["Missing"].sum())
                with st.expander(
                    f"Metric coverage — {total_missing:,} missing value(s) across {len(df):,} securities",
                    expanded=False,
                ):
                    st.dataframe(
                        coverage_df,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "% Missing": st.column_config.NumberColumn("% Missing", format="%.1f%%")
                        },
                    )

            filter_search_col, filter_sector_col, filter_missing_col = st.columns([3, 2, 2])
            with filter_search_col:
                row_search = st.text_input(
                    "Search ticker or name",
                    key="period_row_search",
                    placeholder="e.g. AAPL or Apple",
                )
            with filter_sector_col:
                sector_values = (
                    sorted(df["sector"].dropna().unique().tolist())
                    if "sector" in df.columns
                    else []
                )
                sector_label = st.selectbox(
                    "Sector",
                    options=["(All sectors)"] + sector_values,
                    key="period_row_sector",
                )
                sector_choice = "" if sector_label == "(All sectors)" else sector_label
            with filter_missing_col:
                only_missing = st.checkbox(
                    "Only rows with missing values",
                    key="period_row_only_missing",
                    help="Show securities where at least one metric in this period is empty.",
                )

            view_df = _filter_period_rows(
                df,
                search=row_search,
                sector=sector_choice,
                only_missing=only_missing,
                metric_names=metrics,
            )
            filtered = len(view_df) != len(df)
            if filtered:
                st.caption(
                    f"Showing {len(view_df):,} of {len(df):,} securities. "
                    "Changing a filter discards unsaved edits."
                )

            if view_df.empty:
                st.info("No rows match the current filters. Clear them to edit values.")
                edited_df = view_df
            else:
                edited_df = st.data_editor(
                    view_df,
                    width="stretch",
                    key=(
                        "period_data_editor_"
                        + _editor_key_suffix(period_name, row_search, sector_choice, only_missing)
                    ),
                    num_rows="fixed",
                    column_config=_period_editor_column_config(list(view_df.columns)),
                )
                st.download_button(
                    f"Export {len(view_df):,} row(s) to Excel",
                    data=_period_export_bytes(view_df),
                    file_name=f"Period_{period_name.replace('/', '-').replace(' ', '')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="period_export_xlsx",
                    help="Exports exactly the rows shown above, filters included.",
                )

            st.divider()
            st.markdown("**Commands**")

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Save changes", type="primary", key="period_save_btn"):
                    updates, cleared, invalid = _collect_period_edits(df, edited_df, metrics)
                    if not updates and not cleared and not invalid:
                        st.info("No changes to save.")
                    elif not updates:
                        _render_save_report({"saved": 0, "cleared": cleared, "invalid": invalid})
                    else:
                        try:
                            result = client.edit_period(period_name, update_values=updates)
                            st.session_state["period_save_report"] = {
                                "saved": result.get("updated_values", 0),
                                "cleared": cleared,
                                "invalid": invalid,
                            }
                            _refresh_period_content(client, period_name)
                            st.rerun()
                        except ApiError as exc:
                            st.error(str(exc))

            with c2:
                st.caption("Remove securities from this period")
                tickers = sorted(df["ticker"].dropna().unique().tolist())
                securities_to_remove = st.multiselect(
                    "Securities",
                    options=tickers,
                    key="period_remove_sec_select",
                    label_visibility="collapsed",
                    placeholder="Select securities",
                )
                pending_securities = st.session_state.get(_REMOVE_SECURITY_STATE_KEY) or []
                if pending_securities:
                    confirmed = _confirm_pending_action(
                        _REMOVE_SECURITY_STATE_KEY,
                        f"Remove {len(pending_securities)} securit"
                        f"{'y' if len(pending_securities) == 1 else 'ies'} and all their values "
                        f"from {period_name}? {_truncated_list(pending_securities)}",
                        "Remove securities",
                        "period_remove_sec",
                    )
                    if confirmed:
                        security_ids = (
                            df[df["ticker"].isin(pending_securities)]["security_id"]
                            .dropna()
                            .unique()
                            .tolist()
                        )
                        if not security_ids:
                            st.error("Could not resolve any security_id.")
                        else:
                            try:
                                client.edit_period(
                                    period_name,
                                    remove_securities=[int(x) for x in security_ids],
                                )
                                st.session_state.pop(_REMOVE_SECURITY_STATE_KEY, None)
                                # The multiselect still holds tickers that no longer exist.
                                st.session_state.pop("period_remove_sec_select", None)
                                st.session_state["period_save_report"] = {
                                    "saved": 0,
                                    "removed": (
                                        f"Removed {len(pending_securities)} securit"
                                        f"{'y' if len(pending_securities) == 1 else 'ies'}: "
                                        f"{_truncated_list(pending_securities)}"
                                    ),
                                }
                                _refresh_period_content(client, period_name)
                                st.rerun()
                            except ApiError as exc:
                                st.error(str(exc))
                elif securities_to_remove:
                    if st.button("Remove securities", key="period_remove_sec_btn"):
                        st.session_state[_REMOVE_SECURITY_STATE_KEY] = list(securities_to_remove)
                        st.rerun()

            with c3:
                st.caption("Remove metrics from this period")
                metrics_to_remove = st.multiselect(
                    "Metrics",
                    options=sorted(metrics),
                    key="period_remove_met_select",
                    label_visibility="collapsed",
                    placeholder="Select metrics",
                )
                pending_metrics = st.session_state.get(_REMOVE_METRIC_STATE_KEY) or []
                if pending_metrics:
                    affected_values = sum(
                        int(df[name].notna().sum()) for name in pending_metrics if name in df.columns
                    )
                    confirmed = _confirm_pending_action(
                        _REMOVE_METRIC_STATE_KEY,
                        f"Remove {len(pending_metrics)} metric(s) from {period_name}? This deletes "
                        f"{affected_values:,} value(s) in this period. "
                        f"{_truncated_list(pending_metrics)}",
                        "Remove metrics",
                        "period_remove_met",
                    )
                    if confirmed:
                        metric_ids = [
                            metric_ids_map[name] for name in pending_metrics if metric_ids_map.get(name)
                        ]
                        unresolved = [name for name in pending_metrics if not metric_ids_map.get(name)]
                        if not metric_ids:
                            st.error("Could not resolve any metric_id.")
                        else:
                            try:
                                client.edit_period(period_name, remove_metrics=metric_ids)
                                st.session_state.pop(_REMOVE_METRIC_STATE_KEY, None)
                                # The multiselect still holds metrics that no longer exist.
                                st.session_state.pop("period_remove_met_select", None)
                                removed_message = (
                                    f"Removed {len(metric_ids)} metric(s): "
                                    f"{_truncated_list([n for n in pending_metrics if n not in unresolved])}"
                                )
                                if unresolved:
                                    removed_message += (
                                        f" Skipped (no metric id): {_truncated_list(unresolved)}"
                                    )
                                st.session_state["period_save_report"] = {
                                    "saved": 0,
                                    "removed": removed_message,
                                }
                                _refresh_period_content(client, period_name)
                                st.rerun()
                            except ApiError as exc:
                                st.error(str(exc))
                elif metrics_to_remove:
                    if st.button("Remove metrics", key="period_remove_met_btn"):
                        st.session_state[_REMOVE_METRIC_STATE_KEY] = list(metrics_to_remove)
                        st.rerun()

            st.divider()
            st.markdown("**Metric parameters (higher is better & N/A treatment) — global**")
            st.warning(
                "These belong to the metric definition, not to this period. Saving them changes "
                "how the metric behaves in **every** period that uses it, and in every ranking, "
                "portfolio and backtest built from it.",
                icon="⚠️",
            )

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
    if periods_error:
        st.error(periods_error)

    if not periods:
        st.info("No periods to delete.")
    else:
        delete_flash = st.session_state.pop("period_delete_flash", None)
        if delete_flash:
            st.success(delete_flash)

        delete_period = st.selectbox(
            "Select period to delete",
            options=periods,
            key="period_delete_select",
        )
        st.warning(
            f"Delete **{delete_period}**? This removes all fundamental values and index "
            "membership for the period. It cannot be undone."
        )
        typed_period = st.text_input(
            "Type the period name to confirm",
            key="period_delete_confirm_text",
            placeholder=delete_period,
            help="The name must match exactly before the delete button becomes available.",
        )
        delete_confirmed = typed_period.strip() == delete_period
        if st.button(
            "Delete period",
            type="primary",
            key="period_delete_btn",
            disabled=not delete_confirmed,
        ):
            try:
                client.delete_period(delete_period)
                st.session_state["period_delete_flash"] = f"Period '{delete_period}' deleted."
                st.session_state.pop("period_content", None)
                st.session_state.pop("period_name", None)
                # These selectors still hold the deleted period, which is no longer an option.
                st.session_state.pop("period_delete_select", None)
                st.session_state.pop("period_view_select", None)
                st.session_state.pop("period_delete_confirm_text", None)
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))
