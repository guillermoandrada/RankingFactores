"""Metrics - Create, Get & Edit, Delete user-made (derived) metrics."""

from __future__ import annotations

import streamlit as st

from streamlit_app.api_client import ApiError
from streamlit_app.ui import (
    get_api_client,
    inject_custom_css,
    render_page_header,
    render_section,
    render_sidebar_api_test,
)

# Key prefix for Create tab to avoid collisions with other tabs
_CREATE_KEY = "metrics_create"
_EDIT_KEY = "metrics_edit"

# Human-readable label mapping (display -> API value)
_HIB_OPTIONS = [("Leave as is", None), ("Yes", True), ("No", False)]
_NA_OPTIONS = [
    ("Leave as is", None),
    ("Replace with zero", "replace_with_zero"),
    ("Replace with high", "replace_with_high"),
    ("Replace with low", "replace_with_low"),
    ("Eliminate", "eliminate"),
]

# Edit tab options (Keep current = don't include in update)
_EDIT_HIB_OPTIONS = [("Keep current", None), ("Yes", True), ("No", False)]
_EDIT_NA_OPTIONS = [
    ("Keep current", None),
    ("Replace with zero", "replace_with_zero"),
    ("Replace with high", "replace_with_high"),
    ("Replace with low", "replace_with_low"),
    ("Eliminate", "eliminate"),
]

# Map API na_handling value to human-readable label
_NA_LABEL_BY_VALUE = {v: k for k, v in _NA_OPTIONS if v}


def _render_create_metric_tab(client):
    """Render the Create derived metric tab content."""
    render_section(
        "Create Derived Metric",
        "Build a derived metric from existing metrics (DB + derived). Operations are applied left-to-right (e.g. A + B / C).",
    )
    try:
        available_metrics = client.list_metrics()
        metric_names = sorted([m["metric_name"] for m in available_metrics])
    except ApiError as exc:
        st.error(f"Unable to fetch available metrics: {exc}")
        metric_names = []

    if not metric_names:
        st.info(
            "No metrics available. Go to **Periods** to upload Excel data and create base metrics first."
        )
        return

    chain_key = f"{_CREATE_KEY}_chain"
    op_key = f"{_CREATE_KEY}_op_chain"
    if chain_key not in st.session_state:
        st.session_state[chain_key] = [None, None]
    if op_key not in st.session_state:
        st.session_state[op_key] = ["+"]

    def add_metric():
        st.session_state[chain_key].append(None)
        st.session_state[op_key].append("+")

    def remove_metric():
        if len(st.session_state[chain_key]) > 2:
            st.session_state[chain_key].pop()
            st.session_state[op_key].pop()

    chain = st.session_state[chain_key]
    op_chain = st.session_state[op_key]

    # --- Section 1: Formula ---
    st.markdown("#### Formula")
    for i in range(len(chain)):
        col_metric, col_op = st.columns([4, 1])
        with col_metric:
            idx = min(i, len(metric_names) - 1)
            chain[i] = st.selectbox(
                f"Metric {i + 1}",
                options=metric_names,
                index=idx,
                key=f"{_CREATE_KEY}_metric_{i}",
                help="Select the base or derived metric.",
            )
        with col_op:
            if i < len(op_chain):
                st.selectbox(
                    f"Op after {i + 1}",
                    options=["+", "-", "*", "/"],
                    index=0,
                    key=f"{_CREATE_KEY}_op_{i}",
                    help="Operation between this metric and the next.",
                )

    add_col, remove_col = st.columns(2)
    with add_col:
        st.button("+ Add metric", on_click=add_metric, key=f"{_CREATE_KEY}_add")
    with remove_col:
        if len(chain) > 2:
            st.button("- Remove last", on_click=remove_metric, key=f"{_CREATE_KEY}_remove")

    chain_names = [
        st.session_state.get(f"{_CREATE_KEY}_metric_{i}")
        for i in range(len(chain))
        if st.session_state.get(f"{_CREATE_KEY}_metric_{i}")
    ]
    chain_ops = [
        st.session_state.get(f"{_CREATE_KEY}_op_{i}", "+")
        for i in range(max(0, len(chain_names) - 1))
    ]
    preview_parts = []
    for j, name in enumerate(chain_names):
        preview_parts.append(name)
        if j < len(chain_ops):
            preview_parts.append(f" [{chain_ops[j]}] ")
    formula_preview = "".join(preview_parts) if preview_parts else "(select metrics)"
    st.info(f"**Formula preview:** {formula_preview}")
    st.divider()

    # --- Section 2: Output ---
    st.markdown("#### Output")
    default_name = " / ".join(chain_names) if len(chain_names) >= 2 else ""
    new_metric_name = st.text_input(
        "New metric name",
        value=default_name,
        placeholder="e.g. Debt / Assets",
        help="Name for the new derived metric.",
    )
    st.divider()

    # --- Section 3: Behavior ---
    st.markdown("#### Behavior")
    hib_labels = [x[0] for x in _HIB_OPTIONS]
    hib_values = [x[1] for x in _HIB_OPTIONS]
    hib_choice = st.selectbox(
        "Higher values are better",
        options=hib_labels,
        index=0,
        key=f"{_CREATE_KEY}_hib",
        help="Whether higher values indicate better performance.",
    )
    higher_is_better = hib_values[hib_labels.index(hib_choice)]

    na_labels = [x[0] for x in _NA_OPTIONS]
    na_values = [x[1] for x in _NA_OPTIONS]
    na_choice = st.selectbox(
        "Handle missing values",
        options=na_labels,
        index=0,
        key=f"{_CREATE_KEY}_na",
        help="How to treat NA/missing values in calculations.",
    )
    na_handling = na_values[na_labels.index(na_choice)]
    st.divider()

    # --- Section 4: Actions ---
    st.markdown("#### Actions")
    if st.button("Create metric", type="primary", key=f"{_CREATE_KEY}_btn"):
        if len(chain_names) < 2:
            st.error("Add at least 2 metrics.")
        elif not new_metric_name.strip():
            st.error("New metric name cannot be empty.")
        elif len(chain_ops) != len(chain_names) - 1:
            st.error("Operations count mismatch.")
        else:
            try:
                result = client.create_metric_operation(
                    metric_names=chain_names,
                    operations=chain_ops,
                    new_metric_name=new_metric_name.strip(),
                    higher_is_better=higher_is_better,
                    na_handling=na_handling,
                )
                st.success(f"Derived metric '{new_metric_name.strip()}' created successfully.")
                with st.expander("Raw response"):
                    st.json(result)
            except ApiError as exc:
                st.error(str(exc))


def _render_edit_metric_tab(client):
    """Render the Get & Edit derived metric tab content."""
    render_section(
        "Get & Edit Derived Metric",
        "View and edit a derived metric. Base metrics are loaded when editing the formula.",
    )
    try:
        derived_list = client.list_derived_metrics()
    except ApiError as exc:
        st.error(f"Unable to fetch derived metrics: {exc}")
        derived_list = []

    if not derived_list:
        st.info(
            "No derived metrics yet. Go to the **Create** tab to define your first derived metric."
        )
        return

    derived_names = sorted([m["metric_name"] for m in derived_list])
    selected = st.selectbox(
        "Select metric",
        options=derived_names,
        key=f"{_EDIT_KEY}_select",
        help="Choose which derived metric to view and edit.",
    )
    if not selected:
        return

    try:
        metric = client.get_derived_metric(selected)
    except ApiError as exc:
        st.error(str(exc))
        return

    m_names = metric.get("metric_names", [])
    m_ops = metric.get("operations", [])
    if len(m_ops) < len(m_names) - 1:
        m_ops = m_ops + ["+"] * (len(m_names) - 1 - len(m_ops))

    # --- Section 1: Summary ---
    st.markdown("#### Summary")
    hib_val = metric.get("higher_is_better")
    hib_str = "Yes" if hib_val is True else "No" if hib_val is False else "Not set"
    na_val = metric.get("na_handling")
    na_str = _NA_LABEL_BY_VALUE.get(na_val, na_val or "Not set")
    formula_str = " ".join(
        f"{n} [{o}]" for n, o in zip(m_names, m_ops + [""])
    ).rstrip(" []")
    st.caption(f"**Formula:** {formula_str}")
    st.caption(f"**Higher values better:** {hib_str}  |  **NA handling:** {na_str}")
    with st.expander("Technical details"):
        st.json(metric)
    st.divider()

    # --- Section 2: Behavior ---
    st.markdown("#### Behavior")
    chain_key = f"{_EDIT_KEY}_chain"
    op_key = f"{_EDIT_KEY}_op_chain"
    last_key = f"{_EDIT_KEY}_last_selected"
    if last_key not in st.session_state or st.session_state[last_key] != selected:
        st.session_state[last_key] = selected
        st.session_state[chain_key] = list(m_names)
        st.session_state[op_key] = list(m_ops) if len(m_ops) == len(m_names) - 1 else ["+"] * (len(m_names) - 1)

    hib_labels = [x[0] for x in _EDIT_HIB_OPTIONS]
    hib_values = [x[1] for x in _EDIT_HIB_OPTIONS]
    hib_help = f"Current: {hib_str}. Select 'Keep current' to leave unchanged."
    hib_choice = st.selectbox(
        "Higher values are better",
        options=hib_labels,
        index=0,
        key=f"{_EDIT_KEY}_hib",
        help=hib_help,
    )
    higher_is_better = hib_values[hib_labels.index(hib_choice)]

    na_labels = [x[0] for x in _EDIT_NA_OPTIONS]
    na_values = [x[1] for x in _EDIT_NA_OPTIONS]
    na_help = f"Current: {na_str}. Select 'Keep current' to leave unchanged."
    na_choice = st.selectbox(
        "Handle missing values",
        options=na_labels,
        index=0,
        key=f"{_EDIT_KEY}_na",
        help=na_help,
    )
    na_handling = na_values[na_labels.index(na_choice)]
    st.divider()

    # --- Section 3: Formula ---
    st.markdown("#### Formula")
    try:
        edit_available = client.list_metrics()
        edit_metric_names = sorted([m["metric_name"] for m in edit_available])
    except ApiError:
        edit_metric_names = []

    chain = st.session_state[chain_key]
    op_chain = st.session_state[op_key]

    def add_edit_metric():
        st.session_state[chain_key].append(
            edit_metric_names[0] if edit_metric_names else ""
        )
        st.session_state[op_key].append("+")

    def remove_edit_metric():
        if len(st.session_state[chain_key]) > 2:
            st.session_state[chain_key].pop()
            st.session_state[op_key].pop()

    if edit_metric_names and len(chain) >= 2:
        for i in range(len(chain)):
            col_metric, col_op = st.columns([4, 1])
            with col_metric:
                idx = (
                    edit_metric_names.index(chain[i])
                    if chain[i] in edit_metric_names
                    else 0
                )
                chain[i] = st.selectbox(
                    f"Metric {i + 1}",
                    options=edit_metric_names,
                    index=idx,
                    key=f"{_EDIT_KEY}_m_{i}",
                    help="Select the base or derived metric.",
                )
            with col_op:
                if i < len(op_chain):
                    op_val = op_chain[i] if op_chain[i] in ["+", "-", "*", "/"] else "+"
                    op_idx = ["+", "-", "*", "/"].index(op_val)
                    st.selectbox(
                        f"Op after {i + 1}",
                        options=["+", "-", "*", "/"],
                        index=op_idx,
                        key=f"{_EDIT_KEY}_o_{i}",
                        help="Operation between this metric and the next.",
                    )

        add_col, remove_col = st.columns(2)
        with add_col:
            st.button("+ Add metric", on_click=add_edit_metric, key=f"{_EDIT_KEY}_add")
        with remove_col:
            if len(chain) > 2:
                st.button("- Remove last", on_click=remove_edit_metric, key=f"{_EDIT_KEY}_remove")

        edit_chain_names = [
            st.session_state.get(f"{_EDIT_KEY}_m_{i}")
            for i in range(len(chain))
            if st.session_state.get(f"{_EDIT_KEY}_m_{i}")
        ]
        edit_chain_ops = [
            st.session_state.get(f"{_EDIT_KEY}_o_{i}", "+")
            for i in range(max(0, len(edit_chain_names) - 1))
        ]
        preview_parts = []
        for j, name in enumerate(edit_chain_names):
            preview_parts.append(name)
            if j < len(edit_chain_ops):
                preview_parts.append(f" [{edit_chain_ops[j]}] ")
        formula_preview = "".join(preview_parts) if preview_parts else "(select metrics)"
        st.info(f"**Formula preview:** {formula_preview}")
    else:
        edit_chain_names = chain
        edit_chain_ops = op_chain
        if len(chain) < 2:
            st.warning("This metric has fewer than 2 components. Formula editing is limited.")

    st.divider()

    # --- Section 4: Actions ---
    st.markdown("#### Actions")
    if st.button("Save changes", type="primary", key=f"{_EDIT_KEY}_save"):
        body = {}
        if higher_is_better is not None:
            body["higher_is_better"] = higher_is_better
        if na_handling is not None:
            body["na_handling"] = na_handling
        if edit_metric_names and edit_chain_names and len(edit_chain_names) >= 2 and len(edit_chain_ops) == len(edit_chain_names) - 1:
            body["metric_names"] = edit_chain_names
            body["operations"] = edit_chain_ops
        if not body:
            st.warning("No changes to save.")
        else:
            try:
                client.update_derived_metric(selected, **body)
                st.success(f"Updated '{selected}'.")
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))


st.set_page_config(page_title="Metrics", layout="wide")
inject_custom_css()
render_page_header("Metrics", "Manage user-made derived metrics. Base metrics are only loaded when building formulas.")

client = get_api_client("metrics")
render_sidebar_api_test(client, "metrics_test_api")

st.divider()
tabs = st.tabs(["Create", "Get & Edit", "Delete"])

# --- Create tab ---
with tabs[0]:
    _render_create_metric_tab(client)

# --- Get & Edit tab ---
with tabs[1]:
    _render_edit_metric_tab(client)

# --- Delete tab ---
with tabs[2]:
    render_section("Delete Derived Metric", "Remove a derived metric. This only deletes the formula, not historical data.")
    try:
        derived_list = client.list_derived_metrics()
    except ApiError as exc:
        st.error(f"Unable to fetch derived metrics: {exc}")
        derived_list = []

    if not derived_list:
        st.info("No derived metrics to delete.")
    else:
        delete_names = sorted([m["metric_name"] for m in derived_list])
        to_delete = st.selectbox(
            "Select metric to delete",
            options=delete_names,
            key="delete_select",
        )
        if to_delete:
            st.warning(f"Delete **{to_delete}**? This removes the formula only.")
            if st.button("Delete metric", type="primary", key="delete_btn"):
                try:
                    client.delete_derived_metric(to_delete)
                    st.success(f"Deleted '{to_delete}'.")
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))
