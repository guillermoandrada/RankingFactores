"""Metrics Operations - Create, Get & Edit, Delete user-made (derived) metrics."""

from __future__ import annotations

import streamlit as st

from frontend_streamlit.api_client import ApiError, RankingApiClient


st.set_page_config(page_title="Metrics Operations", layout="wide")
st.title("Metrics Operations")
st.caption("Manage user-made derived metrics. Base metrics are only loaded when building formulas.")


def _client() -> RankingApiClient:
    api_url = st.sidebar.text_input(
        "API Base URL",
        value="http://127.0.0.1:8000",
        key="metrics_ops_api_url",
    )
    return RankingApiClient(api_url)


client = _client()

if st.sidebar.button("Test API connection", key="metrics_ops_test_api"):
    try:
        derived = client.list_derived_metrics()
        st.sidebar.success(f"Connected. Derived metrics: {len(derived)}")
    except ApiError as exc:
        st.sidebar.error(str(exc))

tabs = st.tabs(["Create", "Get & Edit", "Delete"])

# --- Create tab ---
with tabs[0]:
    st.subheader("Create Derived Metric")
    st.caption(
        "Build a derived metric from existing metrics (DB + derived). "
        "Operations are applied left-to-right (e.g. A + B / C)."
    )
    # Fetch metrics only when building the formula
    try:
        available_metrics = client.list_metrics()
        metric_names = sorted([m["metric_name"] for m in available_metrics])
    except ApiError as exc:
        st.error(f"Unable to fetch available metrics: {exc}")
        metric_names = []

    if metric_names:
        if "create_metric_chain" not in st.session_state:
            st.session_state.create_metric_chain = [None, None]
        if "create_op_chain" not in st.session_state:
            st.session_state.create_op_chain = ["+"]

        def add_metric():
            st.session_state.create_metric_chain.append(None)
            st.session_state.create_op_chain.append("+")

        def remove_metric():
            if len(st.session_state.create_metric_chain) > 2:
                st.session_state.create_metric_chain.pop()
                st.session_state.create_op_chain.pop()

        for i in range(len(st.session_state.create_metric_chain)):
            c1, c2 = st.columns(2)
            with c1:
                idx = min(i, len(metric_names) - 1)
                st.session_state.create_metric_chain[i] = st.selectbox(
                    f"Metric {i + 1}",
                    options=metric_names,
                    index=idx,
                    key=f"create_metric_{i}",
                )
            with c2:
                if i < len(st.session_state.create_op_chain):
                    st.selectbox(
                        f"Op after {i + 1}",
                        options=["+", "-", "*", "/"],
                        index=0,
                        key=f"create_op_{i}",
                    )
            if i < len(st.session_state.create_metric_chain) - 1:
                st.caption(f"  {st.session_state.get(f'create_op_{i}', '+')}  ")

        st.button("+ Add metric", on_click=add_metric)
        if len(st.session_state.create_metric_chain) > 2:
            st.button("- Remove last", on_click=remove_metric)

        chain_names = [
            st.session_state.get(f"create_metric_{i}")
            for i in range(len(st.session_state.create_metric_chain))
            if st.session_state.get(f"create_metric_{i}")
        ]
        chain_ops = [
            st.session_state.get(f"create_op_{i}", "+")
            for i in range(max(0, len(chain_names) - 1))
        ]
        default_name = " / ".join(chain_names) if len(chain_names) >= 2 else ""
        new_metric_name = st.text_input("New metric name", value=default_name)

        hib_choice = st.selectbox(
            "higher_is_better",
            options=["leave as is", "true", "false"],
            index=0,
            key="create_hib",
        )
        higher_is_better = None if hib_choice == "leave as is" else (hib_choice == "true")

        na_choice = st.selectbox(
            "na_handling",
            options=["leave as is", "replace_with_zero", "replace_with_high", "replace_with_low", "eliminate"],
            index=0,
            key="create_na",
        )
        na_handling = None if na_choice == "leave as is" else na_choice

        if st.button("Create metric", type="primary", key="create_btn"):
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
                    st.success("Derived metric created successfully.")
                    st.json(result)
                except ApiError as exc:
                    st.error(str(exc))
    else:
        st.info("No metrics available. Upload data via the Periods page to create base metrics.")

# --- Get & Edit tab ---
with tabs[1]:
    st.subheader("Get & Edit Derived Metric")
    st.caption("View and edit a derived metric. Base metrics are loaded when editing the formula.")
    try:
        derived_list = client.list_derived_metrics()
    except ApiError as exc:
        st.error(f"Unable to fetch derived metrics: {exc}")
        derived_list = []

    if not derived_list:
        st.info("No derived metrics. Create one in the Create tab.")
    else:
        derived_names = sorted([m["metric_name"] for m in derived_list])
        selected = st.selectbox(
            "Select metric",
            options=derived_names,
            key="edit_select",
        )
        if selected:
            try:
                metric = client.get_derived_metric(selected)
            except ApiError as exc:
                st.error(str(exc))
                metric = {}
            else:
                st.json(metric)
                st.divider()
                st.markdown("**Edit**")
                # Fetch available metrics for formula editing (only when editing)
                try:
                    edit_available = client.list_metrics()
                    edit_metric_names = sorted([m["metric_name"] for m in edit_available])
                except ApiError:
                    edit_metric_names = []

                m_names = metric.get("metric_names", [])
                m_ops = metric.get("operations", [])

                hib_choice = st.selectbox(
                    "higher_is_better",
                    options=["unchanged", "true", "false"],
                    index=0,
                    key="edit_hib",
                )
                na_choice = st.selectbox(
                    "na_handling",
                    options=["unchanged", "replace_with_zero", "replace_with_high", "replace_with_low", "eliminate"],
                    index=0,
                    key="edit_na",
                )

                st.caption("Edit formula (metric_names and operations):")
                new_m_names = []
                new_m_ops = []
                if edit_metric_names and len(m_names) >= 2:
                    for i in range(len(m_names)):
                        idx = edit_metric_names.index(m_names[i]) if m_names[i] in edit_metric_names else 0
                        new_m_names.append(
                            st.selectbox(
                                f"Metric {i + 1}",
                                options=edit_metric_names,
                                index=idx,
                                key=f"edit_m_{i}",
                            )
                        )
                        if i < len(m_ops):
                            op_idx = ["+", "-", "*", "/"].index(m_ops[i]) if m_ops[i] in ["+", "-", "*", "/"] else 0
                            new_m_ops.append(
                                st.selectbox(
                                    f"Op after metric {i + 1}",
                                    options=["+", "-", "*", "/"],
                                    index=op_idx,
                                    key=f"edit_o_{i}",
                                )
                            )

                if st.button("Save changes", type="primary", key="edit_save"):
                    body = {}
                    if hib_choice != "unchanged":
                        body["higher_is_better"] = hib_choice == "true"
                    if na_choice != "unchanged":
                        body["na_handling"] = na_choice
                    if new_m_names and len(new_m_names) >= 2 and len(new_m_ops) == len(new_m_names) - 1:
                        body["metric_names"] = new_m_names
                        body["operations"] = new_m_ops
                    if not body:
                        st.warning("No changes to save.")
                    else:
                        try:
                            client.update_derived_metric(selected, **body)
                            st.success(f"Updated '{selected}'.")
                            st.rerun()
                        except ApiError as exc:
                            st.error(str(exc))

# --- Delete tab ---
with tabs[2]:
    st.subheader("Delete Derived Metric")
    st.caption("Remove a derived metric. This only deletes the formula, not historical data.")
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
