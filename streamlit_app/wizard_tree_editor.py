"""Wizard steps 2, 3, 4 using Tree Navigator + Node Editor layout."""

from __future__ import annotations

from typing import Any

import streamlit as st

from streamlit_app.profile_editor.profile_store import ProfileStore, flat_store_to_factors_layers
from streamlit_app.profile_editor.tree_nav import render_tree_nav
from streamlit_app.profile_editor.node_editor import render_node_editor
from streamlit_app.profile_editor.validators import validate_nodes

TOP_NODE_NAME = "Scoring"

# st.dialog (1.37+) or st.experimental_dialog (1.35–1.36)
_st_dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)


def _make_add_metric_dialog(store: ProfileStore, nodes: dict[str, Any], metrics: list[str], key_prefix: str):
    """Create a modal dialog for adding a metric. Used when _st_dialog is available."""
    if not _st_dialog:
        return None

    @_st_dialog("Add metric")
    def add_metric_dialog(parent_id: str):
        parent_name = nodes.get(parent_id, {}).get("name", parent_id)
        st.caption(f"Add a metric to **{parent_name}**")
        m = st.selectbox("Select metric", options=[""] + sorted(metrics), key=f"{key_prefix}_dlg_m_sel")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Add", type="primary", key=f"{key_prefix}_dlg_m_add"):
                if m:
                    store.add_metric_child(parent_id, m, 0.0)
                st.session_state.pop(f"{key_prefix}_add_metric_parent", None)
                st.rerun()
        with c2:
            if st.button("Cancel", key=f"{key_prefix}_dlg_m_cancel"):
                st.session_state.pop(f"{key_prefix}_add_metric_parent", None)
                st.rerun()

    return add_metric_dialog


def _make_add_subfactor_dialog(store: ProfileStore, nodes: dict[str, Any], key_prefix: str):
    """Create a modal dialog for adding a subfactor. Used when _st_dialog is available."""
    if not _st_dialog:
        return None

    @_st_dialog("Add subfactor")
    def add_subfactor_dialog(parent_id: str):
        parent_name = nodes.get(parent_id, {}).get("name", parent_id)
        st.caption(f"Add a subfactor to **{parent_name}**")
        name = st.text_input("Subfactor name", value="New factor", key=f"{key_prefix}_dlg_sf_name")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Add", type="primary", key=f"{key_prefix}_dlg_sf_add"):
                st.session_state.pop(f"{key_prefix}_add_subfactor_parent", None)
                if name and name.strip():
                    new_id = store.add_subfactor(parent_id, name.strip())
                    if new_id:
                        st.session_state["wizard_selected_node_id"] = new_id
                st.rerun()
        with c2:
            if st.button("Cancel", key=f"{key_prefix}_dlg_sf_cancel"):
                st.session_state.pop(f"{key_prefix}_add_subfactor_parent", None)
                st.rerun()

    return add_subfactor_dialog


def _empty_base_store() -> ProfileStore:
    """Create store with root Scoring and no children."""
    store = ProfileStore()
    store.load_from_profile({
        "nodes": {TOP_NODE_NAME: {"inputs": {}}},
        "normalization": "zscore",
        "winsorization": False,
    })
    return store


def _handle_step2_dialogs(
    store: ProfileStore,
    nodes: dict[str, Any],
    metrics: list[str],
    key_prefix: str,
) -> bool:
    """Handle add/delete/duplicate dialogs. Returns True if handled (caller should return early)."""
    root_id = store.get_root_id()

    # Delete confirmation
    confirm_del = st.session_state.get(f"{key_prefix}_confirm_delete")
    if confirm_del:
        st.warning(f"Delete **{nodes.get(confirm_del, {}).get('name', confirm_del)}** and its sub-tree?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Confirm delete", type="primary", key=f"{key_prefix}_confirm_del_btn"):
                store.delete_node(confirm_del)
                st.session_state.pop(f"{key_prefix}_confirm_delete", None)
                if st.session_state.get("wizard_selected_node_id") == confirm_del:
                    st.session_state["wizard_selected_node_id"] = root_id
                st.rerun()
        with c2:
            if st.button("Cancel", key=f"{key_prefix}_cancel_del"):
                st.session_state.pop(f"{key_prefix}_confirm_delete", None)
                st.rerun()
        return True

    # Duplicate
    dup_node = st.session_state.get(f"{key_prefix}_duplicate_node")
    if dup_node:
        default_name = f"{nodes.get(dup_node, {}).get('name', 'Copy')} (copy)"
        new_name = st.text_input("Name for duplicate", value=default_name, key=f"{key_prefix}_dup_name")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Create duplicate", type="primary", key=f"{key_prefix}_dup_btn"):
                new_id = store.duplicate_subtree(dup_node, new_name or "Copy")
                st.session_state.pop(f"{key_prefix}_duplicate_node", None)
                if new_id:
                    st.session_state["wizard_selected_node_id"] = new_id
                st.rerun()
        with c2:
            if st.button("Cancel", key=f"{key_prefix}_cancel_dup"):
                st.session_state.pop(f"{key_prefix}_duplicate_node", None)
                st.rerun()
        return True

    # Add metric (modal if available, else inline)
    add_metric_parent = st.session_state.get(f"{key_prefix}_add_metric_parent")
    if add_metric_parent:
        add_metric_dialog_fn = _make_add_metric_dialog(store, nodes, metrics, key_prefix)
        if add_metric_dialog_fn:
            add_metric_dialog_fn(add_metric_parent)
            return False  # Modal overlay: let main layout render underneath
        parent_name = nodes.get(add_metric_parent, {}).get("name", add_metric_parent)
        st.info(f"Add metric to **{parent_name}**")
        m = st.selectbox("Select metric", options=[""] + sorted(metrics), key=f"{key_prefix}_add_m_sel")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Add", type="primary", key=f"{key_prefix}_add_m_btn"):
                if m:
                    store.add_metric_child(add_metric_parent, m, 0.0)
                st.session_state.pop(f"{key_prefix}_add_metric_parent", None)
                st.rerun()
        with c2:
            if st.button("Cancel", key=f"{key_prefix}_cancel_add_m"):
                st.session_state.pop(f"{key_prefix}_add_metric_parent", None)
                st.rerun()
        return True

    # Add subfactor (modal if available, else inline)
    add_subfactor_parent = st.session_state.get(f"{key_prefix}_add_subfactor_parent")
    if add_subfactor_parent:
        add_subfactor_dialog_fn = _make_add_subfactor_dialog(store, nodes, key_prefix)
        if add_subfactor_dialog_fn:
            add_subfactor_dialog_fn(add_subfactor_parent)
            return False  # Modal overlay: let main layout render underneath
        parent_name = nodes.get(add_subfactor_parent, {}).get("name", add_subfactor_parent)
        st.info(f"Add subfactor to **{parent_name}**")
        name = st.text_input("Subfactor name", value="New factor", key=f"{key_prefix}_add_sf_name")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Add", type="primary", key=f"{key_prefix}_add_sf_btn"):
                st.session_state.pop(f"{key_prefix}_add_subfactor_parent", None)
                if name and name.strip():
                    new_id = store.add_subfactor(add_subfactor_parent, name.strip())
                    if new_id:
                        st.session_state["wizard_selected_node_id"] = new_id
                st.rerun()
        with c2:
            if st.button("Cancel", key=f"{key_prefix}_cancel_add_sf"):
                st.session_state.pop(f"{key_prefix}_add_subfactor_parent", None)
                st.rerun()
        return True

    return False


def render_step2_base_structure(metrics: list[str], key_prefix: str) -> dict[str, Any]:
    """Step 2: Hierarchy navigator (left) + Node editor (right). Returns factors+layers."""
    store_key = f"{key_prefix}_base_store"
    if store_key not in st.session_state:
        st.session_state[store_key] = _empty_base_store()

    store = st.session_state[store_key]
    nodes = store.get_nodes()
    root_id = store.get_root_id()

    # Sync selection (tree_nav uses profile_editor_selected_node_id)
    global_sel = st.session_state.get("profile_editor_selected_node_id")
    if global_sel and global_sel in nodes:
        st.session_state["wizard_selected_node_id"] = global_sel
    if "wizard_selected_node_id" not in st.session_state:
        st.session_state["wizard_selected_node_id"] = root_id
    selected_id = st.session_state["wizard_selected_node_id"]
    if selected_id not in nodes:
        st.session_state["wizard_selected_node_id"] = root_id
        selected_id = root_id
    st.session_state["profile_editor_selected_node_id"] = selected_id

    # Handle dialogs first
    if _handle_step2_dialogs(store, nodes, metrics, key_prefix):
        factors, layers = flat_store_to_factors_layers(nodes, root_id)
        return {"factors": factors, "layers": layers}

    # Validation
    issues = validate_nodes(nodes, root_id)
    if issues:
        st.warning(" · ".join(i["message"] for i in issues[:3]))

    st.caption("1. Select a node in the hierarchy. 2. Add metrics or subfactors. 3. Open subfactors to build nested structure.")
    st.divider()

    col_tree, col_editor = st.columns([0.32, 0.68])
    with col_tree:
        with st.container():
            st.markdown("#### Hierarchy")
            render_tree_nav(nodes, root_id, selected_id, metrics, key_prefix)
    with col_editor:
        with st.container():
            st.markdown("#### Edit node")
            render_node_editor(
                store,
                selected_id,
                metrics,
                key_prefix,
                selection_state_key="profile_editor_selected_node_id",
                add_metric_key=f"{key_prefix}_add_metric_parent",
                add_subfactor_key=f"{key_prefix}_add_subfactor_parent",
            )

    factors, layers = flat_store_to_factors_layers(nodes, root_id)
    return {"factors": factors, "layers": layers}


    # Override-specific editors were removed; overrides are now modeled as
    # separate profiles instead of patches on a single profile.
