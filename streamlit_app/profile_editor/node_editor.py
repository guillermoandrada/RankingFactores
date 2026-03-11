"""Node editor (right pane). Header, settings, weighted inputs, and actions."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from streamlit_app.profile_editor.profile_store import ProfileStore


def _norm_and_rerun(store: ProfileStore, node_id: str) -> None:
    store.normalize_weights(node_id)
    st.rerun()


def _eq_and_rerun(store: ProfileStore, node_id: str) -> None:
    store.equal_weights(node_id)
    st.rerun()


def _sort_and_rerun(store: ProfileStore, node_id: str) -> None:
    store.sort_children_by_weight(node_id)
    st.rerun()


def _hierarchy_path(nodes: dict[str, Any], root_id: str, target_id: str) -> list[tuple[str, str]]:
    """Return path from root to target as [(id, name), ...]."""
    path: list[tuple[str, str]] = []

    def find(nid: str, trail: list[tuple[str, str]]) -> bool:
        node = nodes.get(nid)
        if not node:
            return False
        trail.append((nid, node.get("name", nid)))
        if nid == target_id:
            return True
        for c in node.get("children", []):
            cid = c.get("child_id", "")
            if cid in nodes and find(cid, trail):
                return True
        trail.pop()
        return False

    find(root_id, path)
    return path


def render_node_editor(
    store: ProfileStore,
    selected_id: str | None,
    metrics: list[str],
    key_prefix: str,
    selection_state_key: str = "profile_editor_selected_node_id",
    add_metric_key: str | None = None,
    add_subfactor_key: str | None = None,
) -> None:
    """Render editor for the selected node."""
    nodes = store.get_nodes()
    root_id = store.get_root_id()

    if not selected_id or selected_id not in nodes:
        st.info("Select a node from the hierarchy on the left.")
        return

    node = nodes[selected_id]
    path = _hierarchy_path(nodes, root_id, selected_id)
    path_names = [n for _, n in path]

    # --- Header ---
    st.markdown(f"## {node.get('name', selected_id)}")
    st.caption(" » ".join(path_names))
    st.divider()

    # --- Node settings ---
    with st.expander("Node settings", expanded=False):
        col_name, col_method = st.columns(2)
        with col_name:
            new_name = st.text_input(
                "Name",
                value=node.get("name", ""),
                key=f"{key_prefix}_name",
                help="Display name in the hierarchy.",
            )
        with col_method:
            method = st.selectbox(
                "Aggregation",
                options=["linear", "softplus"],
                index=["linear", "softplus"].index(node.get("method", "linear")),
                key=f"{key_prefix}_method",
                help="Linear = weighted sum. Softplus = smooth maximum.",
            )
        if new_name != node.get("name"):
            node["name"] = new_name
        node["method"] = method

    # --- Add actions (prominent when keys provided) ---
    if add_metric_key is not None and add_subfactor_key is not None:
        add_col1, add_col2, _ = st.columns([1, 1, 2])
        with add_col1:
            if st.button("+ Add metric", type="primary", key=f"{key_prefix}_add_metric_btn"):
                st.session_state[add_metric_key] = selected_id
                st.rerun()
        with add_col2:
            if st.button("+ Add subfactor", key=f"{key_prefix}_add_subfactor_btn"):
                st.session_state[add_subfactor_key] = selected_id
                st.rerun()
        st.divider()

    # --- Weighted inputs (main focus) ---
    st.markdown("**Weighted inputs**")
    st.caption(
        "Add metrics or subfactors. Each subfactor is a nested node you can open and edit. "
        "Weights must sum to 1.0 across all inputs."
    )

    children = node.get("children", [])
    node_names = set(nodes.keys())

    # Build options: metrics and subfactors by display name
    def _to_display(val: str) -> str:
        if not val:
            return ""
        if val in nodes:
            return nodes[val].get("name", val)
        return val

    subfactor_names = [nodes[nid].get("name", nid) for nid in sorted(node_names)]
    input_options = [""] + sorted(set(metrics) | set(subfactor_names) | {_to_display(c.get("child_id", "")) for c in children if c.get("child_id")})

    rows = []
    for c in children:
        cid = c.get("child_id", "")
        raw_type = "subfactor" if cid in nodes else "metric"
        display_val = _to_display(cid) if cid else ""
        rows.append({
            "child_type": raw_type.capitalize(),
            "child_name": display_val,
            "weight": c.get("weight", 0.0),
            "delete": False,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["child_type", "child_name", "weight", "delete"])
    if df.empty:
        st.info("No inputs yet. Use **+ Add metric** or **+ Add subfactor** above to build the scoring tree.")

    edited = st.data_editor(
        df,
        use_container_width=True,
        key=f"{key_prefix}_children",
        num_rows="fixed",
        column_config={
            "_index": None,  # Hide index column
            "child_type": st.column_config.TextColumn("Type", width="small", disabled=True),
            "child_name": st.column_config.SelectboxColumn("Input", options=input_options, width="medium"),
            "weight": st.column_config.NumberColumn("Weight", format="%.2f", min_value=0.0, max_value=10.0),
            "delete": st.column_config.CheckboxColumn("Delete"),
        },
        hide_index=True,
    )

    def _parse_child_id(display_val: str) -> str:
        """Resolve display name to child_id: subfactor name -> node id, else metric name."""
        val = str(display_val or "").strip()
        if not val:
            return ""
        if val in nodes:
            return val
        for nid, nd in nodes.items():
            if nd.get("name", nid) == val:
                return nid
        return val  # Metric name

    node["children"] = []
    for _, row in edited.iterrows():
        # Skip rows explicitly marked for deletion
        if bool(row.get("delete", False)):
            continue
        display_val = str(row.get("child_name", "")).strip()
        if not display_val:
            continue
        cid = _parse_child_id(display_val)
        if not cid:
            continue
        node["children"].append({
            "child_id": cid,
            "weight": float(row.get("weight", 0)),
            "enabled": True,
        })

    total = sum(c.get("weight", 0) for c in node["children"] if c.get("enabled", True))
    is_valid = abs(total - 1.0) < 0.001

    # --- Weight validation (only when there are inputs) ---
    if node["children"]:
        if is_valid:
            st.success(f"Weight sum: {total:.2f} ✓")
        else:
            st.error(f"Weight sum: {total:.2f} (must be 1.0). Use **Normalize** or **Equal weights** below.")

    # --- Open subfactors (navigate into nested node) ---
    subfactor_children = [c for c in node["children"] if c.get("child_id") in nodes]
    if subfactor_children:
        st.caption("Open subfactor to edit its inputs")
        n = len(subfactor_children)
        sub_cols = st.columns(min(n, 4))
        for i, c in enumerate(subfactor_children):
            cid = c.get("child_id", "")
            sub_name = nodes.get(cid, {}).get("name", cid)
            with sub_cols[i % len(sub_cols)]:
                if st.button(f"→ {sub_name}", key=f"{key_prefix}_open_{cid}", help=f"Edit {sub_name}"):
                    st.session_state[selection_state_key] = cid
                    st.rerun()

    # --- Weight tools ---
    st.divider()
    st.caption("Weight tools")
    wt_col1, wt_col2, wt_col3, wt_col4 = st.columns(4)
    with wt_col1:
        st.button(
            "Normalize",
            key=f"{key_prefix}_norm",
            help="Scale weights to sum to 1.0.",
            on_click=lambda: _norm_and_rerun(store, selected_id),
        )
    with wt_col2:
        st.button(
            "Equal weights",
            key=f"{key_prefix}_eq",
            help="Set all enabled inputs to equal weight.",
            on_click=lambda: _eq_and_rerun(store, selected_id),
        )
    with wt_col3:
        st.button(
            "Sort by weight",
            key=f"{key_prefix}_sort",
            help="Order by weight descending.",
            on_click=lambda: _sort_and_rerun(store, selected_id),
        )
    with wt_col4:
        if not is_valid:
            st.button(
                "Fix",
                key=f"{key_prefix}_fix",
                help="Normalize weights to 1.0.",
                type="primary",
                on_click=lambda: _norm_and_rerun(store, selected_id),
            )
