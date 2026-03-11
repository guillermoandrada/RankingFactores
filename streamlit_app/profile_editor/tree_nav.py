"""Tree navigator (left pane). Hierarchy view with selection and actions.

Selection is stored in st.session_state["profile_editor_selected_node_id"].
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from streamlit_app.profile_editor.validators import validate_nodes


def _weight_from_parent(nodes: dict[str, Any], parent_id: str | None, child_id: str) -> float:
    if not parent_id:
        return 1.0
    parent = nodes.get(parent_id)
    if not parent:
        return 0.0
    for c in parent.get("children", []):
        if c.get("child_id") == child_id:
            return c.get("weight", 0.0)
    return 0.0


def render_tree_nav(
    nodes: dict[str, Any],
    root_id: str,
    selected_id: str | None,
    metrics: list[str],
    key_prefix: str,
) -> None:
    """Render the hierarchy tree in the left pane. Updates selected_node_id on click."""
    issues = validate_nodes(nodes, root_id)
    issue_nodes = {i["node_id"] for i in issues}

    def render_node(node_id: str, parent_id: str | None, depth: int) -> None:
        node = nodes.get(node_id)
        if not node:
            return
        name = node.get("name", node_id)
        method = node.get("method", "linear")
        w = _weight_from_parent(nodes, parent_id, node_id)
        has_issue = node_id in issue_nodes
        is_selected = node_id == selected_id

        indent = " " * depth  # em-space for visual hierarchy
        label = f"{indent}{name}"
        if depth > 0:
            label += f" · w={w:.2f}"
        if has_issue:
            label += " ⚠"

        col_main, col_menu = st.columns([4, 1])
        with col_main:
            btn_kwargs = {
                "key": f"{key_prefix}_nav_{node_id}",
                "use_container_width": True,
            }
            if is_selected:
                btn_kwargs["type"] = "primary"
            if st.button(label, **btn_kwargs):
                st.session_state["profile_editor_selected_node_id"] = node_id
                st.rerun()

        with col_menu:
            with st.popover("⋯", help="Actions"):
                if node_id != root_id:
                    if st.button("Delete", key=f"{key_prefix}_del_{node_id}"):
                        st.session_state[f"{key_prefix}_confirm_delete"] = node_id
                        st.rerun()
                    if st.button("Duplicate", key=f"{key_prefix}_dup_{node_id}"):
                        st.session_state[f"{key_prefix}_duplicate_node"] = node_id
                        st.rerun()
                if st.button("+ Metric", key=f"{key_prefix}_addm_{node_id}"):
                    st.session_state[f"{key_prefix}_add_metric_parent"] = node_id
                    st.rerun()
                if st.button("+ Subfactor", key=f"{key_prefix}_adds_{node_id}"):
                    st.session_state[f"{key_prefix}_add_subfactor_parent"] = node_id
                    st.rerun()

        for c in node.get("children", []):
            cid = c.get("child_id", "")
            if cid in nodes:
                render_node(cid, node_id, depth + 1)
            elif cid:
                wc = c.get("weight", 0)
                st.caption(f"{'  ' * (depth + 1)}· {cid} w={wc:.2f}")

    st.caption("Click a node to edit. Use ⋯ for actions.")
    root = nodes.get(root_id)
    if root:
        render_node(root_id, None, 0)
        children = root.get("children", [])
        if not children:
            st.divider()
            st.info("Select **Scoring** on the right, then use **Add metric** or **Add subfactor** to build the hierarchy.")
