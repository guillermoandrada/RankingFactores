"""Graph and JSON previews."""

from __future__ import annotations

from typing import Any

import streamlit as st

from streamlit_app.profile_editor.profile_store import flat_to_export_payload


def _build_dot(nodes: dict[str, Any], root_id: str) -> str:
    """Build GraphViz DOT for tree. Metrics use unique ids to avoid collisions."""
    lines = ["digraph G {", "  rankdir=TB;", "  node [shape=box];"]
    metric_count = [0]

    def visit(nid: str) -> None:
        node = nodes.get(nid)
        if not node:
            return
        name = node.get("name", nid).replace('"', '\\"')
        lines.append(f'  "{nid}" [label="{name}"];')
        for c in node.get("children", []):
            cid = c.get("child_id", "")
            if cid in nodes:
                visit(cid)
                lines.append(f'  "{nid}" -> "{cid}";')
            elif cid:
                metric_count[0] += 1
                mid = f"m_{metric_count[0]}"
                cid_safe = cid.replace('"', '\\"')[:25]
                lines.append(f'  "{mid}" [label="{cid_safe}", shape=ellipse];')
                lines.append(f'  "{nid}" -> "{mid}";')

    if root_id:
        visit(root_id)
    lines.append("}")
    return "\n".join(lines)


def render_previews(
    nodes: dict[str, Any],
    root_id: str,
    normalization: str,
    winsorization: Any,
    winsor_mode: str,
    key_prefix: str,  # kept for API compatibility
) -> None:
    """Render Graph and JSON tabs."""
    tab1, tab2 = st.tabs(["Graph", "JSON"])
    with tab1:
        dot = _build_dot(nodes, root_id)
        try:
            st.graphviz_chart(dot, use_container_width=True)
        except Exception:
            st.code(dot, language="dot")
    with tab2:
        payload = flat_to_export_payload(nodes, root_id, normalization, winsorization, winsor_mode)
        st.json(payload)
