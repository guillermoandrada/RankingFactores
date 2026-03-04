"""Scoring Profiles - View, edit, and delete in a single page."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from frontend_streamlit.api_client import ApiError, RankingApiClient


st.set_page_config(page_title="Scoring Profiles", layout="wide")
st.title("Scoring Profiles")
st.caption("View, edit, and delete scoring profiles. Edit nodes and overrides, then save.")


def _client() -> RankingApiClient:
    api_url = st.sidebar.text_input(
        "API Base URL",
        value="http://127.0.0.1:8000",
        key="scoring_profiles_api_url",
    )
    return RankingApiClient(api_url)


def _build_hierarchy(nodes: dict[str, Any]) -> list[tuple[str, list[tuple[str, float]]]]:
    """Build hierarchy: (node_name, [(input, weight), ...]) ordered by level."""
    nodes = nodes or {}
    if not nodes:
        return []

    if "Scoring" in nodes:
        top = "Scoring"
    elif "score" in nodes:
        top = "score"
    else:
        top = next(iter(nodes), None)
    if not top:
        return []

    result: list[tuple[str, list[tuple[str, float]]]] = []
    level_nodes = [top]
    seen = {top}

    while level_nodes:
        for node_name in level_nodes:
            inputs = (nodes.get(node_name) or {}).get("inputs") or {}
            rows = [(inp, float(w)) for inp, w in inputs.items()]
            result.append((node_name, rows))
        next_level = []
        for node_name in level_nodes:
            inputs = (nodes.get(node_name) or {}).get("inputs") or {}
            for inp in inputs:
                if inp in nodes and inp not in seen:
                    seen.add(inp)
                    next_level.append(inp)
        level_nodes = next_level

    for name, data in nodes.items():
        if name not in seen:
            inputs = (data or {}).get("inputs") or {}
            rows = [(inp, float(w)) for inp, w in inputs.items()]
            result.append((name, rows))
    return result


def _inputs_to_nodes(
    node_inputs: list[tuple[str, list[tuple[str, float]]]],
) -> dict[str, Any]:
    """Convert dropdown/weight groups back into nodes dict."""
    out: dict[str, Any] = {}
    for node_name, rows in node_inputs:
        inputs: dict[str, float] = {}
        for inp, w in rows:
            inp = str(inp or "").strip()
            if not inp:
                continue
            try:
                inputs[inp] = float(w)
            except (TypeError, ValueError):
                inputs[inp] = 0.0
        out[node_name] = {"inputs": inputs}
    return out


def _format_overrides(overrides: dict[str, Any]) -> str:
    """Format overrides for readable display."""
    if not overrides:
        return "None"
    lines = []
    for scope, items in overrides.items():
        if not isinstance(items, dict):
            continue
        if not items:
            lines.append(f"**{scope.title()}:** (none)")
            continue
        for name, data in items.items():
            patch = (data or {}).get("patch") if isinstance(data, dict) else {}
            if not patch:
                lines.append(f"**{scope.title()}:** {name} — (empty patch)")
            else:
                for path, val in patch.items():
                    val_str = json.dumps(val) if not isinstance(val, (int, float)) else str(val)
                    lines.append(f"**{scope.title()}:** {name} — `{path}` → {val_str}")
    return "\n".join(lines) if lines else "None"


client = _client()

if st.sidebar.button("Test API connection", key="scoring_profiles_test"):
    try:
        profiles = client.list_scoring_profiles()
        st.sidebar.success(f"Connected. Profiles: {len(profiles)}")
    except ApiError as exc:
        st.sidebar.error(str(exc))

try:
    profiles = client.list_scoring_profiles()
    profile_names = sorted(profiles.keys())
except ApiError as exc:
    st.error(f"Cannot load profiles: {exc}")
    profiles = {}
    profile_names = []

if not profile_names:
    st.info("No scoring profiles found.")
    st.stop()

selected = st.selectbox(
    "Select profile",
    options=profile_names,
    key="scoring_profile_select",
)

if not selected:
    st.stop()

profile_data = profiles.get(selected, {})
nodes = profile_data.get("nodes") or {}
overrides = profile_data.get("overrides") or {}

st.divider()

# --- Normalization & Winsorization ---
st.subheader("Normalization & Winsorization")
norm_options = ["zscore", "normalized_zscore", "percentile"]
norm_idx = norm_options.index(profile_data.get("normalization", "zscore")) if profile_data.get("normalization") in norm_options else 0
edit_normalization = st.selectbox(
    "Normalization method",
    options=norm_options,
    index=norm_idx,
    key=f"scoring_profile_norm_{selected}",
)
winsor = profile_data.get("winsorization")
winsor_lower = (winsor or {}).get("lower", 0.01) if isinstance(winsor, dict) else 0.01
winsor_upper = (winsor or {}).get("upper", 0.99) if isinstance(winsor, dict) else 0.99
edit_use_winsor = st.checkbox("Use winsorization", value=bool(winsor), key=f"scoring_profile_winsor_use_{selected}")
edit_winsor_lower = st.number_input("Winsor lower", value=winsor_lower, min_value=0.0, max_value=0.49, step=0.01, key=f"scoring_profile_winsor_lower_{selected}")
edit_winsor_upper = st.number_input("Winsor upper", value=winsor_upper, min_value=0.51, max_value=1.0, step=0.01, key=f"scoring_profile_winsor_upper_{selected}")

st.divider()

# --- Fetch metrics for dropdowns ---
try:
    metrics_resp = client.list_metrics()
    metric_options = sorted({m.get("metric_name", "") for m in metrics_resp if m.get("metric_name")})
except ApiError:
    metric_options = []
node_names = list(nodes.keys())
# Include current inputs in case metrics were removed from DB
all_input_names: set[str] = set()
for n, data in nodes.items():
    all_input_names.update((data or {}).get("inputs") or {})
# Dropdown options: metrics + node names + current inputs (keep legacy names)
dropdown_options = [""] + sorted(set(metric_options) | set(node_names) | all_input_names)

# --- Nodes section ---
st.subheader("Nodes")
st.caption("Composition graph by hierarchy. Score at top, then composing factors below.")
hierarchy = _build_hierarchy(nodes)

# Session state for node inputs (supports add/remove rows)
state_key = f"scoring_profile_nodes_{selected}"
if state_key not in st.session_state:
    st.session_state[state_key] = {node_name: list(rows) for node_name, rows in hierarchy}

# Sync state when profile or hierarchy changes (e.g. after rerun from save)
current_state = st.session_state[state_key]
for node_name, rows in hierarchy:
    if node_name not in current_state:
        current_state[node_name] = list(rows)

edited_node_inputs: list[tuple[str, list[tuple[str, float]]]] = []
first_level = True
shown_divider_caption = False

for node_name, _ in hierarchy:
    if first_level:
        st.markdown(f"**{node_name}**")
        first_level = False
    else:
        st.divider()
        if not shown_divider_caption:
            st.caption("Composing factors (lower hierarchy)")
            shown_divider_caption = True
        st.markdown(f"**{node_name}**")

    rows = current_state.get(node_name, [])
    if not rows:
        rows = [("", 0.0)]
        current_state[node_name] = rows

    new_rows: list[tuple[str, float]] = []
    for i, (inp, w) in enumerate(rows):
        col_sel, col_w, col_rm = st.columns([3, 1, 0.4])
        with col_sel:
            idx = dropdown_options.index(inp) if inp in dropdown_options else 0
            chosen = st.selectbox(
                "Input",
                options=dropdown_options,
                index=idx,
                key=f"node_{selected}_{node_name}_inp_{i}",
                label_visibility="collapsed",
            )
        with col_w:
            weight = st.number_input(
                "Weight",
                value=float(w),
                min_value=0.0,
                max_value=10.0,
                step=0.01,
                format="%.4f",
                key=f"node_{selected}_{node_name}_w_{i}",
                label_visibility="collapsed",
            )
        with col_rm:
            if len(rows) > 1 and st.button("—", key=f"node_{selected}_{node_name}_rm_{i}", help="Remove this input"):
                rows.pop(i)
                st.session_state[state_key][node_name] = rows
                st.rerun()
        new_rows.append((chosen, weight))

    # Persist edits from selectbox/number_input into state
    current_state[node_name] = new_rows

    if st.button("+ Add input", key=f"node_{selected}_{node_name}_add"):
        current_state[node_name].append(("", 0.0))
        st.rerun()

    edited_node_inputs.append((node_name, current_state[node_name]))

st.caption("Select metric or factor from dropdown, set weight. Use + Add input to add rows.")

# --- Overrides section ---
st.subheader("Overrides")
st.caption("Sector or industry-specific patches to override node inputs.")
_has_overrides = bool(overrides) and any(overrides.get(k) for k in ("sector", "industry"))
with st.expander("View and edit overrides", expanded=_has_overrides):
    st.markdown(_format_overrides(overrides))
    st.caption("Edit the JSON below to change overrides.")
    overrides_json = st.text_area(
        "Overrides JSON",
        value=json.dumps(overrides, indent=2),
        height=140,
        key=f"scoring_profile_overrides_{selected}",
        label_visibility="collapsed",
    )

# --- Actions ---
st.divider()
st.markdown("**Actions**")
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("Save changes", type="primary", key="scoring_profile_save"):
        try:
            new_nodes = _inputs_to_nodes(edited_node_inputs)
            if not new_nodes:
                st.error("At least one node with inputs is required.")
            else:
                try:
                    new_overrides = json.loads(overrides_json)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid overrides JSON: {e}")
                else:
                    edit_winsorization: Any = {"lower": edit_winsor_lower, "upper": edit_winsor_upper} if edit_use_winsor else False
                    payload = {
                        "nodes": new_nodes,
                        "overrides": new_overrides,
                        "normalization": edit_normalization,
                        "winsorization": edit_winsorization,
                    }
                    client.upsert_scoring_profile(selected, payload)
                    st.success(f"Profile '{selected}' updated.")
                    st.rerun()
        except ApiError as exc:
            st.error(str(exc))

with col2:
    if st.button("Delete profile", key="scoring_profile_delete"):
        try:
            client.delete_scoring_profile(selected)
            st.success(f"Profile '{selected}' deleted.")
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))
    st.caption("Deleting cannot be undone.")
