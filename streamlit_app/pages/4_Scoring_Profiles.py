"""Scoring Profiles - Tree Navigator (left) + Node Editor (right)."""

from __future__ import annotations

import streamlit as st

from streamlit_app.api_client import ApiError
from streamlit_app.profile_editor.profile_store import ProfileStore, flat_to_export_payload
from streamlit_app.profile_editor.tree_nav import render_tree_nav
from streamlit_app.profile_editor.node_editor import render_node_editor
from streamlit_app.profile_editor.previews import render_previews
from streamlit_app.profile_editor.validators import validate_nodes
from streamlit_app.ui import get_api_client, inject_custom_css, render_page_header, render_sidebar_api_test
from streamlit_app.wizard_tree_editor import _make_add_metric_dialog, _make_add_subfactor_dialog

st.set_page_config(page_title="Scoring Profiles", layout="wide")
inject_custom_css()
render_page_header("Scoring Profiles", "Tree navigator + node editor. Select a node to edit.")

client = get_api_client("scoring_profiles")
render_sidebar_api_test(client, "scoring_profiles_test")

try:
    profiles = client.list_scoring_profiles()
    profile_names = sorted(profiles.keys())
except ApiError as exc:
    st.error(f"Cannot load profiles: {exc}")
    profiles = {}
    profile_names = []

try:
    metrics_resp = client.list_metrics()
    metric_names = sorted({m.get("metric_name", "") for m in metrics_resp if m.get("metric_name")})
except ApiError:
    metric_names = []

if not profile_names:
    st.info("No scoring profiles found. Create one via the Scoring Profile Wizard.")
    st.stop()

st.divider()
selector_col, reload_col = st.columns([3, 1])
with selector_col:
    selected_profile = st.selectbox(
        "Profile to edit",
        options=profile_names,
        key="scoring_profile_select",
    )
with reload_col:
    if st.button("Reload from API", key="reload_profile"):
        profile_data = client.get_scoring_profile(selected_profile)
        store = ProfileStore()
        store.load_from_profile(profile_data)
        store_key = f"profile_editor_store_{selected_profile}"
        st.session_state[store_key] = store
        st.session_state.pop("profile_editor_selected_node_id", None)
        st.rerun()

if not selected_profile:
    st.stop()

profile_data = profiles.get(selected_profile, {})
if not profile_data.get("nodes"):
    st.warning("Profile has no nodes. Cannot edit.")
    st.stop()

# Initialize or reload flat store when profile changes
store_key = f"profile_editor_store_{selected_profile}"
if store_key not in st.session_state:
    store = ProfileStore()
    store.load_from_profile(profile_data)
    st.session_state[store_key] = store
else:
    store = st.session_state[store_key]

# Session state for profile-level fields
if f"profile_norm_{selected_profile}" not in st.session_state:
    st.session_state[f"profile_norm_{selected_profile}"] = profile_data.get("normalization", "zscore")
winsor = profile_data.get("winsorization")
if f"profile_winsor_{selected_profile}" not in st.session_state:
    st.session_state[f"profile_winsor_{selected_profile}"] = winsor if isinstance(winsor, dict) else {"lower": 0.01, "upper": 0.99}
if f"profile_use_winsor_{selected_profile}" not in st.session_state:
    st.session_state[f"profile_use_winsor_{selected_profile}"] = bool(winsor)
if f"profile_winsor_mode_{selected_profile}" not in st.session_state:
    st.session_state[f"profile_winsor_mode_{selected_profile}"] = profile_data.get("winsor_mode", "quantile")
if f"profile_method_{selected_profile}" not in st.session_state:
    st.session_state[f"profile_method_{selected_profile}"] = profile_data.get("method", "linear")
normalization = st.session_state[f"profile_norm_{selected_profile}"]
winsor_dict = st.session_state[f"profile_winsor_{selected_profile}"]
use_winsor = st.session_state[f"profile_use_winsor_{selected_profile}"]

# Selection
if "profile_editor_selected_node_id" not in st.session_state:
    st.session_state["profile_editor_selected_node_id"] = store.get_root_id()
selected_id = st.session_state["profile_editor_selected_node_id"]
nodes = store.get_nodes()
root_id = store.get_root_id()
if selected_id and selected_id not in nodes:
    st.session_state["profile_editor_selected_node_id"] = root_id
    selected_id = root_id

# Global validation summary
issues = validate_nodes(nodes, root_id)
if issues:
    with st.container():
        st.markdown("**Validation issues** (click to navigate):")
        for issue in issues[:10]:
            if st.button(
                f"⚠ {issue['message']} → {issue['node_id']}",
                key=f"nav_issue_{issue['node_id']}_{issue['kind']}",
            ):
                st.session_state["profile_editor_selected_node_id"] = issue["node_id"]
                st.rerun()
else:
    st.success("No validation issues.")

st.divider()

# Handle dialogs and confirmations
key_prefix = f"pe_{selected_profile}"
confirm_del = st.session_state.get(f"{key_prefix}_confirm_delete")
if confirm_del:
    st.warning(f"Delete node '{confirm_del}' and its subtree?")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Confirm delete", type="primary"):
            store.delete_node(confirm_del)
            st.session_state.pop(f"{key_prefix}_confirm_delete", None)
            if st.session_state.get("profile_editor_selected_node_id") == confirm_del:
                st.session_state["profile_editor_selected_node_id"] = root_id
            st.rerun()
    with c2:
        if st.button("Cancel"):
            st.session_state.pop(f"{key_prefix}_confirm_delete", None)
            st.rerun()
    st.stop()

dup_node = st.session_state.get(f"{key_prefix}_duplicate_node")
if dup_node:
    new_name = st.text_input("Name for duplicate", value=f"{nodes.get(dup_node, {}).get('name', 'Copy')} (copy)")
    if st.button("Create duplicate"):
        new_id = store.duplicate_subtree(dup_node, new_name or "Copy")
        st.session_state.pop(f"{key_prefix}_duplicate_node", None)
        if new_id:
            st.session_state["profile_editor_selected_node_id"] = new_id
        st.rerun()
    if st.button("Cancel"):
        st.session_state.pop(f"{key_prefix}_duplicate_node", None)
        st.rerun()
    st.stop()

add_metric_parent = st.session_state.get(f"{key_prefix}_add_metric_parent")
if add_metric_parent:
    add_metric_dialog_fn = _make_add_metric_dialog(store, nodes, metric_names, key_prefix)
    if add_metric_dialog_fn:
        add_metric_dialog_fn(add_metric_parent)
    else:
        m = st.selectbox("Select metric", options=[""] + metric_names, key="add_metric_sel")
        if st.button("Add metric"):
            if m:
                store.add_metric_child(add_metric_parent, m, 0.0)
            st.session_state.pop(f"{key_prefix}_add_metric_parent", None)
            st.rerun()
        if st.button("Cancel"):
            st.session_state.pop(f"{key_prefix}_add_metric_parent", None)
            st.rerun()
        st.stop()

add_subfactor_parent = st.session_state.get(f"{key_prefix}_add_subfactor_parent")
if add_subfactor_parent:
    add_subfactor_dialog_fn = _make_add_subfactor_dialog(store, nodes, key_prefix)
    if add_subfactor_dialog_fn:
        add_subfactor_dialog_fn(add_subfactor_parent)
    else:
        name = st.text_input("Subfactor name", value="New factor")
        if st.button("Add subfactor"):
            if name:
                new_id = store.add_subfactor(add_subfactor_parent, name.strip())
                st.session_state.pop(f"{key_prefix}_add_subfactor_parent", None)
                st.session_state["profile_editor_selected_node_id"] = new_id
            st.rerun()
        if st.button("Cancel"):
            st.session_state.pop(f"{key_prefix}_add_subfactor_parent", None)
            st.rerun()
        st.stop()

# Main layout: Tree (left) + Editor (right)
col_tree, col_editor = st.columns([0.35, 0.65])

with col_tree:
    st.subheader("Tree")
    render_tree_nav(nodes, root_id, selected_id, metric_names, key_prefix)

with col_editor:
    st.subheader("Editor")
    render_node_editor(
        store,
        selected_id,
        metric_names,
        key_prefix,
        selection_state_key="profile_editor_selected_node_id",
        add_metric_key=f"{key_prefix}_add_metric_parent",
        add_subfactor_key=f"{key_prefix}_add_subfactor_parent",
    )

st.divider()

# Normalization, Winsorization & Aggregation (global)
with st.expander("Normalization, Winsorization & Aggregation"):
    norm_options = ["zscore", "normalized_zscore", "percentile"]
    idx = norm_options.index(normalization) if normalization in norm_options else 0
    st.session_state[f"profile_norm_{selected_profile}"] = st.selectbox(
        "Normalization", options=norm_options, index=idx, key="norm_sel"
    )
    method_labels = ["Linear (weighted sum)", "Softplus (smooth combination)"]
    method_values = ["linear", "softplus"]
    method_key = f"profile_method_{selected_profile}"
    method_idx = method_values.index(st.session_state[method_key]) if st.session_state[method_key] in method_values else 0
    method_choice = st.selectbox(
        "Aggregation method",
        options=method_labels,
        index=method_idx,
        key=f"profile_agg_method_{selected_profile}",
        help="Linear = sum of weighted z-scores. Softplus = geometric-like combination. Per-node override in Node settings.",
    )
    st.session_state[method_key] = method_values[method_labels.index(method_choice)]
    use_winsor = st.checkbox("Use winsorization", value=use_winsor, key="winsor_use")
    st.session_state[f"profile_use_winsor_{selected_profile}"] = use_winsor
    winsor_mode_key = f"profile_winsor_mode_{selected_profile}"
    winsor_mode = st.session_state[winsor_mode_key]
    if use_winsor:
        mode_labels = ["Quantile (percentiles)", "Semi (mean ± k·σ)"]
        mode_values = ["quantile", "semi"]
        try:
            mode_index = mode_values.index(winsor_mode)
        except ValueError:
            mode_index = 0
        mode_label = st.selectbox(
            "Winsorization method",
            options=mode_labels,
            index=mode_index,
            key="winsor_method_sel",
        )
        winsor_mode = mode_values[mode_labels.index(mode_label)]
        st.session_state[winsor_mode_key] = winsor_mode

        if winsor_mode == "quantile":
            st.session_state[f"profile_winsor_{selected_profile}"] = {
                "lower": st.number_input("Winsor lower", value=winsor_dict.get("lower", 0.01), min_value=0.0, max_value=0.49, step=0.01, key="winsor_lo"),
                "upper": st.number_input("Winsor upper", value=winsor_dict.get("upper", 0.99), min_value=0.51, max_value=1.0, step=0.01, key="winsor_hi"),
            }
        else:
            st.session_state[f"profile_winsor_{selected_profile}"] = {
                "k": st.number_input("k (std dev multiplier)", value=float(winsor_dict.get("k", 3.0)), min_value=0.1, max_value=10.0, step=0.1, key="winsor_k"),
            }
    else:
        # Keep last winsor settings but mark as unused
        st.session_state[f"profile_winsor_{selected_profile}"] = winsor_dict

# Previews
with st.expander("Preview"):
    w = st.session_state[f"profile_winsor_{selected_profile}"] if st.session_state[f"profile_use_winsor_{selected_profile}"] else False
    render_previews(
        nodes,
        root_id,
        st.session_state[f"profile_norm_{selected_profile}"],
        w,
        st.session_state[f"profile_winsor_mode_{selected_profile}"],
        key_prefix,
        method=st.session_state[f"profile_method_{selected_profile}"],
    )

# Actions
st.divider()
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    if st.button("Save changes", type="primary", key="save_btn"):
        try:
            w = st.session_state[f"profile_winsor_{selected_profile}"] if st.session_state[f"profile_use_winsor_{selected_profile}"] else False
            payload = flat_to_export_payload(
                store.get_nodes(),
                store.get_root_id(),
                st.session_state[f"profile_norm_{selected_profile}"],
                w,
                st.session_state[f"profile_winsor_mode_{selected_profile}"],
                method=st.session_state[f"profile_method_{selected_profile}"],
            )
            client.upsert_scoring_profile(selected_profile, payload)
            st.success(f"Saved '{selected_profile}'.")
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))
with c2:
    if st.button("Delete profile", key="del_profile_btn"):
        try:
            client.delete_scoring_profile(selected_profile)
            st.success(f"Deleted '{selected_profile}'.")
            st.session_state.pop(store_key, None)
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))
    st.caption("Deleting cannot be undone.")
