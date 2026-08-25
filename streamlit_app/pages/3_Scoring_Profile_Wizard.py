from __future__ import annotations

import json

import streamlit as st

from streamlit_app.client.api_client import ApiError
from streamlit_app.ui import get_api_client, render_page_header, render_sidebar_api_status
from streamlit_app.components.wizard.steps import convert_wizard_to_profile, render_transforms, validate_profile_payload
from streamlit_app.components.wizard.tree_editor import render_step2_base_structure

render_page_header("Scoring Profile Wizard", "Step-by-step builder for scoring methodologies with nested composition boxes.")

client = get_api_client()
render_sidebar_api_status(client)

if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 1
if "wizard_profile_name" not in st.session_state:
    st.session_state.wizard_profile_name = ""

try:
    metric_names = [m["metric_name"] for m in client.list_metrics()]
except ApiError as exc:
    st.error(f"Cannot load metrics from API: {exc}")
    metric_names = []

# Step content runs first so values are captured before navigation triggers rerun
if st.session_state.wizard_step == 1:
    st.header("Step 1 - Transform Chain")
    st.session_state.wizard_transforms = render_transforms(
        key_prefix="wizard_base_transforms",
        title="Base Transform Chain",
    )
    st.divider()
    agg_labels = ["Linear (weighted sum)", "Softplus (smooth combination)"]
    agg_values = ["linear", "softplus"]
    if "wizard_base_method" not in st.session_state:
        st.session_state.wizard_base_method = "linear"
    agg_idx = agg_values.index(st.session_state.wizard_base_method) if st.session_state.wizard_base_method in agg_values else 0
    choice = st.selectbox(
        "Aggregation method",
        options=agg_labels,
        index=agg_idx,
        key="wizard_agg_method",
        help="Linear = sum of weighted z-scores. Softplus = geometric-like combination using log(1+exp(z)).",
    )
    st.session_state.wizard_base_method = agg_values[agg_labels.index(choice)]

elif st.session_state.wizard_step == 2:
    st.header("Step 2 - Base Structure")
    st.session_state.wizard_base_profile = render_step2_base_structure(
        metrics=metric_names,
        key_prefix="wizard_base",
    )

elif st.session_state.wizard_step == 3:
    st.header("Step 3 - Review & Save")
    st.text_input(
        "Scoring profile name",
        key="wizard_profile_name",
        placeholder="Enter a name for this scoring profile",
    )
    profile_name = str(st.session_state.get("wizard_profile_name", "")).strip()
    base_profile = st.session_state.get("wizard_base_profile", {})

    legacy_payload = {
        "base": base_profile,
    }

    warnings = []
    if not profile_name:
        warnings.append("Profile name is empty.")
    warnings.extend(validate_profile_payload(legacy_payload))

    try:
        existing_profile_names = set(client.list_scoring_profiles().keys())
    except ApiError as exc:
        existing_profile_names = set()
        st.warning(f"Could not check existing profile names: {exc}")

    name_taken = bool(profile_name) and profile_name in existing_profile_names
    overwrite_confirmed = False
    if name_taken:
        st.warning(
            f"A scoring profile named '{profile_name}' already exists. "
            "Saving replaces it and its current structure cannot be recovered."
        )
        overwrite_confirmed = st.checkbox(
            f"Overwrite '{profile_name}'",
            key="wizard_confirm_overwrite",
        )

    transforms = st.session_state.get("wizard_transforms", [])
    method = st.session_state.get("wizard_base_method", "linear")
    profile = convert_wizard_to_profile(base_profile, transforms=transforms, method=method)

    if warnings:
        for msg in warnings:
            st.warning(msg)

    st.subheader("Generated profile (nodes format)")
    st.caption("Normalization, winsorization, and aggregation method come from Step 1.")
    st.code(json.dumps(profile, indent=2), language="json")

    save_disabled = bool(warnings) or not profile_name or (name_taken and not overwrite_confirmed)
    save_label = "Overwrite scoring profile" if name_taken else "Save scoring profile"
    if st.button(save_label, type="primary", disabled=save_disabled):
        try:
            client.upsert_scoring_profile(profile_name, profile)
            st.success(f"Saved scoring profile '{profile_name}'.")
            # Reset wizard state back to step 1 for a fresh profile
            st.session_state.wizard_step = 1
            # Clear per-run wizard memory (keep only the current step)
            keys_to_clear = [
                k for k in list(st.session_state.keys())
                if k.startswith("wizard_") and k not in ("wizard_step",)
            ]
            for k in keys_to_clear:
                st.session_state.pop(k, None)
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))

# Navigation runs after step content so values are captured before rerun
st.divider()
nav_col1, nav_col2 = st.columns([1, 1])
with nav_col1:
    if st.button("← Back", disabled=st.session_state.wizard_step <= 1, key="wizard_back"):
        st.session_state.wizard_step -= 1
        st.rerun()
with nav_col2:
    if st.button("Next →", type="primary", disabled=st.session_state.wizard_step >= 3, key="wizard_next"):
        st.session_state.wizard_step += 1
        st.rerun()
