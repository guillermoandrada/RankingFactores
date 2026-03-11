from __future__ import annotations

import json

import streamlit as st

from streamlit_app.api_client import ApiError
from streamlit_app.ui import get_api_client, inject_custom_css, render_page_header, render_sidebar_api_test
from streamlit_app.wizard_components import convert_wizard_to_profile, render_transforms, validate_profile_payload
from streamlit_app.wizard_tree_editor import render_step2_base_structure

st.set_page_config(page_title="Scoring Profile Wizard", layout="wide")
inject_custom_css()
render_page_header("Scoring Profile Wizard", "Step-by-step builder for scoring methodologies with nested composition boxes.")

client = get_api_client("wizard")
render_sidebar_api_test(client, "wizard_test_api")

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

    transforms = st.session_state.get("wizard_transforms", [])
    profile = convert_wizard_to_profile(base_profile, transforms=transforms)

    if warnings:
        for msg in warnings:
            st.warning(msg)

    st.subheader("Generated profile (nodes format)")
    st.caption("Normalization and winsorization come from the Transform Chain step.")
    st.code(json.dumps(profile, indent=2), language="json")

    save_disabled = len(warnings) > 0 or not profile_name
    if st.button("Save scoring profile", type="primary", disabled=save_disabled):
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
