from __future__ import annotations

import json

import streamlit as st

from frontend_streamlit.api_client import ApiError, RankingApiClient
from frontend_streamlit.wizard_components import (
    build_multistage_config,
    build_overrides,
    convert_wizard_to_profile,
    render_transforms,
    validate_profile_payload,
)


st.set_page_config(page_title="Scoring Profile Wizard", layout="wide")
st.title("Scoring Profile Wizard")
st.caption("Step-by-step builder for scoring methodologies with nested composition boxes.")


def _client() -> RankingApiClient:
    api_url = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000", key="wizard_api_url")
    return RankingApiClient(api_url)


client = _client()

if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 1
if "wizard_profile_name" not in st.session_state:
    st.session_state.wizard_profile_name = ""

if st.sidebar.button("Test API connection", key="wizard_test_api"):
    try:
        count = len(client.list_metrics())
        st.sidebar.success(f"Connected. Metrics available: {count}")
    except ApiError as exc:
        st.sidebar.error(str(exc))

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
    st.session_state.wizard_base_profile = build_multistage_config(
        metrics=metric_names,
        key_prefix="wizard_base_multi",
        title="Composition",
        max_depth=5,
    )

elif st.session_state.wizard_step == 3:
    st.header("Step 3 - Overrides")
    try:
        sector_options = client.list_sectors()
        industry_options = client.list_industries()
    except ApiError:
        sector_options = []
        industry_options = []
    col_sector, col_industry = st.columns(2)
    with col_sector:
        st.session_state.wizard_sector_overrides = build_overrides(
            scope_name="sector",
            metrics=metric_names,
            key_prefix="wizard_overrides",
            max_depth=3,
            options=sector_options,
        )
    with col_industry:
        st.session_state.wizard_industry_overrides = build_overrides(
            scope_name="industry",
            metrics=metric_names,
            key_prefix="wizard_overrides",
            max_depth=3,
            options=industry_options,
        )

elif st.session_state.wizard_step == 4:
    st.header("Step 4 - Review & Save")
    st.text_input(
        "Scoring profile name",
        key="wizard_profile_name",
        placeholder="Enter a name for this scoring profile",
    )
    profile_name = str(st.session_state.get("wizard_profile_name", "")).strip()
    base_profile = st.session_state.get("wizard_base_profile", {})
    sector_overrides = st.session_state.get("wizard_sector_overrides", {})
    industry_overrides = st.session_state.get("wizard_industry_overrides", {})

    legacy_payload = {
        "base": base_profile,
        "overrides": {"sector": sector_overrides, "industry": industry_overrides},
    }

    warnings = []
    if not profile_name:
        warnings.append("Profile name is empty.")
    warnings.extend(validate_profile_payload(legacy_payload))

    transforms = st.session_state.get("wizard_transforms", [])
    profile = convert_wizard_to_profile(
        base_profile,
        {"sector": sector_overrides, "industry": industry_overrides},
        transforms=transforms,
    )

    if warnings:
        for msg in warnings:
            st.warning(msg)

    st.subheader("Generated profile (nodes format)")
    st.caption("Normalization and winsorization come from the Transform Chain step. Override patches via Edit.")
    st.code(json.dumps(profile, indent=2), language="json")

    save_disabled = len(warnings) > 0 or not profile_name
    if st.button("Save scoring profile", type="primary", disabled=save_disabled):
        try:
            client.upsert_scoring_profile(profile_name, profile)
            st.success(f"Saved scoring profile '{profile_name}'.")
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
    if st.button("Next →", disabled=st.session_state.wizard_step >= 4, key="wizard_next"):
        st.session_state.wizard_step += 1
        st.rerun()
