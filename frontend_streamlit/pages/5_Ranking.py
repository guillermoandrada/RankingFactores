"""Ranking - Execute ranking endpoint for a period and scoring profile."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from frontend_streamlit.api_client import ApiError, RankingApiClient


st.set_page_config(page_title="Ranking", layout="wide")
st.title("Ranking")
st.caption("Compute rankings for a period and scoring profile. Choose sector or industry to see rankings in collapsible sections.")


def _client() -> RankingApiClient:
    api_url = st.sidebar.text_input(
        "API Base URL",
        value="http://127.0.0.1:8000",
        key="ranking_api_url",
    )
    return RankingApiClient(api_url)


client = _client()

if st.sidebar.button("Test API connection", key="ranking_test_api"):
    try:
        periods = client.list_periods()
        st.sidebar.success(f"Connected. Periods: {len(periods)}")
    except ApiError as exc:
        st.sidebar.error(str(exc))

try:
    periods = client.list_periods()
    profiles = client.list_scoring_profiles()
    profile_names = sorted(profiles.keys())
    sectors = client.list_sectors()
    industries = client.list_industries()
except ApiError as exc:
    st.error(f"Cannot load data: {exc}")
    periods = []
    profile_names = []
    sectors = []
    industries = []

if not periods:
    st.info("No periods found. Upload data via the Periods page.")
elif not profile_names:
    st.info("No scoring profiles found. Create one via the Scoring Profile Wizard.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        period = st.selectbox("Period", options=periods, key="ranking_period")
    with col2:
        scoring_profile = st.selectbox("Scoring profile", options=profile_names, key="ranking_profile")
    with col3:
        scope = st.radio(
            "Scope",
            options=["All", "Sector", "Industry"],
            horizontal=True,
            key="ranking_scope",
        )

    st.divider()

    if st.button("Run ranking", type="primary", key="ranking_run"):
        results = {}
        if scope == "All":
            scopes = [("", "")]
            scope_keys = ["All"]
        elif scope == "Sector":
            scopes = [(s, "") for s in sorted(sectors)]
            scope_keys = list(s for s, _ in scopes)
        else:
            scopes = [("", i) for i in sorted(industries)]
            scope_keys = list(i for _, i in scopes)

        if len(scopes) == 1:
            progress = st.progress(0.0, "Computing ranking...")
            try:
                result = client.run_ranking(
                    period=period,
                    scoring_profile=scoring_profile,
                    industry=scopes[0][1],
                    sector=scopes[0][0],
                )
                results[scope_keys[0]] = result
            except ApiError as e:
                results[scope_keys[0]] = {"error": str(e), "ranking": []}
            progress.empty()
        else:
            progress = st.progress(0.0, "Computing rankings in parallel...")
            try:
                batch = client.run_ranking_batch(period, scoring_profile, scopes)
                results = batch.get("results", {})
            except ApiError as e:
                for k in scope_keys:
                    results[k] = {"error": str(e), "ranking": []}
            progress.empty()
        st.session_state["ranking_results"] = results
        st.session_state["ranking_results_scope"] = scope
        st.rerun()

    results = st.session_state.get("ranking_results", {})
    results_scope = st.session_state.get("ranking_results_scope", "")
    if results and results_scope == scope:
        st.subheader("Results")
        st.caption(f"Period: {period} | Profile: {scoring_profile} | Scope: {scope}")
        for scope_key, result in results.items():
            if result.get("error"):
                with st.expander(f"**{scope_key}** — Error", expanded=False):
                    st.error(result["error"])
            else:
                df = pd.DataFrame(result.get("ranking", []))
                count = result.get("count", len(df))
                with st.expander(f"**{scope_key}** — {count} companies", expanded=(len(results) == 1)):
                    st.dataframe(df, use_container_width=True)

        # Export to Excel: append all rankings vertically in one worksheet
        parts: list[pd.DataFrame] = []
        for scope_key, result in results.items():
            if result.get("error"):
                continue
            ranking = result.get("ranking", [])
            if not ranking:
                continue
            df_part = pd.DataFrame(ranking)
            if len(results) > 1:
                df_part.insert(0, scope, scope_key)
            parts.append(df_part)
        if parts:
            export_df = pd.concat(parts, ignore_index=True)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                export_df.to_excel(writer, index=False, sheet_name="Ranking")
            buffer.seek(0)
            st.download_button(
                "Export to Excel",
                data=buffer.getvalue(),
                file_name=f"Ranking_{period.replace(' ', '').replace('/', '-')}_{scoring_profile.replace(' ', '_')}_{scope}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="ranking_export_xlsx",
            )
