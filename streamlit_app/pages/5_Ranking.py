"""Ranking - Execute ranking endpoint for a period and scoring profile."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from typing import Any

from streamlit_app.api_client import ApiError
from streamlit_app.ui import get_api_client, inject_custom_css, render_page_header, render_sidebar_api_test

st.set_page_config(page_title="Ranking", layout="wide")
inject_custom_css()
render_page_header("Ranking", "Compute rankings for a period and scoring profile. Choose sector or industry to see rankings in collapsible sections.")

client = get_api_client("ranking")
render_sidebar_api_test(client, "ranking_test_api")

try:
    periods = client.list_periods()
    profiles = client.list_scoring_profiles()
    profile_names = sorted(profiles.keys())
    sectors = client.list_sectors()
    industries = client.list_industries()
    indices = client.list_indices()
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
    st.divider()
    with st.container(border=True):
        st.markdown("**Filters**")
        st.caption("Choose a period, scoring profile, and scope, then run the ranking.")

        # Top-level selectors: period, default scoring profile, and optional index
        row1_col1, row1_col2, row1_col3 = st.columns([2, 2, 2])
        with row1_col1:
            period = st.selectbox("Period", options=periods, key="ranking_period")
        with row1_col2:
            scoring_profile = st.selectbox(
                "Scoring profile", options=profile_names, key="ranking_profile"
            )
        with row1_col3:
            index_options = ["(All indices)"] + sorted(indices)
            index_label = st.selectbox(
                "Index",
                options=index_options,
                key="ranking_index",
            )
            index_choice = "" if index_label == "(All indices)" else index_label
        # Scope selector below the main lists
        scope = st.radio(
            "Scope",
            options=["All", "Sector", "Industry"],
            horizontal=True,
            key="ranking_scope",
        )
        st.caption("All = full universe. Sector/Industry = define specific overrides below.")

        # Track scope changes so we can reset custom rows if desired
        prev_scope = st.session_state.get("ranking_prev_scope")
        if prev_scope != scope:
            st.session_state["ranking_prev_scope"] = scope
            # Reset custom rows when switching scope type
            st.session_state.pop("ranking_scope_rows", None)

        # Button to add per-scope overrides (sector/industry + profile)
        rows = st.session_state.get("ranking_scope_rows", [])
        if scope in ("Sector", "Industry"):
            if st.button("Add override", key="ranking_add_scope_row"):
                rows = st.session_state.get("ranking_scope_rows", [])
                next_id = max(rows) + 1 if rows else 0
                rows.append(next_id)
                st.session_state["ranking_scope_rows"] = rows
                st.rerun()

            rows = st.session_state.get("ranking_scope_rows", [])
            if rows:
                entities = sorted(sectors) if scope == "Sector" else sorted(industries)
                with st.container(border=True):
                    title = "Overrides by sector" if scope == "Sector" else "Overrides by industry"
                    st.markdown(f"**{title}**")
                    st.caption(
                        "Each override defines one ranking run for a specific sector/industry and scoring profile. "
                        "All other sectors/industries use the default profile above."
                    )

                    # Clear all overrides for the current scope
                    if st.button("Clear all overrides", key="ranking_clear_scope_rows"):
                        for row_id in rows:
                            ent_key = f"ranking_scope_entity_{row_id}"
                            prof_key = f"ranking_scope_profile_{row_id}"
                            st.session_state.pop(ent_key, None)
                            st.session_state.pop(prof_key, None)
                        st.session_state["ranking_scope_rows"] = []
                        st.rerun()

                    for row_id in rows:
                        row_col1, row_col2, row_col3 = st.columns([2, 2, 1])
                        ent_key = f"ranking_scope_entity_{row_id}"
                        prof_key = f"ranking_scope_profile_{row_id}"
                        with row_col1:
                            if ent_key not in st.session_state:
                                st.session_state[ent_key] = ""
                            st.selectbox(
                                "Sector" if scope == "Sector" else "Industry",
                                options=[""] + entities,
                                key=ent_key,
                            )
                        with row_col2:
                            if prof_key not in st.session_state:
                                st.session_state[prof_key] = scoring_profile
                            st.selectbox(
                                "Profile",
                                options=profile_names,
                                key=prof_key,
                            )
                        with row_col3:
                            # Spacer to align the button with the selectbox labels
                            st.write(" ")
                            if st.button("Remove override", key=f"ranking_remove_scope_row_{row_id}"):
                                # Remove this row and its associated state
                                rows = [r for r in rows if r != row_id]
                                st.session_state["ranking_scope_rows"] = rows
                                st.session_state.pop(ent_key, None)
                                st.session_state.pop(prof_key, None)
                                st.rerun()

        # Run button for all scopes
        run_clicked = st.button("Run ranking", type="primary", key="ranking_run")

        if run_clicked:
            results: dict[str, dict] = {}

            if scope == "All":
                # Single full-universe run
                progress = st.progress(0.0, "Computing ranking...")
                try:
                    result = client.run_ranking(
                        period=period,
                        scoring_profile=scoring_profile,
                        industry="",
                        sector="",
                        index=index_choice,
                    )
                    results["All"] = result
                except ApiError as e:
                    results["All"] = {"error": str(e), "ranking": []}
                progress.empty()
            else:
                rows = st.session_state.get("ranking_scope_rows", [])
                entities = sorted(sectors) if scope == "Sector" else sorted(industries)
                if not entities:
                    st.warning(f"No {'sectors' if scope == 'Sector' else 'industries'} available for this scope.")
                else:
                    # Build map of overrides: entity -> override_profile
                    override_map: dict[str, str] = {}
                    for row_id in rows:
                        ent_key = f"ranking_scope_entity_{row_id}"
                        prof_key = f"ranking_scope_profile_{row_id}"
                        entity = st.session_state.get(ent_key, "")
                        if not entity or entity not in entities:
                            continue
                        override_profile = st.session_state.get(prof_key, scoring_profile)
                        if entity in override_map:
                            st.warning(
                                "Duplicate overrides detected. Each override must target a unique sector or industry."
                            )
                            override_map = {}
                            break
                        override_map[entity] = override_profile

                    if not entities or (rows and override_map == {} and len(rows) > 0):
                        # Either no valid entities, or duplicate overrides cleared the map
                        st.warning(
                            "No valid overrides configured. Please ensure each override selects a unique sector or industry."
                        )
                    else:
                        # Build scopes for all entities: non-overridden ones use default profile
                        scoped_profiles: list[dict[str, Any]] = []
                        for entity in entities:
                            profile_for_entity = override_map.get(entity, scoring_profile)
                            if scope == "Sector":
                                sector_val, industry_val = entity, ""
                            else:
                                sector_val, industry_val = "", entity
                            item: dict[str, Any] = {
                                "sector": sector_val,
                                "industry": industry_val,
                            }
                            # Only send scoring_profile when overriding the default
                            if profile_for_entity != scoring_profile:
                                item["scoring_profile"] = profile_for_entity
                            scoped_profiles.append(item)

                        progress = st.progress(0.0, "Computing rankings in parallel...")
                        try:
                            batch = client.run_ranking_batch_with_profiles(
                                period,
                                scoring_profile,
                                scoped_profiles,
                                index=index_choice,
                            )
                            results = batch.get("results", {})
                        except ApiError as e:
                            results = {}
                            for item in scoped_profiles:
                                key_name = item.get("sector") or item.get("industry") or "All"
                                results[key_name] = {"error": str(e), "ranking": []}
                        progress.empty()

            if results:
                st.session_state["ranking_results"] = results
                st.session_state["ranking_results_scope"] = scope
                st.rerun()

    # Hint when no results have been computed yet
    results = st.session_state.get("ranking_results", {})
    if not results:
        st.info("Run a ranking to see results here.")

    results_scope = st.session_state.get("ranking_results_scope", "")
    if results and results_scope == scope:
        st.subheader("Results")
        total_scopes = len(results)
        error_count = sum(1 for r in results.values() if r.get("error"))
        ok_count = total_scopes - error_count
        index_caption = ""
        if "ranking_index" in st.session_state:
            idx_val = st.session_state.get("ranking_index") or "(All indices)"
            index_caption = f" | Index: {idx_val}"
        st.caption(f"Period: {period} | Profile: {scoring_profile} | Scope: {scope}{index_caption}")
        st.caption(f"Scopes: {total_scopes} · Successful: {ok_count} · Errors: {error_count}")
        st.divider()
        for scope_key in sorted(results):
            result = results[scope_key]
            if result.get("error"):
                with st.expander(f"**{scope_key}** — Error", expanded=False):
                    st.error(result["error"])
            else:
                df = pd.DataFrame(result.get("ranking", []))
                count = result.get("count", len(df))
                profile_for_scope = result.get("scoring_profile", scoring_profile)
                title = f"**{scope_key}** — {count} companies · {profile_for_scope}"
                with st.expander(title, expanded=(len(results) == 1)):
                    st.dataframe(df, use_container_width=True)

        # Export to Excel: append all rankings vertically in one worksheet
        parts: list[pd.DataFrame] = []
        for scope_key in sorted(results):
            result = results[scope_key]
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
            with st.container(border=True):
                st.markdown("**Export**")
                st.caption("Combine all successful scopes into a single Excel worksheet.")
                st.download_button(
                    "Export to Excel",
                    data=buffer.getvalue(),
                    file_name=f"Ranking_{period.replace(' ', '').replace('/', '-')}_{scoring_profile.replace(' ', '_')}_{scope}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="ranking_export_xlsx",
                )
