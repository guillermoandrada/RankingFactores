"""TTL-cached reference lists (periods, profiles, sectors, etc.) for heavy Streamlit pages."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from streamlit_app.client.api_client import RankingApiClient

REFERENCE_CACHE_TTL_SEC = 120.0


def _bundle_key(cache_key: str) -> str:
    return f"{cache_key}_reference_bundle_v1"


def invalidate_reference_bundle(cache_key: str) -> None:
    st.session_state.pop(_bundle_key(cache_key), None)


def render_reference_refresh_button(cache_key: str) -> None:
    """Sidebar control to drop cached reference data and refetch on next run."""
    if st.sidebar.button(
        "Refresh reference lists",
        key=f"{cache_key}_sidebar_refresh_reference",
        help="Reload periods, sectors, industries, indices, and profiles from the API.",
    ):
        invalidate_reference_bundle(cache_key)
        st.rerun()


def load_reference_data_bundle(
    client: RankingApiClient,
    *,
    cache_key: str,
) -> tuple[list[str], dict[str, Any], list[str], list[str], list[str]]:
    """
    Return (periods, profiles_dict, sectors, industries, indices) using a short TTL cache.
    """
    key = _bundle_key(cache_key)
    now = time.monotonic()
    cached = st.session_state.get(key)
    if isinstance(cached, dict) and (now - cached["monotonic_t"]) < REFERENCE_CACHE_TTL_SEC:
        return (
            cached["periods"],
            cached["profiles"],
            cached["sectors"],
            cached["industries"],
            cached["indices"],
        )

    periods = client.list_periods()
    profiles = client.list_scoring_profiles()
    sectors = client.list_sectors()
    industries = client.list_industries()
    indices = client.list_indices()

    st.session_state[key] = {
        "monotonic_t": now,
        "periods": periods,
        "profiles": profiles,
        "sectors": sectors,
        "industries": industries,
        "indices": indices,
    }
    return periods, profiles, sectors, industries, indices
