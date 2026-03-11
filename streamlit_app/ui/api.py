"""Shared API client factory for Streamlit pages."""

from __future__ import annotations

import streamlit as st

from streamlit_app.api_client import ApiError, RankingApiClient


def get_api_client(key_prefix: str) -> RankingApiClient:
    """
    Render API Base URL input in sidebar and return RankingApiClient.
    Call render_sidebar_api_test after this to add the Test button.
    """
    api_url = st.sidebar.text_input(
        "API Base URL",
        value="http://127.0.0.1:8000",
        key=f"{key_prefix}_api_url",
    )
    return RankingApiClient(api_url.rstrip("/") if api_url else "http://127.0.0.1:8000")


def render_sidebar_api_test(client: RankingApiClient, key: str) -> None:
    """
    Render Test API connection button in sidebar.
    Uses list_periods as a generic connectivity check.
    """
    if st.sidebar.button("Test API connection", key=key):
        try:
            periods = client.list_periods()
            st.sidebar.success(f"Connected. Periods: {len(periods)}")
        except ApiError as exc:
            st.sidebar.error(str(exc))
