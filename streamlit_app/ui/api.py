"""Shared API client factory and connection status for Streamlit pages."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from streamlit_app.client.api_client import ApiError, RankingApiClient

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
API_BASE_URL_STATE_KEY = "api_base_url"

_STATUS_STATE_KEY = "api_connection_status"
_STATUS_TTL_SEC = 30.0
_PROBE_TIMEOUT_SEC = 5.0


def get_api_client() -> RankingApiClient:
    """
    Render the shared API Base URL input in the sidebar and return a client for it.

    A single workspace-wide key backs the input, so the URL entered on any page is
    the URL every other page uses.
    """
    base_url = st.sidebar.text_input(
        "API Base URL",
        value=DEFAULT_API_BASE_URL,
        key=API_BASE_URL_STATE_KEY,
        help="Shared by every page.",
    )
    return RankingApiClient((base_url or DEFAULT_API_BASE_URL).rstrip("/"))


def render_sidebar_api_status(client: RankingApiClient) -> None:
    """
    Show whether the API answers, probing at most once per TTL window.

    Replaces the per-page manual test button: the check runs on its own and the user
    only clicks when they want to force a fresh probe.
    """
    status = _cached_status(client.base_url)
    if status["ok"]:
        st.sidebar.success(status["summary"])
    else:
        st.sidebar.error(status["summary"])
        st.sidebar.caption(status["detail"])
    if st.sidebar.button("Recheck connection", key="api_status_recheck"):
        invalidate_api_status()
        st.rerun()


def invalidate_api_status() -> None:
    """Drop the cached probe so the next render checks the API again."""
    st.session_state.pop(_STATUS_STATE_KEY, None)


def _cached_status(base_url: str) -> dict[str, Any]:
    cached = st.session_state.get(_STATUS_STATE_KEY)
    now = time.monotonic()
    if (
        isinstance(cached, dict)
        and cached.get("base_url") == base_url
        and (now - cached.get("monotonic_t", 0.0)) < _STATUS_TTL_SEC
    ):
        return cached

    status = {"base_url": base_url, "monotonic_t": now, **_probe(base_url)}
    st.session_state[_STATUS_STATE_KEY] = status
    return status


def _probe(base_url: str) -> dict[str, Any]:
    """Probe with a short timeout so an unreachable host cannot stall the page."""
    try:
        periods = RankingApiClient(base_url, timeout_seconds=_PROBE_TIMEOUT_SEC).list_periods()
    except ApiError as exc:
        return {
            "ok": False,
            "summary": "API unreachable",
            "detail": str(exc),
        }
    return {
        "ok": True,
        "summary": f"Connected · {len(periods)} period(s)",
        "detail": base_url,
    }
