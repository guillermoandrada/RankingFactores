"""Shared UI components and layout helpers."""

from __future__ import annotations

from streamlit_app.ui.api import (
    get_api_client,
    invalidate_api_status,
    render_sidebar_api_status,
)
from streamlit_app.ui.layout import inject_custom_css, render_page_header, render_section
from streamlit_app.ui.results import (
    PageResult,
    clear_result,
    format_inputs,
    is_stale,
    read_result,
    render_result_caption,
    store_result,
)
from streamlit_app.ui.selection import (
    current_period,
    current_scoring_profile,
    select_period,
    select_scoring_profile,
)

__all__ = [
    "get_api_client",
    "invalidate_api_status",
    "render_sidebar_api_status",
    "inject_custom_css",
    "render_page_header",
    "render_section",
    "PageResult",
    "clear_result",
    "format_inputs",
    "is_stale",
    "read_result",
    "render_result_caption",
    "store_result",
    "current_period",
    "current_scoring_profile",
    "select_period",
    "select_scoring_profile",
]
