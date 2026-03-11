"""Shared layout helpers for Streamlit pages."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def inject_custom_css() -> None:
    """Inject custom CSS for consistent spacing and polish."""
    css_path = Path(__file__).resolve().parent.parent.parent / ".streamlit" / "custom.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str) -> None:
    """Render standardized page title and subtitle."""
    st.title(title)
    st.caption(subtitle)


def render_section(title: str, caption: str | None = None) -> None:
    """Render a section subheader with optional caption."""
    st.subheader(title)
    if caption:
        st.caption(caption)
