"""Persisted page results: keep computed output across reruns and flag stale inputs.

Streamlit reruns the whole script on every widget interaction, so output rendered
inside an `if st.button(...)` branch disappears as soon as the user touches anything.
Pages store their result here instead, together with the inputs that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st


@dataclass(frozen=True)
class PageResult:
    """A computed payload together with the inputs it was computed from."""

    payload: Any
    inputs: dict[str, Any]


def store_result(state_key: str, payload: Any, *, inputs: dict[str, Any]) -> None:
    """Persist a computed payload and the inputs that produced it."""
    st.session_state[state_key] = PageResult(payload=payload, inputs=dict(inputs))


def read_result(state_key: str) -> PageResult | None:
    """Return the stored result, or None when the page has not computed one yet."""
    stored = st.session_state.get(state_key)
    return stored if isinstance(stored, PageResult) else None


def clear_result(state_key: str) -> None:
    """Drop the stored result."""
    st.session_state.pop(state_key, None)


def is_stale(result: PageResult, current_inputs: dict[str, Any]) -> bool:
    """True when the inputs on screen no longer match the ones behind the result."""
    return result.inputs != dict(current_inputs)


def format_inputs(inputs: dict[str, Any]) -> str:
    """Render inputs as a one-line caption, e.g. `Metric: ROE · Breakdown: Total`."""
    return " · ".join(f"{label}: {_format_value(value)}" for label, value in inputs.items())


def render_result_caption(result: PageResult, current_inputs: dict[str, Any]) -> None:
    """Caption the result with its inputs, warning when they no longer match the form."""
    st.caption(format_inputs(result.inputs))
    if is_stale(result, current_inputs):
        st.warning("Inputs changed since this result was computed. Run it again to refresh.")


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value) or "—"
    return str(value)
