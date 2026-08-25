"""Workspace selections shared across pages: the period and scoring profile in focus.

Streamlit deletes the state of any widget that was not rendered on the last script run,
which happens every time the user switches page. A widget key therefore cannot carry a
choice between pages on its own. Each selection is held in a plain state key instead,
with the widget key kept in sync in both directions around it.
"""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

PERIOD_STATE_KEY = "workspace_period"
SCORING_PROFILE_STATE_KEY = "workspace_scoring_profile"

_PERIOD_WIDGET_KEY = "workspace_period_widget"
_SCORING_PROFILE_WIDGET_KEY = "workspace_scoring_profile_widget"

_SHARED_HELP = "Shared with the other pages: changing it here changes it everywhere."


def select_period(
    options: Iterable[str],
    *,
    label: str = "Period",
    help_text: str = _SHARED_HELP,
) -> str:
    """Render the workspace period selector and return the chosen period."""
    return _shared_selectbox(
        PERIOD_STATE_KEY, _PERIOD_WIDGET_KEY, label, list(options), help_text
    )


def select_scoring_profile(
    options: Iterable[str],
    *,
    label: str = "Scoring profile",
    help_text: str = _SHARED_HELP,
) -> str:
    """Render the workspace scoring profile selector and return the chosen profile."""
    return _shared_selectbox(
        SCORING_PROFILE_STATE_KEY, _SCORING_PROFILE_WIDGET_KEY, label, list(options), help_text
    )


def current_period(options: Iterable[str]) -> str:
    """Return the workspace period when it is still valid, else the first option."""
    return _valid_choice(PERIOD_STATE_KEY, list(options))


def current_scoring_profile(options: Iterable[str]) -> str:
    """Return the workspace scoring profile when it is still valid, else the first option."""
    return _valid_choice(SCORING_PROFILE_STATE_KEY, list(options))


def _shared_selectbox(
    state_key: str,
    widget_key: str,
    label: str,
    options: list[str],
    help_text: str,
) -> str:
    """
    Selectbox backed by a workspace-wide key, so the choice follows the user between pages.

    The stored value is reconciled with the options before the widget is created: Streamlit
    raises when a keyed selectbox holds a value the current options no longer contain.
    """
    if not options:
        return ""
    st.session_state[widget_key] = _valid_choice(state_key, options)
    chosen = st.selectbox(
        label,
        options=options,
        key=widget_key,
        help=help_text,
        on_change=_persist_choice,
        args=(widget_key, state_key),
    )
    st.session_state[state_key] = chosen
    return chosen


def _persist_choice(widget_key: str, state_key: str) -> None:
    """Copy the widget's new value into the state key that survives a page switch."""
    st.session_state[state_key] = st.session_state[widget_key]


def _valid_choice(state_key: str, options: list[str]) -> str:
    if not options:
        return ""
    stored = st.session_state.get(state_key)
    return stored if stored in options else options[0]
