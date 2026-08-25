"""Workspace period/profile selection shared across pages."""

from __future__ import annotations

import pytest
import streamlit as st

from streamlit_app.ui.selection import (
    PERIOD_STATE_KEY,
    SCORING_PROFILE_STATE_KEY,
    current_period,
    current_scoring_profile,
)


@pytest.fixture(autouse=True)
def _clean_session_state():
    st.session_state.clear()
    yield
    st.session_state.clear()


def test_period_falls_back_to_first_option_when_nothing_is_stored() -> None:
    assert current_period(["2024Q1", "2024Q2"]) == "2024Q1"


def test_period_is_kept_when_it_is_still_available() -> None:
    st.session_state[PERIOD_STATE_KEY] = "2024Q2"

    assert current_period(["2024Q1", "2024Q2"]) == "2024Q2"


def test_period_falls_back_when_the_stored_one_disappeared() -> None:
    """A deleted period must not leave a keyed selectbox holding an invalid value."""
    st.session_state[PERIOD_STATE_KEY] = "2023Q4"

    assert current_period(["2024Q1", "2024Q2"]) == "2024Q1"


def test_period_with_no_options_is_empty() -> None:
    st.session_state[PERIOD_STATE_KEY] = "2024Q1"

    assert current_period([]) == ""


def test_scoring_profile_uses_its_own_key() -> None:
    st.session_state[SCORING_PROFILE_STATE_KEY] = "Quality"
    st.session_state[PERIOD_STATE_KEY] = "2024Q1"

    assert current_scoring_profile(["Value", "Quality"]) == "Quality"
    assert current_period(["2024Q1"]) == "2024Q1"


def test_scoring_profile_falls_back_when_the_stored_one_was_deleted() -> None:
    st.session_state[SCORING_PROFILE_STATE_KEY] = "Deleted"

    assert current_scoring_profile(["Value", "Quality"]) == "Value"
