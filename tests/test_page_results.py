"""Persisted page results: storage, staleness, and caption formatting."""

from __future__ import annotations

import pytest
import streamlit as st

from streamlit_app.ui.results import (
    PageResult,
    clear_result,
    format_inputs,
    is_stale,
    read_result,
    store_result,
)

_KEY = "test_page_result"


@pytest.fixture(autouse=True)
def _clean_session_state():
    st.session_state.clear()
    yield
    st.session_state.clear()


def test_read_returns_none_before_anything_is_stored() -> None:
    assert read_result(_KEY) is None


def test_store_then_read_round_trip() -> None:
    store_result(_KEY, {"rows": 3}, inputs={"Metric": "ROE"})

    result = read_result(_KEY)

    assert isinstance(result, PageResult)
    assert result.payload == {"rows": 3}
    assert result.inputs == {"Metric": "ROE"}


def test_stored_inputs_are_copied_so_later_mutation_cannot_corrupt_them() -> None:
    inputs = {"Metric": "ROE"}
    store_result(_KEY, "payload", inputs=inputs)
    inputs["Metric"] = "Debt"

    assert read_result(_KEY).inputs == {"Metric": "ROE"}


def test_clear_removes_the_result() -> None:
    store_result(_KEY, "payload", inputs={})
    clear_result(_KEY)
    assert read_result(_KEY) is None


def test_unrelated_session_value_is_not_mistaken_for_a_result() -> None:
    st.session_state[_KEY] = "not a PageResult"
    assert read_result(_KEY) is None


def test_matching_inputs_are_not_stale() -> None:
    store_result(_KEY, "payload", inputs={"Metric": "ROE", "Breakdown": "Total"})

    assert not is_stale(read_result(_KEY), {"Metric": "ROE", "Breakdown": "Total"})


def test_changed_inputs_are_stale() -> None:
    store_result(_KEY, "payload", inputs={"Metric": "ROE", "Breakdown": "Total"})

    assert is_stale(read_result(_KEY), {"Metric": "ROE", "Breakdown": "By sector"})


def test_format_inputs_renders_one_line() -> None:
    assert format_inputs({"Metric": "ROE", "Breakdown": "Total"}) == "Metric: ROE · Breakdown: Total"


def test_format_inputs_joins_collections() -> None:
    assert format_inputs({"Metrics": ["ROE", "Debt"]}) == "Metrics: ROE, Debt"


def test_format_inputs_marks_an_empty_collection() -> None:
    assert format_inputs({"Metrics": []}) == "Metrics: —"
