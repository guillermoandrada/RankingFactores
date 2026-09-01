"""Where a derived metric is referenced, from data the Metrics page already fetches."""

from __future__ import annotations

import pathlib

import pytest

_PAGE = pathlib.Path(__file__).resolve().parents[1] / "streamlit_app" / "pages" / "2_Metrics.py"


@pytest.fixture(scope="module")
def page_helpers() -> dict:
    """
    Expose the page's module-level helpers without running its Streamlit body.

    Everything above the first `def _render_create_metric_tab` is definitions only.
    """
    source = _PAGE.read_text(encoding="utf-8-sig")
    definitions = source.split("def _render_create_metric_tab")[0]
    namespace: dict = {}
    exec(compile(definitions, str(_PAGE), "exec"), namespace)
    return namespace


def _profile(nodes: dict) -> dict:
    return {"nodes": nodes, "normalization": "zscore"}


# --- metrics_referenced_by_profile -----------------------------------------------------


def test_leaf_inputs_are_metrics(page_helpers) -> None:
    profile = _profile({"root": {"inputs": {"ROE": 0.5, "Debt": 0.5}}})

    assert page_helpers["metrics_referenced_by_profile"](profile) == {"ROE", "Debt"}


def test_inputs_naming_another_node_are_subfactors_not_metrics(page_helpers) -> None:
    """A profile stores subfactors and metrics the same way; only the leaves are metrics."""
    profile = _profile(
        {
            "root": {"inputs": {"quality": 1.0}},
            "quality": {"inputs": {"ROE": 0.7, "Margin": 0.3}},
        }
    )

    assert page_helpers["metrics_referenced_by_profile"](profile) == {"ROE", "Margin"}


def test_profile_without_nodes_references_nothing(page_helpers) -> None:
    assert page_helpers["metrics_referenced_by_profile"]({}) == set()
    assert page_helpers["metrics_referenced_by_profile"]({"nodes": None}) == set()


# --- find_metric_usage -----------------------------------------------------------------


@pytest.fixture
def derived() -> list[dict]:
    return [
        {"metric_name": "Book to Price Change", "metric_names": ["Current B/P", "Past B/P"]},
        {"metric_name": "Composite", "metric_names": ["Book to Price Change", "ROE"]},
        {"metric_name": "Unrelated", "metric_names": ["A", "B"]},
    ]


def test_finds_derived_metrics_that_depend_on_it(page_helpers, derived) -> None:
    usage = page_helpers["find_metric_usage"]("Book to Price Change", derived, {})

    assert usage["derived_metrics"] == ["Composite"]


def test_finds_profiles_that_use_it(page_helpers, derived) -> None:
    profiles = {
        "Value": _profile({"root": {"inputs": {"Book to Price Change": 1.0}}}),
        "Momentum": _profile({"root": {"inputs": {"ROE": 1.0}}}),
    }

    usage = page_helpers["find_metric_usage"]("Book to Price Change", derived, profiles)

    assert usage["scoring_profiles"] == ["Value"]


def test_a_metric_does_not_count_as_using_itself(page_helpers) -> None:
    derived = [{"metric_name": "Self", "metric_names": ["Self", "Other"]}]

    assert page_helpers["find_metric_usage"]("Self", derived, {})["derived_metrics"] == []


def test_unused_metric_reports_nothing(page_helpers, derived) -> None:
    usage = page_helpers["find_metric_usage"]("Composite", derived, {})

    assert usage == {"derived_metrics": [], "scoring_profiles": []}


def test_results_are_sorted(page_helpers) -> None:
    derived = [
        {"metric_name": "Zeta", "metric_names": ["Target", "X"]},
        {"metric_name": "Alpha", "metric_names": ["Target", "Y"]},
    ]
    profiles = {
        "Zulu": _profile({"root": {"inputs": {"Target": 1.0}}}),
        "Alfa": _profile({"root": {"inputs": {"Target": 1.0}}}),
    }

    usage = page_helpers["find_metric_usage"]("Target", derived, profiles)

    assert usage["derived_metrics"] == ["Alpha", "Zeta"]
    assert usage["scoring_profiles"] == ["Alfa", "Zulu"]


def test_missing_profiles_argument_is_tolerated(page_helpers, derived) -> None:
    assert page_helpers["find_metric_usage"]("ROE", derived, None)["scoring_profiles"] == []
