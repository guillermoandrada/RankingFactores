"""A formula may name the same dependency twice: that is a diamond, not a cycle."""

from __future__ import annotations

import pytest

from modules.domain.analytics.metric_loader import resolve_metric_dependencies


BASE_NAMES = {"Current X", "Past X", "Estimated X"}


def test_repeated_base_dependency_resolves_once():
    """The (Current - Past) / Past shape names 'Past X' twice."""
    formulas = {
        "X Change": {
            "metric_names": ["Current X", "Past X", "Past X"],
            "operations": ["-", "/"],
        }
    }
    order = resolve_metric_dependencies("X Change", formulas, BASE_NAMES)

    assert order == ["Current X", "Past X", "X Change"]


def test_repeated_derived_dependency_resolves_once():
    """
    Same shape, but the repeated dependency is itself derived.

    Regression: the resolver used `visited` as if it were the set of everything
    already seen, so the second reference to 'Current Ratio' was reported as a
    circular dependency and the formula could not be created at all.
    """
    formulas = {
        "Current Ratio": {"metric_names": ["Current X", "Past X"], "operations": ["/"]},
        "Past Ratio": {"metric_names": ["Past X", "Estimated X"], "operations": ["/"]},
        "Ratio Change": {
            "metric_names": ["Current Ratio", "Past Ratio", "Past Ratio"],
            "operations": ["-", "/"],
        },
    }
    order = resolve_metric_dependencies("Ratio Change", formulas, BASE_NAMES)

    assert order.count("Past Ratio") == 1
    assert order[-1] == "Ratio Change"
    assert order.index("Past Ratio") < order.index("Ratio Change")
    assert order.index("Current Ratio") < order.index("Ratio Change")


def test_direct_cycle_is_rejected():
    formulas = {"X": {"metric_names": ["Current X", "X"], "operations": ["/"]}}

    with pytest.raises(ValueError, match="Circular dependency"):
        resolve_metric_dependencies("X", formulas, BASE_NAMES)


def test_indirect_cycle_is_rejected():
    formulas = {
        "X": {"metric_names": ["Current X", "Y"], "operations": ["/"]},
        "Y": {"metric_names": ["Past X", "X"], "operations": ["/"]},
    }

    with pytest.raises(ValueError, match="Circular dependency"):
        resolve_metric_dependencies("X", formulas, BASE_NAMES)


def test_unknown_dependency_is_rejected():
    formulas = {"X": {"metric_names": ["Current X", "Nope"], "operations": ["/"]}}

    with pytest.raises(ValueError, match="not found"):
        resolve_metric_dependencies("X", formulas, BASE_NAMES)
