"""The shipped derived-metric formulas and factor scoring profiles keep the conventions they were built with."""

from __future__ import annotations

import pytest

from modules.config.derived_metrics import DerivedMetricStore
from modules.config.ranking_profiles import RankingProfileStore

# Base ratios where a lower value is better; their Change and Trend must point the same way.
LOWER_IS_BETTER_BASES = {
    "P/E Ratio", "P/B Ratio", "P/S Ratio", "EV / EBITDA", "EV / EBITDAR", "Price to Tangible Book",
    "Net Debt / EBITDA", "Total Assets Growth",
    "LT Debt/Assets", "Net Debt/Assets", "Net Debt/OCF",
}
PROFILE_FAMILIES = ("Value_24-09", "Quality_24-09", "Quality_28-09", "Quality_29-09")
# These profiles spread their weight across the quality pillars instead of leaning on one metric.
PILLAR_PROFILES = ("Quality_28-09", "Quality_29-09")
MAX_PILLAR_PROFILE_WEIGHT = 0.25


def _ratio_formulas() -> dict[str, dict]:
    """Change and Trend formulas in the (A - B) / B format; older difference-only formulas are skipped."""
    return {
        name: formula
        for name, formula in DerivedMetricStore().list_formulas().items()
        if name.endswith((" Change", " Trend")) and formula["operations"] == ["-", "/"]
    }


def _family_profiles() -> dict[str, dict]:
    return {
        name: profile
        for name, profile in RankingProfileStore().list_profiles().items()
        if name.startswith(PROFILE_FAMILIES)
    }


def _base_ratio(formula: dict) -> str:
    return formula["metric_names"][0].split(" ", 1)[1]


@pytest.mark.parametrize("name", sorted(_ratio_formulas()))
def test_change_divides_by_past_and_trend_by_current(name):
    formula = _ratio_formulas()[name]
    first, second, third = formula["metric_names"]
    base = _base_ratio(formula)

    assert second == third
    if name.endswith(" Change"):
        assert (first, second) == (f"Current {base}", f"Past {base}")
    else:
        assert (first, second) == (f"Estimated {base}", f"Current {base}")


@pytest.mark.parametrize("name", sorted(_ratio_formulas()))
def test_change_and_trend_keep_the_direction_of_their_base(name):
    formula = _ratio_formulas()[name]

    assert formula["higher_is_better"] is (_base_ratio(formula) not in LOWER_IS_BETTER_BASES)


@pytest.mark.parametrize("name", sorted(_family_profiles()))
def test_factor_profile_weights_sum_to_one(name):
    weights = _family_profiles()[name]["nodes"]["Scoring"]["inputs"]

    assert sum(weights.values()) == pytest.approx(1.0)
    assert min(weights.values()) > 0


@pytest.mark.parametrize("name", sorted(_family_profiles()))
def test_factor_profiles_share_the_value_24_09_configuration(name):
    keys = ("normalization", "winsorization", "winsor_mode", "method")
    reference = RankingProfileStore().get_profile("Value_24-09")
    profile = _family_profiles()[name]

    assert {k: profile[k] for k in keys} == {k: reference[k] for k in keys}


@pytest.mark.parametrize("name", PILLAR_PROFILES)
def test_pillar_profile_caps_every_metric(name):
    weights = RankingProfileStore().get_profile(name)["nodes"]["Scoring"]["inputs"]

    assert max(weights.values()) <= MAX_PILLAR_PROFILE_WEIGHT


def test_accruals_ratio_is_lower_is_better():
    formula = DerivedMetricStore().get_formula("Current Accruals/Assets")

    assert formula["metric_names"] == ["Current Net Income", "Current Operating Cash Flow", "Current Total Assets"]
    assert formula["operations"] == ["-", "/"]
    assert formula["higher_is_better"] is False
