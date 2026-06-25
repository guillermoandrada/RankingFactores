"""Pure helpers for constraint target dataframes."""

from __future__ import annotations

import pandas as pd

from streamlit_app.ui.constraints import targets_from_dataframe


def test_targets_from_dataframe_enabled_only_and_fraction() -> None:
    df = pd.DataFrame(
        [
            {"enabled": True, "group": "Tech", "weight": 30.0},
            {"enabled": False, "group": "Energy", "weight": 50.0},
            {"enabled": True, "group": "Health", "weight": 12.5},
        ]
    )
    assert targets_from_dataframe(df) == {"Tech": 0.3, "Health": 0.125}


def test_targets_from_dataframe_empty() -> None:
    assert targets_from_dataframe(pd.DataFrame()) == {}
