"""Ranking computation based on registered combiners."""

from __future__ import annotations

import pandas as pd

from modules.analytics.combiners import build_default_combiner_registry


class RankingEngine:
    """Combines transformed metric columns into a score."""

    def __init__(self, zsuffix: str = "_zscore") -> None:
        self.zsuffix = zsuffix
        self._registry = build_default_combiner_registry()

    def compute(
        self,
        df_z: pd.DataFrame,
        weights: dict[str, float],
        direction_map: dict[str, bool],
        method: str = "linear",
        out_col: str = "score",
    ) -> pd.DataFrame:
        if not weights:
            raise ValueError("weights cannot be empty.")

        combiner = self._registry.get(method)
        out = df_z.copy()
        out[out_col] = combiner.combine(
            df=df_z,
            weights=weights,
            direction_map=direction_map,
            zsuffix=self.zsuffix,
        )
        return out


def export_to_excel(df: pd.DataFrame, filepath: str, index: bool = True) -> None:
    """Export DataFrame to Excel file."""
    df.to_excel(filepath, index=index)
