"""
Ranking computation: linear and softplus combination of z-scores.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class RankingEngine:
    """
    Combines z-scores into a single ranking score.
    Supports linear (weighted sum) and softplus (product) methods.
    """

    ZSUFFIX = "_zscore"

    def compute(
        self,
        df_z: pd.DataFrame,
        weights: dict[str, float],
        direction_map: dict[str, bool],
        method: str = "linear",
        out_col: str = "score",
    ) -> pd.DataFrame:
        """
        Compute ranking score from z-score DataFrame.

        Args:
            df_z: DataFrame with *_zscore columns.
            weights: {metric_name: weight}
            direction_map: {metric_name: higher_is_better}
            method: "linear" or "softplus"
            out_col: Output column name.
        """
        if not weights:
            raise ValueError("weights cannot be empty.")

        if method == "linear":
            return self._linear_combination(
                df_z, weights, direction_map, out_col
            )
        if method == "softplus":
            return self._softplus_combination(
                df_z, weights, direction_map, out_col
            )
        raise ValueError(f"Unknown method: {method}. Use 'linear' or 'softplus'.")

    def _linear_combination(
        self,
        df: pd.DataFrame,
        weights: dict[str, float],
        direction_map: dict[str, bool],
        out_col: str,
    ) -> pd.DataFrame:
        """Score = sum_i (sign_i * weight_i * z_i)"""
        result = df.copy()
        score = pd.Series(0.0, index=df.index, dtype="float64")

        for metric_name, w in weights.items():
            col = f"{metric_name}{self.ZSUFFIX}"
            self._assert_column(df, col)
            sign = 1.0 if direction_map.get(metric_name, True) else -1.0
            score += sign * float(w) * df[col].fillna(0.0)

        result[out_col] = score
        return result

    def _softplus_combination(
        self,
        df: pd.DataFrame,
        weights: dict[str, float],
        direction_map: dict[str, bool],
        out_col: str,
    ) -> pd.DataFrame:
        """Score = prod_i (softplus(z_i) ** (sign_i * weight_i))"""
        result = df.copy()
        log_score = pd.Series(0.0, index=df.index, dtype="float64")

        for metric_name, w in weights.items():
            col = f"{metric_name}{self.ZSUFFIX}"
            self._assert_column(df, col)
            sign = 1.0 if direction_map.get(metric_name, True) else -1.0
            z = df[col].fillna(0.0).to_numpy(dtype="float64")
            s = self._softplus(z)
            s = np.where(s <= 0, 1e-12, s)
            log_score += sign * float(w) * np.log(s)

        result[out_col] = np.exp(log_score)
        return result

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        """Numerically stable softplus(x) = ln(1 + e^x)."""
        x = np.clip(x, -50, 50)
        return np.log1p(np.exp(x))

    def _assert_column(self, df: pd.DataFrame, col: str) -> None:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")


def export_to_excel(
    df: pd.DataFrame, filepath: str, index: bool = True
) -> None:
    """Export DataFrame to Excel file."""
    df.to_excel(filepath, index=index)
