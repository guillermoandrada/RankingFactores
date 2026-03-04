"""Combiner registry for factor/metric aggregation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseCombiner(ABC):
    name: str

    @abstractmethod
    def combine(
        self,
        df: pd.DataFrame,
        weights: dict[str, float],
        direction_map: dict[str, bool],
        zsuffix: str,
    ) -> pd.Series:
        """Return combined score series."""


class LinearCombiner(BaseCombiner):
    name = "linear"

    def combine(
        self,
        df: pd.DataFrame,
        weights: dict[str, float],
        direction_map: dict[str, bool],
        zsuffix: str,
    ) -> pd.Series:
        score = pd.Series(0.0, index=df.index, dtype="float64")
        for metric_name, weight in weights.items():
            col = f"{metric_name}{zsuffix}"
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
            sign = 1.0 if direction_map.get(metric_name, True) else -1.0
            score += sign * float(weight) * df[col].fillna(0.0)
        return score


class SoftplusCombiner(BaseCombiner):
    name = "softplus"

    def combine(
        self,
        df: pd.DataFrame,
        weights: dict[str, float],
        direction_map: dict[str, bool],
        zsuffix: str,
    ) -> pd.Series:
        log_score = pd.Series(0.0, index=df.index, dtype="float64")
        for metric_name, weight in weights.items():
            col = f"{metric_name}{zsuffix}"
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
            sign = 1.0 if direction_map.get(metric_name, True) else -1.0
            z = df[col].fillna(0.0).to_numpy(dtype="float64")
            z = np.clip(z, -50, 50)
            softplus = np.log1p(np.exp(z))
            softplus = np.where(softplus <= 0, 1e-12, softplus)
            log_score += sign * float(weight) * np.log(softplus)
        return np.exp(log_score)


class CombinerRegistry:
    def __init__(self) -> None:
        self._combiners: dict[str, BaseCombiner] = {}

    def register(self, combiner: BaseCombiner) -> None:
        self._combiners[combiner.name] = combiner

    def get(self, name: str) -> BaseCombiner:
        if name not in self._combiners:
            raise ValueError(f"Unknown combiner: {name}")
        return self._combiners[name]


def build_default_combiner_registry() -> CombinerRegistry:
    registry = CombinerRegistry()
    registry.register(LinearCombiner())
    registry.register(SoftplusCombiner())
    return registry
