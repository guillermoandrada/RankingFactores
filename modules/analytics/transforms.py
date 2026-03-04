"""Transform registry for metric preprocessing and normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class TransformSpec:
    """Declarative transform specification."""

    name: str
    params: dict[str, Any] | None = None


class BaseTransform(ABC):
    """Base class for column transforms."""

    name: str

    @abstractmethod
    def apply(self, series: pd.Series, **params: Any) -> pd.Series:
        """Apply transform to a metric series."""


class IdentityTransform(BaseTransform):
    name = "identity"

    def apply(self, series: pd.Series, **params: Any) -> pd.Series:
        return series.copy()


class WinsorTransform(BaseTransform):
    name = "winsor"

    def apply(self, series: pd.Series, **params: Any) -> pd.Series:
        lower = float(params.get("lower", 0.01))
        upper = float(params.get("upper", 0.99))
        lo = series.quantile(lower)
        hi = series.quantile(upper)
        return series.clip(lower=lo, upper=hi)


class ZScoreTransform(BaseTransform):
    name = "zscore"

    def apply(self, series: pd.Series, **params: Any) -> pd.Series:
        mean = series.mean()
        std = series.std(ddof=0)
        filled = series.fillna(mean)
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index)
        return (filled - mean) / std


class NormalizedZScoreTransform(BaseTransform):
    """
    Z-score with linear interpolation to [0, 10].
    Computes z-score, then maps min→0 and max→10 via linear interpolation.
    """

    name = "normalized_zscore"

    def apply(self, series: pd.Series, **params: Any) -> pd.Series:
        mean = series.mean()
        std = series.std(ddof=0)
        filled = series.fillna(mean)
        if std == 0 or pd.isna(std):
            return pd.Series(5.0, index=series.index)
        z = (filled - mean) / std
        z_min, z_max = z.min(), z.max()
        if z_max == z_min:
            return pd.Series(5.0, index=series.index)
        return (z - z_min) / (z_max - z_min) * 10.0


class PercentileTransform(BaseTransform):
    """
    Map values to percentile rank [0, 100].
    Higher raw values map to higher percentiles when higher_is_better.
    """

    name = "percentile"

    def apply(self, series: pd.Series, **params: Any) -> pd.Series:
        filled = series.fillna(series.median())
        return filled.rank(pct=True, method="average") * 100


class TransformRegistry:
    """Runtime registry of available transforms."""

    def __init__(self) -> None:
        self._transforms: dict[str, BaseTransform] = {}

    def register(self, transform: BaseTransform) -> None:
        self._transforms[transform.name] = transform

    def get(self, name: str) -> BaseTransform:
        if name not in self._transforms:
            raise ValueError(f"Unknown transform: {name}")
        return self._transforms[name]

    def apply_chain(
        self,
        series: pd.Series,
        specs: list[dict[str, Any]] | list[TransformSpec],
        stop_before: str | list[str] | None = None,
    ) -> pd.Series:
        """Apply transform chain. If stop_before is set, stop before that transform (or any of them)."""
        stop_set = {stop_before} if isinstance(stop_before, str) else set(stop_before or ())
        out = series.copy()
        for spec in specs:
            if isinstance(spec, TransformSpec):
                name = spec.name
                params = spec.params or {}
            else:
                name = str(spec.get("name", "")).strip()
                params = spec.get("params", {}) or {}
            if stop_set and name in stop_set:
                break
            transform = self.get(name)
            out = transform.apply(out, **params)
        return out


def build_default_transform_registry() -> TransformRegistry:
    registry = TransformRegistry()
    registry.register(IdentityTransform())
    registry.register(WinsorTransform())
    registry.register(ZScoreTransform())
    registry.register(NormalizedZScoreTransform())
    registry.register(PercentileTransform())
    return registry


DEFAULT_TRANSFORM_CHAIN = [
    {"name": "winsor", "params": {"lower": 0.01, "upper": 0.99}},
    {"name": "zscore", "params": {}},
]
