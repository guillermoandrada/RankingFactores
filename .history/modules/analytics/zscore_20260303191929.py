"""Metric matrix loading plus transform-based score preparation."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from sqlalchemy.engine import Engine

from modules.analytics.metric_loader import fetch_metric_matrix
from modules.analytics.transforms import (
    DEFAULT_TRANSFORM_CHAIN,
    build_default_transform_registry,
)
from modules.config import DB_URL
from modules.config.derived_metrics import DerivedMetricStore


class ZScoreCalculator:
    """
    Loads fundamental metric values and builds transformed metric scores.
    Fetches base metrics from DB and computes derived metrics on the fly.

    NA policy:
    - Raw NA metric values are preserved in the database as NULL.
    - During transformation, each transform defines how NA is handled.
      Default zscore fills NA with column mean for score construction.
    """

    def __init__(
        self,
        engine: Optional[Engine] = None,
        derived_store: Optional[DerivedMetricStore] = None,
    ) -> None:
        from sqlalchemy import create_engine

        self._engine = engine or create_engine(DB_URL)
        self._derived_store = derived_store or DerivedMetricStore()
        self._transforms = build_default_transform_registry()

    def compute(
        self,
        period: str,
        metric_names: list[str],
        index_name: Optional[str] = None,
        industry_name: Optional[str] = None,
        sector_name: Optional[str] = None,
        transform_chain: Optional[list[dict[str, Any]]] = None,
        out_suffix: str = "_zscore",
    ) -> tuple[pd.DataFrame, dict[str, bool]]:
        """Compute transformed scores (default winsor + zscore)."""
        if not metric_names:
            raise ValueError("metric_names cannot be empty.")

        df_wide, direction_map = fetch_metric_matrix(
            engine=self._engine,
            period=period,
            metric_names=metric_names,
            derived_store=self._derived_store,
            index_name=index_name,
            industry_name=industry_name,
            sector_name=sector_name,
        )

        chain = transform_chain or DEFAULT_TRANSFORM_CHAIN
        result = df_wide.copy()
        # Only transform metrics that the profile uses; skip composing/dependency metrics
        for col in metric_names:
            if col not in result.columns:
                continue
            result[f"{col}{out_suffix}"] = self._transforms.apply_chain(
                result[col],
                chain,
            )
        # Drop columns for metrics not in the profile (dependencies used only for derived)
        keep = [c for c in metric_names if c in result.columns]
        keep += [f"{m}{out_suffix}" for m in keep]
        result = result[[c for c in result.columns if c in keep]]
        return result, direction_map
