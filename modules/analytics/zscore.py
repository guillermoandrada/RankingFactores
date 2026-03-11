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
        # Treat both quantile winsor ("winsor") and semi-winsor ("semi_winsor")
        # as winsorization steps for preview/output purposes.
        winsor_names = {"winsor", "semi_winsor"}
        use_winsor = any(
            str(s.get("name", "")).strip() in winsor_names
            for s in chain
        )
        result = df_wide.copy()
        # Only transform metrics in the profile (metric_names), not composing base metrics
        for col in metric_names:
            if col not in result.columns:
                continue
            if use_winsor:
                result[f"{col}_winsor"] = self._transforms.apply_chain(
                    result[col], chain, stop_before=["zscore", "normalized_zscore", "percentile"]
                )
            result[f"{col}{out_suffix}"] = self._transforms.apply_chain(result[col], chain)
        # Keep profile metrics and their transforms. When winsor on: _winsor + _zscore.
        # When winsor off: raw value (for display) + _zscore, so metrics flow into the ranking table.
        keep_cols = []
        for col in metric_names:
            if col not in result.columns:
                continue
            if use_winsor:
                keep_cols.append(f"{col}_winsor")
            else:
                keep_cols.append(col)  # raw value for display when no winsorization
            keep_cols.append(f"{col}{out_suffix}")
        result = result[keep_cols]
        return result, direction_map
