"""
Z-score calculation with winsorization.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from modules.config import DB_URL


class ZScoreCalculator:
    """
    Computes winsorized z-scores for fundamental metrics.
    Supports filtering by index and industry.
    """

    def __init__(self, engine: Optional[Engine] = None) -> None:
        from sqlalchemy import create_engine
        self._engine = engine or create_engine(DB_URL)

    def compute(
        self,
        period: str,
        metric_ids: list[int],
        index_name: Optional[str] = None,
        industry_name: Optional[str] = None,
    ) -> tuple[pd.DataFrame, dict[str, bool]]:
        """
        Compute z-scores for each security and metric.
        Applies 1%-99% winsorization before z-scoring.

        Returns:
            df: DataFrame indexed by (security_id, ticker) with metric columns
                and *_zscore columns.
            direction_map: {metric_name: higher_is_better}
        """
        if not metric_ids:
            raise ValueError("metric_ids cannot be empty.")

        df_long = self._fetch_fundamentals(
            period, metric_ids, index_name, industry_name
        )

        direction_map = self._build_direction_map(df_long)

        df_wide = (
            df_long.pivot_table(
                index=["security_id", "ticker"],
                columns="metric_name",
                values="value",
            )
            .sort_index()
        )
        df_wide.columns = [str(c) for c in df_wide.columns]
        metric_cols = df_wide.columns.tolist()

        df_wins = self._winsorize(df_wide, metric_cols)
        df_z = self._compute_zscores(df_wins, metric_cols)

        return df_z, direction_map

    def _fetch_fundamentals(
        self,
        period: str,
        metric_ids: list[int],
        index_name: Optional[str],
        industry_name: Optional[str],
    ) -> pd.DataFrame:
        placeholders = ", ".join(f":m{i}" for i in range(len(metric_ids)))
        joins = [
            "JOIN securities s ON s.id = fv.security_id",
            "JOIN metrics m ON m.metric_id = fv.metric_id",
        ]
        where = [
            "fv.period = :as_of_date",
            f"fv.metric_id IN ({placeholders})",
        ]
        params: dict = {"as_of_date": period}
        params.update({f"m{i}": mid for i, mid in enumerate(metric_ids)})

        # FIX: Use index_membership (singular), not index_memberships
        if index_name:
            joins.append(
                "JOIN index_membership im "
                "ON im.security_id = fv.security_id "
                "AND im.period = :as_of_date"
            )
            joins.append("JOIN indices idx ON idx.index_id = im.index_id")
            where.append("idx.name = :index_name")
            params["index_name"] = index_name

        # FIX: Use industry_id and industry_name (not id/name)
        if industry_name:
            joins.append(
                "JOIN industries ind ON ind.industry_id = s.industry_id"
            )
            where.append("ind.industry_name = :industry_name")
            params["industry_name"] = industry_name

        sql = text(
            f"""
            SELECT
                fv.security_id,
                s.ticker,
                fv.metric_id,
                m.metric_name AS metric_name,
                m.higher_is_better AS higher_is_better,
                fv.value
            FROM fundamental_values fv
            {' '.join(joins)}
            WHERE {' AND '.join(where)}
            """
        )

        df = pd.read_sql_query(sql, con=self._engine, params=params)
        if df.empty:
            raise ValueError(
                "No data found for the given period, metrics, and filters."
            )
        return df

    def _build_direction_map(self, df: pd.DataFrame) -> dict[str, bool]:
        rows = (
            df[["metric_name", "higher_is_better"]]
            .drop_duplicates(subset=["metric_name"])
        )
        result = {}
        for _, row in rows.iterrows():
            name = str(row["metric_name"])
            hib = row["higher_is_better"]
            result[name] = bool(hib) if hib is not None else True
        return result

    def _winsorize(
        self, df: pd.DataFrame, cols: list[str], lower: float = 0.01, upper: float = 0.99
    ) -> pd.DataFrame:
        out = df.copy()
        for col in cols:
            s = out[col]
            lo, hi = s.quantile(lower), s.quantile(upper)
            out[col] = s.clip(lower=lo, upper=hi)
        return out

    def _compute_zscores(
        self, df: pd.DataFrame, cols: list[str], suffix: str = "_zscore"
    ) -> pd.DataFrame:
        out = df.copy()
        for col in cols:
            s = out[col]
            mean = s.mean()
            std = s.std(ddof=0)
            filled = s.fillna(mean)
            if std == 0 or pd.isna(std):
                z = pd.Series(0.0, index=out.index)
            else:
                z = (filled - mean) / std
            out[f"{col}{suffix}"] = z
        return out
