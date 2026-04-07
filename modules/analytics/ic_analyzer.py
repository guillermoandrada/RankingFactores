"""
Information Coefficient (IC) analyzer using cross-sectional Spearman rank correlation.

Supports multivariate factor selection: predictive Rank IC vs forward returns and
inter-factor Spearman correlation (colinearity) on shared cross-sections per period.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import MetaData, and_, select
from sqlalchemy.engine import Engine

from modules.db import FinancialDatabase
from yfinance_service import YFinanceService


@dataclass(frozen=True)
class ICPoint:
    period: str
    start_date: str
    end_date: str
    ic: float
    n: int


class ICAnalyzer:
    """
    Compute Rank IC (Spearman) vs forward returns and inter-factor Spearman matrices.

    - Cross-section at each period: fundamental_values joined with securities -> ticker.
    - Forward return window: [start_date, end_date] derived from period end + publication lag.
    - Robustness: ignores rows with null fundamentals or missing/invalid price data.
    """

    _DEFAULT_PUBLICATION_LAG_DAYS = 45

    def __init__(
        self,
        db: Optional[FinancialDatabase] = None,
        *,
        engine: Optional[Engine] = None,
        price_service: Optional[YFinanceService] = None,
        publication_lag_days: int = _DEFAULT_PUBLICATION_LAG_DAYS,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if engine is None:
            db = db or FinancialDatabase()
            engine = db.engine
        self._engine = engine
        self._price_service = price_service or YFinanceService()
        self._publication_lag_days = int(publication_lag_days)
        self._logger = logger or logging.getLogger(__name__)

        self._metadata = MetaData()
        self._metadata.reflect(bind=self._engine)
        self._tbl_fund = self._metadata.tables["fundamental_values"]
        self._tbl_sec = self._metadata.tables["securities"]
        self._tbl_metrics = self._metadata.tables["metrics"]

    def analyze_multivariate(
        self,
        *,
        metric_names: list[str],
        forward_months: int,
        periods: Optional[Iterable[str]] = None,
    ) -> dict:
        """
        Multivariate factor analysis.

        1) Predictive: for each metric, cross-sectional Spearman (Rank IC) vs forward returns
           per period (same logic as single-metric IC).
        2) Colinearity: per period, Spearman correlation matrix between metrics on the shared
           cross-section (inner join on tickers with all metrics present); matrices are
           averaged across periods.

        Args:
            metric_names: DB metric names (at least two distinct names).
            forward_months: Forward horizon in months.
            periods: Optional filter; if None, periods are taken per metric / intersection
                as described below.

        Returns:
            {
              "metric_names": list[str],
              "forward_months": int,
              "predictive": list[dict]  # per metric: mean_rank_ic, ic_std, information_ratio, n_periods, series
              "inter_factor_correlation": {"labels": list[str], "matrix": list[list[float|None]]}
              "warnings": list[str],
            }
        """
        forward_months = int(forward_months)
        if forward_months <= 0:
            raise ValueError("forward_months must be positive.")

        unique_names = list(dict.fromkeys(str(n).strip() for n in metric_names if str(n).strip()))
        if len(unique_names) < 2:
            raise ValueError("Select at least two distinct metric names.")

        warnings: list[str] = []
        name_to_id = self._resolve_metric_ids(unique_names, warnings)
        if len(name_to_id) < 2:
            raise ValueError("Could not resolve at least two valid metrics from the database.")

        predictive: list[dict] = []
        for name in unique_names:
            mid = name_to_id.get(name)
            if mid is None:
                continue
            use_periods = periods
            if use_periods is None:
                use_periods = self._list_periods_for_metric(mid)
            points, w = self._ic_series_for_metric(
                metric_id=mid,
                forward_months=forward_months,
                periods=use_periods,
            )
            warnings.extend(w)
            ic_vals = [p.ic for p in points]
            mean_ic = float(np.mean(ic_vals)) if ic_vals else None
            std_ic = float(np.std(ic_vals, ddof=1)) if len(ic_vals) >= 2 else None
            ir = (
                (mean_ic / std_ic)
                if (mean_ic is not None and std_ic not in (None, 0.0))
                else None
            )
            predictive.append(
                {
                    "metric_name": name,
                    "mean_rank_ic": mean_ic,
                    "ic_std": std_ic,
                    "information_ratio": ir,
                    "n_periods": len(points),
                    "series": [
                        {
                            "period": p.period,
                            "start_date": p.start_date,
                            "end_date": p.end_date,
                            "ic": p.ic,
                            "n": p.n,
                        }
                        for p in points
                    ],
                }
            )

        inter_labels = [n for n in unique_names if n in name_to_id]
        inter_matrix = self._inter_factor_spearman_matrix(
            name_to_id={n: name_to_id[n] for n in inter_labels},
            periods=periods,
            warnings=warnings,
        )

        return {
            "metric_names": inter_labels,
            "forward_months": forward_months,
            "predictive": predictive,
            "inter_factor_correlation": {
                "labels": inter_labels,
                "matrix": inter_matrix,
            },
            "warnings": warnings,
        }

    def analyze_metric(
        self,
        *,
        metric_id: int,
        forward_months: int,
        periods: Optional[Iterable[str]] = None,
    ) -> dict:
        """
        Single-metric IC summary + per-period IC_t series (backward compatible).

        Returns:
            {
              "metric_id": int,
              "forward_months": int,
              "ic_mean": float|None,
              "ic_std": float|None,
              "ir": float|None,
              "series": list[dict],
              "warnings": list[str],
            }
        """
        forward_months = int(forward_months)
        if forward_months <= 0:
            raise ValueError("forward_months must be positive.")

        if periods is None:
            periods = self._list_periods_for_metric(metric_id)
        points, warnings = self._ic_series_for_metric(
            metric_id=metric_id,
            forward_months=forward_months,
            periods=periods,
        )

        ic_series = [p.ic for p in points]
        ic_mean = float(np.mean(ic_series)) if ic_series else None
        ic_std = float(np.std(ic_series, ddof=1)) if len(ic_series) >= 2 else None
        ir = (ic_mean / ic_std) if (ic_mean is not None and ic_std not in (None, 0.0)) else None

        return {
            "metric_id": int(metric_id),
            "forward_months": int(forward_months),
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ir": ir,
            "series": [
                {
                    "period": p.period,
                    "start_date": p.start_date,
                    "end_date": p.end_date,
                    "ic": p.ic,
                    "n": p.n,
                }
                for p in points
            ],
            "warnings": warnings,
        }

    def _ic_series_for_metric(
        self,
        *,
        metric_id: int,
        forward_months: int,
        periods: Optional[Iterable[str]],
    ) -> tuple[list[ICPoint], list[str]]:
        periods_list = sorted({p for p in (periods or []) if p})
        warnings: list[str] = []
        points: list[ICPoint] = []

        for period in periods_list:
            try:
                start_date, end_date = self._parse_period_to_dates(period, forward_months)
            except Exception as exc:
                self._logger.debug("Skipping period '%s': cannot parse dates (%s)", period, exc)
                warnings.append(f"Skipped period '{period}': invalid period format.")
                continue

            fundamentals = self._load_cross_section(metric_id=metric_id, period=period)
            if fundamentals.empty:
                continue

            fundamentals["value"] = pd.to_numeric(fundamentals["value"], errors="coerce")
            fundamentals = fundamentals.dropna(subset=["ticker", "value"])
            if fundamentals.empty:
                continue

            returns = self._compute_forward_returns(
                tickers=fundamentals["ticker"].astype(str).tolist(),
                start_date=start_date,
                end_date=end_date,
            )
            if returns.empty:
                continue

            merged = fundamentals.merge(returns, on="ticker", how="inner")
            merged["forward_return"] = pd.to_numeric(merged["forward_return"], errors="coerce")
            merged = merged.dropna(subset=["value", "forward_return"])
            if len(merged) < 2:
                continue

            ic_val = self._spearman_ic(merged["value"].to_numpy(), merged["forward_return"].to_numpy())
            if ic_val is None:
                continue

            points.append(
                ICPoint(
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    ic=float(ic_val),
                    n=int(len(merged)),
                )
            )

        return points, warnings

    def _inter_factor_spearman_matrix(
        self,
        *,
        name_to_id: dict[str, int],
        periods: Optional[Iterable[str]],
        warnings: list[str],
    ) -> list[list[Optional[float]]]:
        """Average cross-sectional Spearman correlation matrices across periods (inner join on tickers)."""
        labels = list(name_to_id.keys())
        if len(labels) < 2:
            return []

        if periods is None:
            period_sets = [set(self._list_periods_for_metric(mid)) for mid in name_to_id.values()]
            common = set.intersection(*period_sets) if period_sets else set()
            period_list = sorted(common)
        else:
            period_list = sorted({p for p in periods if p})

        if not period_list:
            warnings.append("No shared periods found for inter-factor correlation.")
            return [[None] * len(labels) for _ in labels]

        matrices: list[np.ndarray] = []
        for period in period_list:
            wide = self._load_wide_cross_section(name_to_id=name_to_id, period=period)
            if wide.shape[0] < 2:
                continue
            sub = wide[labels].apply(pd.to_numeric, errors="coerce")
            if sub.isna().all().all():
                continue
            try:
                corr = sub.corr(method="spearman", min_periods=2)
            except Exception as exc:
                self._logger.debug("Corr failed for period %s: %s", period, exc)
                continue
            mat = corr.reindex(index=labels, columns=labels)
            matrices.append(mat.values.astype(float))

        if not matrices:
            warnings.append(
                "Could not compute inter-factor correlations (need at least two tickers "
                "with overlapping metric values per period)."
            )
            return [[None] * len(labels) for _ in labels]

        stacked = np.stack(matrices, axis=0)
        with np.errstate(invalid="ignore"):
            mean_mat = np.nanmean(stacked, axis=0)

        out: list[list[Optional[float]]] = []
        for i in range(mean_mat.shape[0]):
            row: list[Optional[float]] = []
            for j in range(mean_mat.shape[1]):
                v = mean_mat[i, j]
                row.append(None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v))
            out.append(row)
        return out

    def _resolve_metric_ids(self, names: list[str], warnings: list[str]) -> dict[str, int]:
        tbl = self._tbl_metrics
        out: dict[str, int] = {}
        with self._engine.connect() as conn:
            for name in names:
                row = conn.execute(
                    select(tbl.c.metric_id).where(tbl.c.metric_name == name)
                ).first()
                if row:
                    out[name] = int(row[0])
                else:
                    warnings.append(f"Metric not found in database: '{name}'.")
        return out

    def _load_wide_cross_section(self, *, name_to_id: dict[str, int], period: str) -> pd.DataFrame:
        """Inner join all metrics on ticker; columns = metric names."""
        labels = list(name_to_id.keys())
        merged: Optional[pd.DataFrame] = None
        for name in labels:
            mid = name_to_id[name]
            df = self._load_cross_section(metric_id=mid, period=period)
            if df.empty:
                return pd.DataFrame()
            df = df.rename(columns={"value": name})
            df = df[["ticker", name]]
            df[name] = pd.to_numeric(df[name], errors="coerce")
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on="ticker", how="inner")
        return merged if merged is not None else pd.DataFrame()

    def _parse_period_to_dates(self, period: str, forward_months: int) -> tuple[str, str]:
        quarter_end = self._period_to_quarter_end_date(period)
        start = quarter_end + timedelta(days=self._publication_lag_days)
        end = self._add_months(start, int(forward_months))
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def _list_periods_for_metric(self, metric_id: int) -> list[str]:
        tbl = self._tbl_fund
        q = select(tbl.c.period).distinct().where(tbl.c.metric_id == metric_id).order_by(tbl.c.period)
        with self._engine.connect() as conn:
            rows = conn.execute(q).fetchall()
        return [r[0] for r in rows if r and r[0]]

    def _load_cross_section(self, *, metric_id: int, period: str) -> pd.DataFrame:
        tbl_f = self._tbl_fund
        tbl_s = self._tbl_sec
        q = (
            select(
                tbl_s.c.ticker.label("ticker"),
                tbl_f.c.value.label("value"),
            )
            .select_from(tbl_f)
            .join(tbl_s, tbl_s.c.id == tbl_f.c.security_id)
            .where(and_(tbl_f.c.metric_id == metric_id, tbl_f.c.period == period))
        )
        with self._engine.connect() as conn:
            df = pd.read_sql_query(q, con=conn)
        return df

    def _compute_forward_returns(
        self,
        *,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for ticker in tickers:
            t = (ticker or "").strip()
            if not t:
                continue
            try:
                returns = self._price_service.fetch_historical_returns(
                    t,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as exc:
                self._logger.debug("Returns fetch failed for %s: %s", t, exc)
                continue

            if not returns:
                continue

            try:
                daily = [float(r) for _, r in returns if r is not None and not (isinstance(r, float) and np.isnan(r))]
            except Exception:
                continue

            if not daily:
                continue

            try:
                geom = float(np.prod(1.0 + np.array(daily, dtype=float)) - 1.0)
            except Exception:
                continue

            if np.isnan(geom) or np.isinf(geom):
                continue

            rows.append({"ticker": t, "forward_return": geom})

        return pd.DataFrame(rows)

    @staticmethod
    def _spearman_ic(x: np.ndarray, y: np.ndarray) -> Optional[float]:
        if x.size != y.size or x.size < 2:
            return None
        mask = ~(np.isnan(x) | np.isnan(y))
        x2 = x[mask]
        y2 = y[mask]
        if x2.size < 2:
            return None
        try:
            corr, _ = spearmanr(x2, y2)
        except Exception:
            return None
        if corr is None or np.isnan(corr):
            return None
        return float(corr)

    @staticmethod
    def _period_to_quarter_end_date(period: str) -> date:
        text = str(period or "").strip().upper()
        text = text.replace("-", " ").replace("_", " ")
        text = " ".join(text.split())
        if "Q" not in text:
            raise ValueError("Period must contain a quarter like 'Q1'.")

        parts = text.split()
        if len(parts) == 2 and parts[1].startswith("Q"):
            year_str, q_str = parts[0], parts[1]
        else:
            year_str = text[:4]
            q_str = text[4:].strip()

        year = int(year_str)
        quarter = int(q_str.replace("Q", "").strip())
        if quarter not in (1, 2, 3, 4):
            raise ValueError("Quarter must be 1..4.")

        month_day = {
            1: (3, 31),
            2: (6, 30),
            3: (9, 30),
            4: (12, 31),
        }[quarter]
        return date(year, month_day[0], month_day[1])

    @staticmethod
    def _add_months(d: date, months: int) -> date:
        if months == 0:
            return d
        y = d.year + (d.month - 1 + months) // 12
        m = (d.month - 1 + months) % 12 + 1
        last_day = ICAnalyzer._last_day_of_month(y, m)
        return date(y, m, min(d.day, last_day))

    @staticmethod
    def _last_day_of_month(year: int, month: int) -> int:
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        return (next_month - timedelta(days=1)).day
