"""
Information Coefficient (IC) analyzer using cross-sectional Spearman rank correlation.

Supports multivariate factor selection: predictive Rank IC vs forward returns and
inter-factor Spearman correlation (colinearity) on shared cross-sections per period.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import MetaData, and_, func, select
from sqlalchemy.engine import Engine

from modules.config.derived_metrics import DerivedMetricStore
from modules.domain.analytics.metric_loader import (
    compute_derived_series,
    resolve_metric_dependencies,
)
from modules.infrastructure.db import FinancialDatabase
from modules.infrastructure.market_data.providers.base import BasePriceProvider
from modules.infrastructure.market_data.providers.hybrid_provider import HybridPriceProvider
from modules.infrastructure.market_data.providers.yfinance_provider import YFinancePriceProvider


@dataclass(frozen=True)
class ICPoint:
    period: str
    start_date: str
    end_date: str
    ic: float
    n: int


@dataclass(frozen=True)
class ICSample:
    """One period's usable cross-section: tickers with both a metric value and a return."""

    period: str
    start_date: str
    end_date: str
    frame: pd.DataFrame  # columns: ticker, value, forward_return


class _PeriodSkipped(Exception):
    """A period that yields no IC sample; the message says why, for the skip warning."""


GROUP_BY_OPTIONS = frozenset({"sector", "industry"})


@dataclass(frozen=True)
class DerivedStep:
    """One derived metric in a formula chain: inputs combined left-to-right."""

    name: str
    metric_names: tuple[str, ...]
    operations: tuple[str, ...]


@dataclass(frozen=True)
class MetricPlan:
    """
    How one selected factor is materialised for a period cross-section.

    A base metric resolves to a single DB metric id and no steps. A derived metric
    resolves to every base metric its formula depends on, plus the derived steps in
    dependency order, exactly as fetch_metric_matrix resolves them for rankings.
    """

    name: str
    base_ids: dict[str, int]
    steps: tuple[DerivedStep, ...] = ()

    @property
    def is_derived(self) -> bool:
        return bool(self.steps)


def _summarise_periods(periods: list[str], limit: int = 6) -> str:
    """Comma-separated preview, so a 40-period run cannot flood the warning."""
    shown = ", ".join(periods[:limit])
    return shown if len(periods) <= limit else f"{shown} … (+{len(periods) - limit} more)"


def _skip_warnings(skipped: dict[str, list[str]], total: int) -> list[str]:
    return [
        f"Skipped {len(affected)} of {total} period(s) because "
        f"{reason}: {_summarise_periods(affected)}"
        for reason, affected in skipped.items()
    ]


def _summarise_ic_points(points: list[ICPoint]) -> dict:
    """Time-series mean, volatility, and Information Ratio of a Rank IC series."""
    ic_vals = [p.ic for p in points]
    mean_ic = float(np.mean(ic_vals)) if ic_vals else None
    std_ic = float(np.std(ic_vals, ddof=1)) if len(ic_vals) >= 2 else None
    ir = (mean_ic / std_ic) if (mean_ic is not None and std_ic not in (None, 0.0)) else None
    return {
        "mean_rank_ic": mean_ic,
        "ic_std": std_ic,
        "information_ratio": ir,
        "n_periods": len(points),
    }


class ICAnalyzer:
    """
    Compute Rank IC (Spearman) vs forward returns and inter-factor Spearman matrices.

    - Cross-section at each period: fundamental_values joined with securities -> ticker.
    - Forward return window: [start_date, end_date] derived from period end + publication lag.
    - Robustness: ignores rows with null fundamentals or missing/invalid price data.
    """

    _DEFAULT_PUBLICATION_LAG_DAYS = 45
    MIN_GROUP_SECURITIES = 5

    def __init__(
        self,
        db: Optional[FinancialDatabase] = None,
        *,
        engine: Optional[Engine] = None,
        price_service: Optional[BasePriceProvider] = None,
        derived_store: Optional[DerivedMetricStore] = None,
        publication_lag_days: int = _DEFAULT_PUBLICATION_LAG_DAYS,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if engine is None:
            db = db or FinancialDatabase()
            engine = db.engine
        self._engine = engine
        self._price_service = price_service or self._default_price_provider(db)
        self._derived_store = derived_store or DerivedMetricStore()
        self._publication_lag_days = int(publication_lag_days)
        self._logger = logger or logging.getLogger(__name__)
        # Forward returns per window, shared by every metric in a run. See
        # _compute_forward_returns for why this is the difference between a run
        # finishing and a run timing out.
        self._forward_returns: dict[tuple[str, str], pd.DataFrame] = {}
        self._forward_returns_attempted: dict[tuple[str, str], set[str]] = {}

        self._metadata = MetaData()
        self._metadata.reflect(bind=self._engine)
        self._tbl_fund = self._metadata.tables["fundamental_values"]
        self._tbl_sec = self._metadata.tables["securities"]
        self._tbl_metrics = self._metadata.tables["metrics"]

    @staticmethod
    def _default_price_provider(
        db: Optional[FinancialDatabase],
    ) -> BasePriceProvider:
        """
        Prefer the cache-backed provider so an unwired analyzer still reads
        prices from the database first. Callers that supply only an ``engine``
        have no repository to cache through and fall back to Yahoo directly.
        """
        if db is None:
            return YFinancePriceProvider()
        return HybridPriceProvider(db=db, yf_provider=YFinancePriceProvider())

    def clear_caches(self) -> None:
        """Drop memoised forward returns. Call whenever price data changes."""
        self._forward_returns.clear()
        self._forward_returns_attempted.clear()

    def analyze_multivariate(
        self,
        *,
        metric_names: list[str],
        forward_months: int,
        periods: Optional[Iterable[str]] = None,
        group_by: Optional[str] = None,
    ) -> dict:
        """
        Multivariate factor analysis.

        1) Predictive: for each metric, cross-sectional Spearman (Rank IC) vs forward returns
           per period (same logic as single-metric IC).
        2) Colinearity: per period, Spearman correlation matrix between metrics on the shared
           cross-section (inner join on tickers with all metrics present); matrices are
           averaged across periods.

        Args:
            metric_names: Metric names (at least two distinct names). Derived metrics
                are accepted and computed from their base metrics per period, exactly
                as rankings compute them.
            forward_months: Forward horizon in months.
            periods: Optional filter; if None, periods are taken per metric / intersection
                as described below.
            group_by: Optional "sector" or "industry". When set, Rank IC is also computed
                within each group's cross-section per period and summarised per group.

        Returns:
            {
              "metric_names": list[str],
              "forward_months": int,
              "predictive": list[dict]  # per metric: mean_rank_ic, ic_std, information_ratio, n_periods, series
              "inter_factor_correlation": {"labels": list[str], "matrix": list[list[float|None]]}
              "periods": {
                  "per_metric": list[dict],  # available/used periods per metric
                  "inter_factor": dict,  # available/used shared periods for correlation matrix
              },
              "group_breakdown": None | {"group_by": str, "rows": list[dict]},
              "warnings": list[str],
            }
        """
        forward_months = int(forward_months)
        if forward_months <= 0:
            raise ValueError("forward_months must be positive.")
        if group_by is not None and group_by not in GROUP_BY_OPTIONS:
            raise ValueError("group_by must be 'sector' or 'industry'.")

        unique_names = list(dict.fromkeys(str(n).strip() for n in metric_names if str(n).strip()))
        if len(unique_names) < 2:
            raise ValueError("Select at least two distinct metric names.")

        warnings: list[str] = []
        plans = self._resolve_metric_plans(unique_names, warnings)
        if len(plans) < 2:
            raise ValueError("Could not resolve at least two valid metrics from the database.")

        predictive: list[dict] = []
        periods_per_metric: list[dict] = []
        group_rows: list[dict] = []
        for name in unique_names:
            plan = plans.get(name)
            if plan is None:
                continue
            if periods is None:
                available_periods = self._list_periods_for_plan(plan)
            else:
                available_periods = sorted({p for p in periods if p})
            use_periods = available_periods
            samples, w = self._ic_samples_for_metric(
                plan=plan,
                forward_months=forward_months,
                periods=use_periods,
            )
            points, point_warnings = self._ic_points(samples, total_periods=len(use_periods))
            warnings.extend(f"{name}: {message}" for message in w + point_warnings)
            if group_by is not None:
                group_rows.extend(
                    {"metric_name": name, "group": group, **summary}
                    for group, summary in self._group_ic_summaries(samples, group_by).items()
                )
            used_periods = [p.period for p in points]
            periods_per_metric.append(
                {
                    "metric_name": name,
                    "available_periods": list(available_periods),
                    "used_periods": list(used_periods),
                }
            )
            predictive.append(
                {
                    "metric_name": name,
                    **_summarise_ic_points(points),
                    "available_periods": list(available_periods),
                    "used_periods": list(used_periods),
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

        inter_labels = [n for n in unique_names if n in plans]

        if periods is None:
            period_sets = [set(self._list_periods_for_plan(plans[n])) for n in inter_labels]
            inter_available_periods = sorted(set.intersection(*period_sets) if period_sets else set())
        else:
            inter_available_periods = sorted({p for p in periods if p})

        inter_matrix = self._inter_factor_spearman_matrix(
            plans={n: plans[n] for n in inter_labels},
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
            "periods": {
                "per_metric": periods_per_metric,
                "inter_factor": {
                    "available_periods": inter_available_periods,
                },
            },
            "group_breakdown": (
                None if group_by is None else {"group_by": group_by, "rows": group_rows}
            ),
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
            plan=self._plan_for_metric_id(metric_id),
            forward_months=forward_months,
            periods=periods,
        )

        summary = _summarise_ic_points(points)
        ic_mean, ic_std, ir = (
            summary["mean_rank_ic"], summary["ic_std"], summary["information_ratio"]
        )

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
        plan: MetricPlan,
        forward_months: int,
        periods: Optional[Iterable[str]],
    ) -> tuple[list[ICPoint], list[str]]:
        samples, warnings = self._ic_samples_for_metric(
            plan=plan, forward_months=forward_months, periods=periods
        )
        periods_list = sorted({p for p in (periods or []) if p})
        points, point_warnings = self._ic_points(samples, total_periods=len(periods_list))
        return points, warnings + point_warnings

    def _ic_samples_for_metric(
        self,
        *,
        plan: MetricPlan,
        forward_months: int,
        periods: Optional[Iterable[str]],
    ) -> tuple[list[ICSample], list[str]]:
        """Usable cross-section per period, plus one warning per reason a period was skipped."""
        periods_list = sorted({p for p in (periods or []) if p})
        samples: list[ICSample] = []
        # Grouped by reason rather than reported per period, so a wide run cannot
        # bury the explanation under one warning per period.
        skipped: dict[str, list[str]] = {}
        for period in periods_list:
            try:
                samples.append(
                    self._ic_sample(plan=plan, period=period, forward_months=forward_months)
                )
            except _PeriodSkipped as exc:
                skipped.setdefault(str(exc), []).append(period)
        return samples, _skip_warnings(skipped, len(periods_list))

    def _ic_sample(self, *, plan: MetricPlan, period: str, forward_months: int) -> ICSample:
        """Metric values joined to forward returns for one period. Raises _PeriodSkipped."""
        try:
            start_date, end_date = self._parse_period_to_dates(period, forward_months)
        except Exception as exc:
            self._logger.debug("Skipping period '%s': cannot parse dates (%s)", period, exc)
            raise _PeriodSkipped("the period label could not be parsed as a date") from exc

        fundamentals = self._load_cross_section_for_plan(plan=plan, period=period)
        if fundamentals.empty:
            raise _PeriodSkipped("the metric has no values in that period")

        fundamentals["value"] = pd.to_numeric(fundamentals["value"], errors="coerce")
        fundamentals = fundamentals.dropna(subset=["ticker", "value"])
        if fundamentals.empty:
            raise _PeriodSkipped("the metric has no numeric values in that period")

        returns = self._compute_forward_returns(
            tickers=fundamentals["ticker"].astype(str).tolist(),
            start_date=start_date,
            end_date=end_date,
        )
        if returns.empty:
            raise _PeriodSkipped(
                f"no forward returns were available for {start_date} to {end_date} "
                "(check price coverage)"
            )

        merged = fundamentals.merge(returns, on="ticker", how="inner")
        merged["forward_return"] = pd.to_numeric(merged["forward_return"], errors="coerce")
        merged = merged.dropna(subset=["value", "forward_return"])
        if len(merged) < 2:
            raise _PeriodSkipped("fewer than two securities had both a metric value and a return")
        return ICSample(period=period, start_date=start_date, end_date=end_date, frame=merged)

    def _ic_points(
        self, samples: list[ICSample], *, total_periods: int
    ) -> tuple[list[ICPoint], list[str]]:
        """Rank IC per sample; samples whose correlation is undefined are reported."""
        points: list[ICPoint] = []
        undefined: list[str] = []
        for sample in samples:
            point = self._ic_point(sample, sample.frame)
            if point is None:
                undefined.append(sample.period)
            else:
                points.append(point)
        skipped = {"the rank correlation was undefined (no variation)": undefined} if undefined else {}
        return points, _skip_warnings(skipped, total_periods)

    def _ic_point(self, sample: ICSample, frame: pd.DataFrame) -> Optional[ICPoint]:
        ic_val = self._spearman_ic(frame["value"].to_numpy(), frame["forward_return"].to_numpy())
        if ic_val is None:
            return None
        return ICPoint(
            period=sample.period,
            start_date=sample.start_date,
            end_date=sample.end_date,
            ic=float(ic_val),
            n=int(len(frame)),
        )

    def _group_ic_summaries(
        self, samples: list[ICSample], group_by: str
    ) -> dict[str, dict]:
        """
        Rank IC within each sector/industry cross-section, summarised per group.

        Groups with fewer than MIN_GROUP_SECURITIES securities in a period are left out
        of that period: a rank correlation over a handful of names is mostly noise.
        """
        points_by_group: dict[str, list[ICPoint]] = {}
        for sample in samples:
            labels = self._load_security_groups(period=sample.period, group_by=group_by)
            frame = sample.frame.merge(labels, on="ticker", how="left")
            frame["group"] = frame["group"].fillna("Unknown")
            for group, sub in frame.groupby("group", sort=True):
                if len(sub) < self.MIN_GROUP_SECURITIES:
                    continue
                point = self._ic_point(sample, sub)
                if point is not None:
                    points_by_group.setdefault(str(group), []).append(point)
        return {
            group: {
                **_summarise_ic_points(points),
                "mean_n": float(np.mean([p.n for p in points])),
            }
            for group, points in sorted(points_by_group.items())
        }

    def _load_security_groups(self, *, period: str, group_by: str) -> pd.DataFrame:
        """Ticker -> sector/industry for a period: period classification first, then security."""
        tbl_s = self._tbl_sec
        tbl_c = self._metadata.tables["security_classification"]
        if group_by == "sector":
            tbl_label = self._metadata.tables["sectors"]
            label_col = tbl_label.c.sector_name
            join_label = tbl_label.c.sector_id == func.coalesce(tbl_c.c.sector_id, tbl_s.c.sector_id)
        else:
            tbl_label = self._metadata.tables["industries"]
            label_col = tbl_label.c.industry_name
            join_label = tbl_label.c.industry_id == func.coalesce(
                tbl_c.c.industry_id, tbl_s.c.industry_id
            )
        q = (
            select(tbl_s.c.ticker.label("ticker"), label_col.label("group"))
            .select_from(tbl_s)
            .outerjoin(tbl_c, and_(tbl_c.c.security_id == tbl_s.c.id, tbl_c.c.period == period))
            .outerjoin(tbl_label, join_label)
        )
        with self._engine.connect() as conn:
            return pd.read_sql_query(q, con=conn).drop_duplicates("ticker")

    def _inter_factor_spearman_matrix(
        self,
        *,
        plans: dict[str, MetricPlan],
        periods: Optional[Iterable[str]],
        warnings: list[str],
    ) -> list[list[Optional[float]]]:
        """Average cross-sectional Spearman correlation matrices across periods (inner join on tickers)."""
        labels = list(plans.keys())
        if len(labels) < 2:
            return []

        if periods is None:
            period_sets = [set(self._list_periods_for_plan(plan)) for plan in plans.values()]
            common = set.intersection(*period_sets) if period_sets else set()
            period_list = sorted(common)
        else:
            period_list = sorted({p for p in periods if p})

        if not period_list:
            warnings.append("No shared periods found for inter-factor correlation.")
            return [[None] * len(labels) for _ in labels]

        matrices: list[np.ndarray] = []
        for period in period_list:
            wide = self._load_wide_cross_section(plans=plans, period=period)
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

    def _resolve_metric_plans(
        self,
        names: list[str],
        warnings: list[str],
    ) -> dict[str, MetricPlan]:
        """
        Turn selected metric names into cross-section plans.

        Base metrics map to their DB id; derived metrics map to every base metric their
        formula depends on plus the derived steps, so a formula is analysed with the
        same definition rankings and backtests use. Names that cannot be resolved are
        reported as warnings and dropped, as unknown DB metrics always were.
        """
        db_ids = self._db_metric_ids()
        formulas = self._derived_store.list_formulas()
        base_names = set(db_ids)

        plans: dict[str, MetricPlan] = {}
        for name in names:
            try:
                order = resolve_metric_dependencies(name, formulas, base_names)
            except ValueError as exc:
                warnings.append(f"Metric not usable: {exc}")
                continue
            base_ids = {dep: db_ids[dep] for dep in order if dep in base_names}
            if not base_ids:
                warnings.append(f"Metric not found in database: '{name}'.")
                continue
            steps = tuple(
                DerivedStep(
                    name=dep,
                    metric_names=tuple(formulas[dep].get("metric_names", [])),
                    operations=tuple(formulas[dep].get("operations", [])),
                )
                for dep in order
                if dep not in base_names
            )
            plans[name] = MetricPlan(name=name, base_ids=base_ids, steps=steps)
        return plans

    def _plan_for_metric_id(self, metric_id: int) -> MetricPlan:
        """Plan for a single DB metric, for the id-based single-metric entry point."""
        return MetricPlan(name=str(metric_id), base_ids={str(metric_id): int(metric_id)})

    def _db_metric_ids(self) -> dict[str, int]:
        tbl = self._tbl_metrics
        with self._engine.connect() as conn:
            rows = conn.execute(select(tbl.c.metric_name, tbl.c.metric_id)).fetchall()
        return {str(row[0]): int(row[1]) for row in rows if row[0] is not None}

    def _load_wide_cross_section(self, *, plans: dict[str, MetricPlan], period: str) -> pd.DataFrame:
        """Inner join all metrics on ticker; columns = metric names."""
        merged: Optional[pd.DataFrame] = None
        for name, plan in plans.items():
            df = self._load_cross_section_for_plan(plan=plan, period=period)
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

    def _list_periods_for_plan(self, plan: MetricPlan) -> list[str]:
        """
        Periods a plan can be evaluated in.

        A derived metric only exists where every base metric it depends on has values,
        so its availability is the intersection of its dependencies'.
        """
        period_sets = [set(self._list_periods_for_metric(mid)) for mid in plan.base_ids.values()]
        return sorted(set.intersection(*period_sets)) if period_sets else []

    def _load_cross_section_for_plan(self, *, plan: MetricPlan, period: str) -> pd.DataFrame:
        """
        Ticker/value cross-section for one metric, derived formulas included.

        Base metrics are read straight from fundamental_values. Derived metrics are
        computed from their base metrics on the shared cross-section, applying the
        formula steps in dependency order. NA handling is deliberately not applied
        here: IC drops non-numeric rows anyway, and filling them would change the
        ranks the correlation is measured on.
        """
        if not plan.is_derived:
            (metric_id,) = plan.base_ids.values()
            return self._load_cross_section(metric_id=metric_id, period=period)

        wide: Optional[pd.DataFrame] = None
        for base_name, metric_id in plan.base_ids.items():
            df = self._load_cross_section(metric_id=metric_id, period=period)
            if df.empty:
                return pd.DataFrame(columns=["ticker", "value"])
            df = df[["ticker", "value"]].rename(columns={"value": base_name})
            df[base_name] = pd.to_numeric(df[base_name], errors="coerce")
            merged = df if wide is None else wide.merge(df, on="ticker", how="inner")
            wide = merged
        if wide is None or wide.empty:
            return pd.DataFrame(columns=["ticker", "value"])

        for step in plan.steps:
            wide[step.name] = compute_derived_series(
                wide, list(step.metric_names), list(step.operations)
            )
        values = pd.to_numeric(wide[plan.name], errors="coerce")
        return pd.DataFrame({"ticker": wide["ticker"], "value": values}).dropna(subset=["value"])

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
        """
        Forward returns for one window, memoised per window across metrics.

        Every metric in a run shares the same forward windows, so without this the same
        price fetch is repeated once per metric per period. That is the dominant cost of
        a run whenever prices are not cached locally: the provider is hit N times for
        identical data, and a rate-limited or missing ticker pays its full retry cost
        every time. Tickers are remembered as attempted even when they returned nothing,
        so a gap is not re-fetched either.
        """
        clean_tickers = [t for t in (str(ticker or "").strip() for ticker in tickers) if t]
        if not clean_tickers:
            return pd.DataFrame(columns=["ticker", "forward_return"])

        window = (start_date, end_date)
        attempted = self._forward_returns_attempted.setdefault(window, set())
        pending = [t for t in dict.fromkeys(clean_tickers) if t not in attempted]
        if pending:
            fetched = self._fetch_forward_returns(
                tickers=pending, start_date=start_date, end_date=end_date
            )
            attempted.update(pending)
            known = self._forward_returns.get(window)
            self._forward_returns[window] = (
                fetched
                if known is None or known.empty
                else pd.concat([known, fetched], ignore_index=True)
            )

        known = self._forward_returns.get(window)
        if known is None or known.empty:
            return pd.DataFrame(columns=["ticker", "forward_return"])
        wanted = set(clean_tickers)
        return known[known["ticker"].isin(wanted)].reset_index(drop=True)

    def _fetch_forward_returns(
        self,
        *,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Geometric return per ticker over the window, straight from the price provider."""
        clean_tickers = [t for t in (str(ticker or "").strip() for ticker in tickers) if t]
        if not clean_tickers:
            return pd.DataFrame(columns=["ticker", "forward_return"])

        result = self._price_service.fetch_price_matrix(
            clean_tickers,
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
        )

        if result.prices.empty:
            return pd.DataFrame(columns=["ticker", "forward_return"])

        rows: list[dict] = []
        for ticker in clean_tickers:
            if ticker not in result.prices.columns:
                continue
            series = result.prices[ticker].dropna()
            if len(series) < 2:
                continue
            try:
                geom = float(series.iloc[-1] / series.iloc[0]) - 1.0
            except Exception as exc:
                self._logger.debug("Return computation failed for %s: %s", ticker, exc)
                continue
            if np.isnan(geom) or np.isinf(geom):
                continue
            rows.append({"ticker": ticker, "forward_return": geom})

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
        text = str(period or "").strip()

        # Accept full date strings: YYYY/MM/DD or YYYY-MM-DD (used when periods
        # are stored as end-of-period dates rather than quarter labels).
        normalized = text.replace("/", "-")
        try:
            return datetime.strptime(normalized, "%Y-%m-%d").date()
        except ValueError:
            pass

        text = text.upper().replace("-", " ").replace("_", " ")
        text = " ".join(text.split())
        if "Q" not in text:
            raise ValueError(
                "Period must be a date (YYYY/MM/DD, YYYY-MM-DD) or a quarter label (e.g. '2024 Q4')."
            )

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
