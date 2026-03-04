"""Load metric matrix from DB + compute derived metrics on the fly."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from modules.config.derived_metrics import DerivedMetricStore

# NA handling options: replace_with_zero, replace_with_high, replace_with_low, eliminate
NA_HANDLING_OPTIONS = frozenset({
    "replace_with_zero",
    "replace_with_high",
    "replace_with_low",
    "eliminate",
})


def _apply_na_handling(
    df: pd.DataFrame,
    na_handling_map: dict[str, str | None],
    metric_names: list[str],
) -> pd.DataFrame:
    """Apply per-metric NA handling. Eliminate drops rows; others fill NA."""
    out = df.copy()
    eliminate_cols = [
        name for name in metric_names
        if name in out.columns and na_handling_map.get(name) == "eliminate"
    ]
    if eliminate_cols:
        out = out.dropna(subset=eliminate_cols)

    for col in metric_names:
        if col not in out.columns:
            continue
        handling = na_handling_map.get(col)
        if not handling or handling == "eliminate":
            continue
        ser = out[col]
        if handling == "replace_with_zero":
            out = out.assign(**{col: ser.fillna(0.0)})
        elif handling == "replace_with_high":
            m = ser.max()
            fill_val = 0.0 if pd.isna(m) else float(m)
            out = out.assign(**{col: ser.fillna(fill_val)})
        elif handling == "replace_with_low":
            m = ser.min()
            fill_val = 0.0 if pd.isna(m) else float(m)
            out = out.assign(**{col: ser.fillna(fill_val)})
    return out


def _compute_derived(
    df: pd.DataFrame,
    metric_names: list[str],
    operations: list[str],
) -> pd.Series:
    """Compute derived metric from existing columns. Operations applied left-to-right."""
    result = pd.to_numeric(df[metric_names[0]], errors="coerce")
    for i, op in enumerate(operations):
        right = pd.to_numeric(df[metric_names[i + 1]], errors="coerce")
        op = str(op).strip()
        if op == "+":
            result = result + right
        elif op == "-":
            result = result - right
        elif op == "*":
            result = result * right
        else:
            result = result / right.replace(0, pd.NA)
    return result


def _resolve_dependencies(
    metric_name: str,
    formulas: dict[str, dict],
    base_names: set[str],
    visited: set[str],
    order: list[str],
) -> None:
    """Append metric names in dependency order to order (mutates order and visited)."""
    if metric_name in base_names:
        if metric_name not in order:
            order.append(metric_name)
        return
    if metric_name in visited:
        raise ValueError(f"Circular dependency in derived metric '{metric_name}'.")
    visited.add(metric_name)
    formula = formulas.get(metric_name)
    if not formula:
        raise ValueError(f"Metric '{metric_name}' not found (not in DB nor derived formulas).")
    for dep in formula.get("metric_names", []):
        _resolve_dependencies(dep, formulas, base_names, visited, order)
    order.append(metric_name)


def fetch_metric_matrix(
    engine: Engine,
    period: str,
    metric_names: list[str],
    derived_store: DerivedMetricStore,
    index_name: Optional[str] = None,
    industry_name: Optional[str] = None,
    sector_name: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """
    Fetch base metrics from DB, compute derived on the fly.
    Returns (wide DataFrame with metric columns, direction_map).
    """
    if not metric_names:
        raise ValueError("metric_names cannot be empty.")

    db_metrics = _get_db_metric_names(engine)
    formulas = derived_store.list_formulas()
    base_names = set(db_metrics.keys())

    # Resolve all metrics we need (base + derived) in dependency order
    all_to_fetch: list[str] = []
    for name in metric_names:
        _resolve_dependencies(name, formulas, base_names, set(), all_to_fetch)
    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for n in all_to_fetch:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    all_to_fetch = unique

    base_to_fetch = [n for n in all_to_fetch if n in base_names]
    if not base_to_fetch:
        raise ValueError("No base metrics to fetch. All requested metrics are derived.")

    metric_ids = _get_metric_ids(engine, base_to_fetch)
    df_long = _fetch_fundamentals(
        engine=engine,
        period=period,
        metric_ids=metric_ids,
        index_name=index_name,
        industry_name=industry_name,
        sector_name=sector_name,
    )
    if df_long.empty:
        raise ValueError("No data found for the given period, metrics, and filters.")

    direction_map = _build_direction_map(df_long, formulas)

    df_wide = df_long.pivot_table(
        index=["security_id", "ticker", "long_name"],
        columns="metric_name",
        values="value",
    ).sort_index()
    df_wide.columns = [str(c) for c in df_wide.columns]

    # Compute derived metrics in dependency order
    for name in all_to_fetch:
        if name in base_names:
            continue
        formula = formulas.get(name)
        if not formula:
            continue
        m_names = formula["metric_names"]
        ops = formula["operations"]
        df_wide[name] = _compute_derived(df_wide, m_names, ops)
        hib = formula.get("higher_is_better")
        direction_map[name] = bool(hib) if hib is not None else True

    # Build na_handling map (base from DB, derived from formulas)
    na_handling_map: dict[str, str | None] = {}
    base_na = _get_base_na_handling(engine, base_to_fetch)
    na_handling_map.update(base_na)
    for name in all_to_fetch:
        if name in base_names:
            continue
        formula = formulas.get(name)
        if not formula:
            continue
        val = formula.get("na_handling")
        val = str(val).strip() if val else None
        if val in NA_HANDLING_OPTIONS:
            na_handling_map[name] = val

    # Apply per-metric NA handling (eliminate drops rows, others fill)
    df_wide = _apply_na_handling(df_wide, na_handling_map, all_to_fetch)

    return df_wide, direction_map


def _get_db_metric_names(engine: Engine) -> dict[str, int]:
    """Return {metric_name: metric_id} for all DB metrics."""
    sql = text("SELECT metric_id, metric_name FROM metrics")
    df = pd.read_sql_query(sql, con=engine)
    return dict(zip(df["metric_name"].astype(str), df["metric_id"]))


def _get_base_na_handling(engine: Engine, metric_names: list[str]) -> dict[str, str | None]:
    """Return {metric_name: na_handling} for base metrics. Only includes valid options."""
    if not metric_names:
        return {}
    placeholders = ", ".join(f":n{i}" for i in range(len(metric_names)))
    sql = text(
        f'SELECT metric_name, "n/a treatment" AS na_handling '
        f"FROM metrics WHERE metric_name IN ({placeholders})"
    )
    params = {f"n{i}": n for i, n in enumerate(metric_names)}
    df = pd.read_sql_query(sql, con=engine, params=params)
    result: dict[str, str | None] = {}
    for _, row in df.iterrows():
        name = str(row["metric_name"])
        val = row.get("na_handling")
        val = str(val).strip() if val else None
        result[name] = val if val in NA_HANDLING_OPTIONS else None
    return result


def _get_metric_ids(engine: Engine, metric_names: list[str]) -> dict[str, int]:
    """Resolve metric names to ids. Raises if any not found."""
    all_db = _get_db_metric_names(engine)
    result = {}
    for name in metric_names:
        if name not in all_db:
            raise ValueError(
                f"Metric '{name}' not found in database. "
                "Ensure the metric exists (import data first)."
            )
        result[name] = all_db[name]
    return result


def _fetch_fundamentals(
    engine: Engine,
    period: str,
    metric_ids: dict[str, int],
    index_name: Optional[str],
    industry_name: Optional[str],
    sector_name: Optional[str],
) -> pd.DataFrame:
    ids = list(metric_ids.values())
    placeholders = ", ".join(f":m{i}" for i in range(len(ids)))
    joins = [
        "JOIN securities s ON s.id = fv.security_id",
        "JOIN metrics m ON m.metric_id = fv.metric_id",
        "LEFT JOIN security_classification sc ON sc.security_id = fv.security_id AND sc.period = :as_of_date",
        "LEFT JOIN sectors sec ON sec.sector_id = COALESCE(sc.sector_id, s.sector_id)",
        "LEFT JOIN industries ind_base ON ind_base.industry_id = COALESCE(sc.industry_id, s.industry_id)",
    ]
    where = [
        "fv.period = :as_of_date",
        f"fv.metric_id IN ({placeholders})",
    ]
    params: dict[str, Any] = {"as_of_date": period}
    params.update({f"m{i}": mid for i, mid in enumerate(ids)})

    if index_name:
        joins.append(
            "JOIN index_membership im "
            "ON im.security_id = fv.security_id "
            "AND im.period = :as_of_date"
        )
        joins.append("JOIN indices idx ON idx.index_id = im.index_id")
        where.append("idx.name = :index_name")
        params["index_name"] = index_name

    if industry_name:
        where.append("ind_base.industry_name = :industry_name")
        params["industry_name"] = industry_name

    if sector_name:
        where.append("sec.sector_name = :sector_name")
        params["sector_name"] = sector_name

    sql = text(
        f"""
        SELECT
            fv.security_id,
            s.ticker,
            s.long_name,
            m.metric_name AS metric_name,
            m.higher_is_better AS higher_is_better,
            fv.value
        FROM fundamental_values fv
        {' '.join(joins)}
        WHERE {' AND '.join(where)}
        """
    )
    return pd.read_sql_query(sql, con=engine, params=params)


def _build_direction_map(
    df_long: pd.DataFrame,
    formulas: dict[str, dict],
) -> dict[str, bool]:
    """Build higher_is_better map from DB and formulas."""
    result: dict[str, bool] = {}
    if "metric_name" in df_long.columns and "higher_is_better" in df_long.columns:
        for _, row in df_long[["metric_name", "higher_is_better"]].drop_duplicates("metric_name").iterrows():
            name = str(row["metric_name"])
            hib = row["higher_is_better"]
            result[name] = bool(hib) if hib is not None else True
    for name, formula in formulas.items():
        if name not in result:
            hib = formula.get("higher_is_better")
            result[name] = bool(hib) if hib is not None else True
    return result
