from __future__ import annotations

import io

import pandas as pd

from api.dependencies import (
    get_db,
    get_derived_store,
    get_profile_resolver,
    get_zscore_calculator,
)
from modules.domain.analytics import FactorScoringService, RankingEngine
from modules.domain.analytics.metric_loader import fetch_metric_matrix


def get_metric_names_from_profile(profile: dict) -> list[str]:
    """Extract metric names from factors in a profile."""
    names: list[str] = []
    for factor in profile.get("factors", []):
        names.extend(list((factor.get("weights") or {}).keys()))
    unique: list[str] = []
    for name in names:
        if name not in unique:
            unique.append(name)
    return unique


def _apply_display_labels(df_ranked: pd.DataFrame) -> pd.DataFrame:
    """Apply user-facing column labels and reject duplicate display names."""
    rename_map: dict[str, str] = {}
    for col in df_ranked.columns:
        new_name = col
        if new_name.endswith("_zscore"):
            new_name = new_name[: -len("_zscore")] + " Score"
        if new_name:
            new_name = new_name[0].upper() + new_name[1:]
        rename_map[col] = new_name

    renamed = df_ranked.rename(columns=rename_map)
    duplicate_columns = renamed.columns[renamed.columns.duplicated()].tolist()
    if duplicate_columns:
        duplicates = ", ".join(sorted(set(str(col) for col in duplicate_columns)))
        raise ValueError(
            "Ranking output contains duplicate display columns: "
            f"{duplicates}. Rename the conflicting profile node or metric."
        )
    return renamed


def compute_ranking(
    *,
    quarter: str,
    industry: str = "",
    sector: str = "",
    index: str = "",
    scoring_profile: str,
) -> pd.DataFrame:
    db = get_db()
    resolver = get_profile_resolver()

    industry_filter = industry.strip() or None
    sector_filter = sector.strip() or None
    index_filter = index.strip() or None

    resolved_profile = resolver.resolve(
        scoring_profile=scoring_profile,
        industry=industry_filter,
        sector=sector_filter,
    )

    metric_names = get_metric_names_from_profile(resolved_profile)
    if not metric_names:
        raise ValueError("No metric weights configured.")

    calculator = get_zscore_calculator()
    ranking_engine = RankingEngine(zsuffix="_zscore")
    factor_service = FactorScoringService(
        calculator=calculator,
        ranking_engine=ranking_engine,
    )

    df_ranked = factor_service.run(
        period=quarter,
        metric_names=metric_names,
        index=index_filter,
        weights=resolved_profile.get("weights"),
        method=resolved_profile.get("method"),
        industry=industry_filter,
        sector=sector_filter,
        profile=resolved_profile,
    )
    warnings = list(df_ranked.attrs.get("warnings", []))
    df_ranked = df_ranked.sort_values("scoring", ascending=False).reset_index()
    if "security_id" in df_ranked.columns:
        df_ranked = df_ranked.drop(columns=["security_id"])
    if "long_name" in df_ranked.columns:
        df_ranked = df_ranked.rename(columns={"long_name": "name"})

    # Order ticker and name first when present
    prioritised = [c for c in ("ticker", "name") if c in df_ranked.columns]
    other = [c for c in df_ranked.columns if c not in prioritised]
    df_ranked = df_ranked[prioritised + other]

    # Human-friendly column labels:
    # - Replace '_zscore' suffix with ' Score'
    # - Capitalize the first letter of each column name
    df_ranked = _apply_display_labels(df_ranked)
    df_ranked.attrs["warnings"] = warnings
    return df_ranked


def compute_metric_coverage(
    *,
    quarter: str,
    industry: str = "",
    sector: str = "",
    index: str = "",
    scoring_profile: str,
) -> dict:
    """
    Raw missing-value counts for every metric a profile uses in one period.

    Covers the profile's metrics and the base metrics their derived formulas read,
    over the same universe the ranking uses, before any NA handling.
    """
    industry_filter = industry.strip() or None
    sector_filter = sector.strip() or None
    resolved_profile = get_profile_resolver().resolve(
        scoring_profile=scoring_profile,
        industry=industry_filter,
        sector=sector_filter,
    )
    metric_names = get_metric_names_from_profile(resolved_profile)
    if not metric_names:
        return {"universe_size": 0, "missing_counts": {}}
    df_wide, _ = fetch_metric_matrix(
        engine=get_db().engine,
        period=quarter,
        metric_names=metric_names,
        derived_store=get_derived_store(),
        index_name=index.strip() or None,
        industry_name=industry_filter,
        sector_name=sector_filter,
    )
    return {
        "universe_size": int(df_wide.attrs.get("universe_size", 0)),
        "missing_counts": dict(df_wide.attrs.get("missing_counts", {})),
    }


def compute_ranking_for_profile(
    *,
    quarter: str,
    industry: str = "",
    sector: str = "",
    index: str = "",
    scoring_profile: str,
) -> pd.DataFrame:
    return compute_ranking(
        quarter=quarter,
        industry=industry,
        sector=sector,
        index=index,
        scoring_profile=scoring_profile,
    )


def export_ranking_to_xlsx(
    df: pd.DataFrame,
    period: str,
    scope: str = "ALL",
) -> tuple[bytes, str]:
    """
    Export ranking DataFrame to XLSX bytes.
    Returns (xlsx_bytes, suggested_filename).
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Ranking")
    buffer.seek(0)
    period_safe = period.replace(" ", "").replace("/", "-")
    scope_safe = scope.replace(" ", "_")
    filename = f"Ranking_{period_safe}_{scope_safe}.xlsx"
    return buffer.getvalue(), filename
