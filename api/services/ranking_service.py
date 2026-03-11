from __future__ import annotations

import io

import pandas as pd

from api.dependencies import get_db, get_profile_resolver
from modules.analytics import FactorScoringService, RankingEngine, ZScoreCalculator


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

    from api.dependencies import get_derived_store
    calculator = ZScoreCalculator(engine=db.engine, derived_store=get_derived_store())
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
    rename_map: dict[str, str] = {}
    for col in df_ranked.columns:
        new_name = col
        if new_name.endswith("_zscore"):
            new_name = new_name[: -len("_zscore")] + " Score"
        # Capitalize first character only; keep rest as-is
        if new_name:
            new_name = new_name[0].upper() + new_name[1:]
        rename_map[col] = new_name
    df_ranked = df_ranked.rename(columns=rename_map)

    return df_ranked


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
