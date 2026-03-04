from __future__ import annotations

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
    scoring_profile: str,
) -> pd.DataFrame:
    db = get_db()
    resolver = get_profile_resolver()

    industry_filter = industry.strip() or None
    sector_filter = sector.strip() or None

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
        weights=resolved_profile.get("weights"),
        method=resolved_profile.get("method"),
        industry=industry_filter,
        sector=sector_filter,
        profile=resolved_profile,
    )
    return df_ranked.sort_values("score", ascending=False).reset_index()


def compute_ranking_for_profile(
    *,
    quarter: str,
    industry: str = "",
    sector: str = "",
    scoring_profile: str,
) -> pd.DataFrame:
    return compute_ranking(
        quarter=quarter,
        industry=industry,
        sector=sector,
        scoring_profile=scoring_profile,
    )
