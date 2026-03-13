"""Smart beta portfolio construction."""

from __future__ import annotations

from modules.portfolio.constraints import (
    apply_group_targets,
    cap_security_weights,
    normalize_weights,
)
from modules.portfolio.models import PortfolioDiagnostics, SecurityCandidate, TargetPosition


def build_smart_beta_portfolio(
    candidates: list[SecurityCandidate],
    *,
    top_n: int | None = None,
    max_weight: float = 0.10,
    sector_targets: dict[str, float] | None = None,
    industry_targets: dict[str, float] | None = None,
) -> tuple[list[TargetPosition], PortfolioDiagnostics]:
    """Build a long-only smart beta portfolio from score signals."""
    diagnostics = PortfolioDiagnostics()
    eligible = [candidate for candidate in candidates if candidate.is_allowed]
    for candidate in candidates:
        if not candidate.is_allowed:
            diagnostics.excluded.append({
                "ticker": candidate.ticker,
                "reason": "blocked_by_filter",
            })

    eligible = sorted(eligible, key=lambda item: item.score, reverse=True)
    if top_n is not None and top_n > 0:
        eligible = eligible[:top_n]

    if not eligible:
        return [], diagnostics

    min_score = min(candidate.score for candidate in eligible)
    raw_weights = {
        candidate.ticker: float(candidate.score - min_score + 1e-6)
        for candidate in eligible
    }
    weights = normalize_weights(raw_weights)

    if sector_targets:
        weights = apply_group_targets(eligible, weights, "sector", sector_targets)
    if industry_targets:
        weights = apply_group_targets(eligible, weights, "industry", industry_targets)

    weights = cap_security_weights(weights, max_weight=max_weight)
    weights = normalize_weights(weights)

    positions = [
        TargetPosition(
            ticker=candidate.ticker,
            target_weight=weights.get(candidate.ticker, 0.0),
            score=candidate.score,
            name=candidate.name,
            sector=candidate.sector,
            industry=candidate.industry,
        )
        for candidate in eligible
        if weights.get(candidate.ticker, 0.0) > 0
    ]
    return positions, diagnostics

