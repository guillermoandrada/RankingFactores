"""Legacy-style rebalance portfolio construction."""

from __future__ import annotations

from collections import defaultdict
import math

from modules.portfolio.models import PortfolioDiagnostics, SecurityCandidate, TargetPosition


def _ranked_by_group(
    candidates: list[SecurityCandidate],
    group_name: str,
) -> dict[str, list[SecurityCandidate]]:
    grouped: dict[str, list[SecurityCandidate]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        key = getattr(candidate, group_name, "") or ""
        if key:
            grouped[key].append(candidate)
    return dict(grouped)


def build_legacy_rebalance_target(
    candidates: list[SecurityCandidate],
    *,
    max_position: float = 0.05,
    neutral_position: float = 0.03,
    score_quantile_cutoff: float = 0.5,
    sector_targets: dict[str, float] | None = None,
    industry_targets: dict[str, float] | None = None,
) -> tuple[list[TargetPosition], PortfolioDiagnostics]:
    """
    Build a target portfolio using the spirit of the legacy model.

    The first release prioritizes explicit industry targets when provided.
    When only sector targets are supplied, it allocates within sector.
    """
    diagnostics = PortfolioDiagnostics()
    eligible = [candidate for candidate in candidates if candidate.is_allowed]
    for candidate in candidates:
        if not candidate.is_allowed:
            diagnostics.excluded.append({
                "ticker": candidate.ticker,
                "reason": "blocked_by_filter",
            })

    if not eligible:
        return [], diagnostics

    if 0.0 <= score_quantile_cutoff <= 1.0:
        industry_scores: dict[str, list[float]] = defaultdict(list)
        for candidate in eligible:
            if candidate.industry:
                industry_scores[candidate.industry].append(candidate.score)

        def _quantile(values: list[float], q: float) -> float:
            values = sorted(values)
            if not values:
                return float("-inf")
            idx = (len(values) - 1) * q
            low = math.floor(idx)
            high = math.ceil(idx)
            if low == high:
                return values[low]
            return values[low] + (values[high] - values[low]) * (idx - low)

        industry_cutoffs = {
            industry: _quantile(scores, score_quantile_cutoff)
            for industry, scores in industry_scores.items()
        }
        filtered = []
        for candidate in eligible:
            cutoff = industry_cutoffs.get(candidate.industry, float("-inf"))
            if candidate.score >= cutoff:
                filtered.append(candidate)
            else:
                diagnostics.excluded.append({
                    "ticker": candidate.ticker,
                    "reason": f"below_industry_quantile_{score_quantile_cutoff:.2f}",
                })
        eligible = filtered

    group_name = "industry" if industry_targets else "sector"
    group_targets = industry_targets or sector_targets or {}
    ranked = _ranked_by_group(eligible, group_name)

    target_weights: dict[str, float] = defaultdict(float)
    for group_key, target_weight in group_targets.items():
        remaining = float(target_weight)
        for candidate in ranked.get(group_key, []):
            if remaining <= 0:
                break
            allocation = min(neutral_position, max_position, remaining)
            if allocation <= 0:
                continue
            target_weights[candidate.ticker] += allocation
            remaining -= allocation
        if remaining > 0:
            diagnostics.constraints.append({
                "group_type": group_name,
                "group": group_key,
                "unallocated_weight": round(remaining, 6),
                "reason": "insufficient_eligible_names",
            })

    # If no explicit targets were supplied, fall back to an equal target weight
    # across the highest-scoring eligible names using neutral position sizing.
    if not target_weights:
        remaining = 1.0
        for candidate in sorted(eligible, key=lambda item: item.score, reverse=True):
            allocation = min(neutral_position, max_position, remaining)
            if allocation <= 0:
                break
            target_weights[candidate.ticker] = allocation
            remaining -= allocation
            if remaining <= 0:
                break

    by_ticker = {candidate.ticker: candidate for candidate in eligible}
    positions = [
        TargetPosition(
            ticker=ticker,
            target_weight=weight,
            score=by_ticker[ticker].score,
            name=by_ticker[ticker].name,
            sector=by_ticker[ticker].sector,
            industry=by_ticker[ticker].industry,
        )
        for ticker, weight in sorted(
            target_weights.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if weight > 0
    ]

    if sector_targets and industry_targets:
        diagnostics.notes.append(
            "Industry targets were used for construction; sector targets are returned as diagnostics for consistency checks."
        )
    return positions, diagnostics

