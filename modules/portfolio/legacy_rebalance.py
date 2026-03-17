"""Legacy-style rebalance portfolio construction."""

from __future__ import annotations

from collections import defaultdict
import math

from modules.portfolio.models import PortfolioDiagnostics, SecurityCandidate, TargetPosition


_WEIGHT_EPSILON = 1e-9


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
    constraint_type: str = "none",
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

    if constraint_type == "industry":
        group_name = "industry"
        group_targets = dict(industry_targets or {})
    elif constraint_type == "sector":
        group_name = "sector"
        group_targets = dict(sector_targets or {})
    else:
        group_name = "sector"
        group_targets = {}
    ranked = _ranked_by_group(eligible, group_name)

    target_weights: dict[str, float] = defaultdict(float)
    restricted_zero_groups = {
        group_key
        for group_key, target_weight in group_targets.items()
        if float(target_weight) <= 0.0
    }
    positive_targets = {
        group_key: float(target_weight)
        for group_key, target_weight in group_targets.items()
        if float(target_weight) > 0.0
    }
    explicitly_specified_groups = set(group_targets)

    for group_key, target_weight in positive_targets.items():
        remaining = float(target_weight)
        for candidate in ranked.get(group_key, []):
            if remaining <= _WEIGHT_EPSILON:
                break
            allocation = min(neutral_position, max_position, remaining)
            if allocation <= _WEIGHT_EPSILON:
                continue
            target_weights[candidate.ticker] += allocation
            remaining -= allocation
        if remaining > _WEIGHT_EPSILON:
            diagnostics.constraints.append({
                "group_type": group_name,
                "group": group_key,
                "unallocated_weight": round(remaining, 6),
                "reason": "insufficient_eligible_names",
            })

    unrestricted_candidates = [
        candidate
        for candidate in sorted(eligible, key=lambda item: item.score, reverse=True)
        if (getattr(candidate, group_name, "") or "") not in explicitly_specified_groups
    ]

    # If explicit targets do not consume the full allocation, fall back only
    # across groups omitted from the target map.
    allocated_weight = sum(target_weights.values())
    remaining = max(0.0, 1.0 - allocated_weight)
    if remaining > _WEIGHT_EPSILON:
        existing_targeted = set(target_weights)
        for candidate in unrestricted_candidates:
            if remaining <= _WEIGHT_EPSILON:
                break
            if candidate.ticker in existing_targeted:
                continue
            allocation = min(neutral_position, max_position, remaining)
            if allocation <= _WEIGHT_EPSILON:
                break
            target_weights[candidate.ticker] += allocation
            remaining -= allocation

    if not target_weights and restricted_zero_groups:
        diagnostics.notes.append(
            f"All eligible {group_name} groups were explicitly restricted to zero; no new buys were allocated."
        )

    if sector_targets and industry_targets:
        diagnostics.notes.append(
            "Only the active constraint_type target map was used for construction."
        )
    elif constraint_type == "none":
        diagnostics.notes.append(
            "No sector or industry restrictions were applied."
        )

    # No additional fallback beyond unrestricted groups when explicit targets exist.
    if not target_weights and not group_targets:
        remaining = 1.0
        for candidate in sorted(eligible, key=lambda item: item.score, reverse=True):
            allocation = min(neutral_position, max_position, remaining)
            if allocation <= _WEIGHT_EPSILON:
                break
            target_weights[candidate.ticker] = allocation
            remaining -= allocation
            if remaining <= _WEIGHT_EPSILON:
                break
    elif remaining > _WEIGHT_EPSILON and group_targets:
        diagnostics.notes.append(
            f"Unallocated weight of {round(remaining, 6)} remains because explicit {group_name} restrictions limited eligible buys."
        )

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
        if weight > _WEIGHT_EPSILON
    ]

    return positions, diagnostics

