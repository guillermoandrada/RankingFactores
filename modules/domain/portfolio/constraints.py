"""Constraint helpers for portfolio strategies."""

from __future__ import annotations

from collections import defaultdict

from modules.domain.portfolio.models import SecurityCandidate, TargetPosition


def compute_group_weights(
    positions: list[TargetPosition],
    group_name: str,
) -> dict[str, float]:
    """Aggregate weights by sector or industry."""
    weights: dict[str, float] = defaultdict(float)
    for position in positions:
        key = getattr(position, group_name, "") or ""
        if not key:
            continue
        weights[key] += float(position.target_weight)
    return {k: round(v, 6) for k, v in weights.items()}


def normalize_weights(
    weights: dict[str, float],
    total: float = 1.0,
) -> dict[str, float]:
    """Normalize positive weights to the requested total."""
    positive_sum = sum(max(0.0, float(v)) for v in weights.values())
    if positive_sum <= 0:
        return {k: 0.0 for k in weights}
    scale = total / positive_sum
    return {k: max(0.0, float(v)) * scale for k, v in weights.items()}


def cap_security_weights(
    weights: dict[str, float],
    max_weight: float,
) -> dict[str, float]:
    """Iteratively cap security weights and redistribute excess."""
    out = dict(weights)
    if max_weight <= 0:
        return {k: 0.0 for k in out}

    while True:
        excess = 0.0
        below_cap: dict[str, float] = {}
        changed = False
        for ticker, value in out.items():
            if value > max_weight:
                excess += value - max_weight
                out[ticker] = max_weight
                changed = True
            else:
                below_cap[ticker] = value
        if not changed or excess <= 0:
            break
        capacity = sum(max_weight - value for value in below_cap.values())
        if capacity <= 0:
            break
        for ticker, value in below_cap.items():
            room = max_weight - value
            out[ticker] += excess * (room / capacity)
    return out


def apply_group_targets(
    candidates: list[SecurityCandidate],
    security_weights: dict[str, float],
    group_name: str,
    targets: dict[str, float],
) -> dict[str, float]:
    """
    Scale security weights inside each group to match target sector/industry weights.
    """
    by_ticker = {candidate.ticker: candidate for candidate in candidates}
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for ticker, weight in security_weights.items():
        candidate = by_ticker.get(ticker)
        if not candidate:
            continue
        group_key = getattr(candidate, group_name, "") or ""
        grouped[group_key][ticker] = float(weight)

    adjusted: dict[str, float] = {ticker: 0.0 for ticker in security_weights}
    for group_key, group_target in targets.items():
        current = grouped.get(group_key, {})
        normalized = normalize_weights(current, total=float(group_target))
        adjusted.update(normalized)
    return adjusted

