"""Legacy-style rebalance portfolio construction."""

from __future__ import annotations

from collections import defaultdict
import math

from modules.domain.portfolio.models import HoldingPosition, PortfolioDiagnostics, SecurityCandidate, TargetPosition


_WEIGHT_EPSILON = 1e-9
# Total score strictly below this cannot be held; holdings below are sold entirely.
SCORE_SALE_FLOOR = 5.0
# Minimum industry weight deviation from target that triggers a rebalancing trade.
MAX_INDUSTRY_DIFFERENCE = 0.003


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


def _quantile_sorted(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float("-inf")
    idx = (len(values) - 1) * q
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (idx - low)


def industry_quantile_cutoffs_for_allowed(
    allowed_candidates: list[SecurityCandidate],
    score_quantile_cutoff: float,
) -> dict[str, float]:
    """
    Per-industry score at the given quantile, over all allowed (filter-passing) names.
    Used for both eligibility and trade reason tagging.
    """
    if not 0.0 <= score_quantile_cutoff <= 1.0:
        return {}
    industry_scores: dict[str, list[float]] = defaultdict(list)
    for candidate in allowed_candidates:
        if candidate.industry:
            industry_scores[candidate.industry].append(candidate.score)
    return {
        industry: _quantile_sorted(scores, score_quantile_cutoff)
        for industry, scores in industry_scores.items()
    }


def legacy_rebalance_trade_reason(
    *,
    current_weight: float,
    target_weight: float,
    weight_delta: float,
    candidate: SecurityCandidate | None,
    score: float | None,
    industry_quantile_cutoffs: dict[str, float],
    max_position: float,
) -> str:
    """
    Classify a legacy rebalance trade for reporting.

    Sell priority: ethical filter, score < 5, below industry quantile, trim above max
    weight, then generic rebalance. Buys: buy_to_objective.
    """
    if weight_delta > _WEIGHT_EPSILON:
        return "buy_to_objective"
    if weight_delta >= -_WEIGHT_EPSILON:
        return "rebalance_to_target"

    resolved_score = float(score) if score is not None else 0.0
    if candidate is not None and not candidate.is_allowed:
        return "sell_ethical_filter"
    if resolved_score < SCORE_SALE_FLOOR:
        return "sell_score_below_5"

    industry = (candidate.industry if candidate else "") or ""
    cutoff = industry_quantile_cutoffs.get(industry, float("-inf"))
    if resolved_score < cutoff:
        return "sell_below_score_quantile"

    if current_weight > max_position + _WEIGHT_EPSILON:
        return "sell_trim_max_position"

    return "sell_rebalance_to_objective"


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
    Objective long weights for legacy rebalance (ranked allocation).

    Eligibility (in order): pass external filters, total score >= SCORE_SALE_FLOOR (5),
    then at or above the industry score quantile when cutoff is in [0, 1].

    Rebalance trades vs current holdings use these targets; sell reasons are assigned
    in the API layer (ethical filter, score < 5, below quantile, trim above max weight,
    or rebalance to objective; buys use buy_to_objective).
    """
    diagnostics = PortfolioDiagnostics()
    allowed = [candidate for candidate in candidates if candidate.is_allowed]
    for candidate in candidates:
        if not candidate.is_allowed:
            diagnostics.excluded.append({
                "ticker": candidate.ticker,
                "reason": "blocked_by_filter",
            })

    if not allowed:
        return [], diagnostics

    industry_cutoffs = industry_quantile_cutoffs_for_allowed(allowed, score_quantile_cutoff)

    eligible = []
    for candidate in allowed:
        if candidate.score < SCORE_SALE_FLOOR:
            diagnostics.excluded.append({
                "ticker": candidate.ticker,
                "reason": "score_below_5",
            })
            continue
        eligible.append(candidate)

    if not eligible:
        return [], diagnostics

    if industry_cutoffs:
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


def build_legacy_rebalance_with_industry_steps(
    candidates: list[SecurityCandidate],
    current_positions: list[HoldingPosition],
    total_capital: float,
    *,
    industry_targets: dict[str, float],
    max_position: float = 0.05,
    neutral_position: float = 0.03,
    score_quantile_cutoff: float = 0.5,
    max_industry_difference: float = MAX_INDUSTRY_DIFFERENCE,
) -> tuple[list[TargetPosition], PortfolioDiagnostics]:
    """
    Build a rebalancing target applying all 7 spec steps.

    Steps 1–5 determine which securities are eligible (filter + score floor + quantile).
    Step 6 sells the lowest-scoring held positions from overweight industries.
    Step 7 buys the highest-scoring allowed securities into underweight industries.

    Only meaningful when rebalancing an existing portfolio with industry targets supplied.
    """
    diagnostics = PortfolioDiagnostics()

    # Steps 1–3: separate allowed from blocked candidates.
    allowed: list[SecurityCandidate] = []
    for candidate in candidates:
        if candidate.is_allowed:
            allowed.append(candidate)
        else:
            diagnostics.excluded.append({"ticker": candidate.ticker, "reason": "blocked_by_filter"})

    if not allowed:
        return [], diagnostics

    industry_cutoffs = industry_quantile_cutoffs_for_allowed(allowed, score_quantile_cutoff)

    # Step 5 eligibility: score floor then per-industry quantile.
    eligible: list[SecurityCandidate] = []
    for candidate in allowed:
        if candidate.score < SCORE_SALE_FLOOR:
            diagnostics.excluded.append({"ticker": candidate.ticker, "reason": "score_below_5"})
            continue
        eligible.append(candidate)

    if industry_cutoffs:
        filtered: list[SecurityCandidate] = []
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

    eligible_by_ticker = {candidate.ticker: candidate for candidate in eligible}
    allowed_by_ticker = {candidate.ticker: candidate for candidate in allowed}

    # Compute post-Steps-1–5 portfolio weights.
    # Only positions that survive the eligibility filters are kept; overweight ones are
    # capped at neutral_position (Step 4 concentration cap).
    post_filter_weights: dict[str, float] = {}
    for position in current_positions:
        if position.ticker not in eligible_by_ticker:
            continue
        w = position.amount() / total_capital
        post_filter_weights[position.ticker] = neutral_position if w > max_position else w

    def _industry_of(ticker: str) -> str:
        candidate = eligible_by_ticker.get(ticker) or allowed_by_ticker.get(ticker)
        return (candidate.industry if candidate else "") or ""

    def _compute_industry_weights() -> dict[str, float]:
        weights: dict[str, float] = defaultdict(float)
        for ticker, w in post_filter_weights.items():
            ind = _industry_of(ticker)
            if ind:
                weights[ind] += w
        return dict(weights)

    # Step 6: Top-Down Sales — sell lowest-scoring positions from overweight industries.
    industry_weights = _compute_industry_weights()
    for industry, target_w in industry_targets.items():
        current_ind_w = industry_weights.get(industry, 0.0)
        if current_ind_w - target_w <= max_industry_difference:
            continue

        positions_in_industry = sorted(
            [
                (ticker, post_filter_weights[ticker], eligible_by_ticker[ticker].score)
                for ticker in list(post_filter_weights)
                if _industry_of(ticker) == industry
            ],
            key=lambda item: item[2],  # ascending score — worst first
        )

        remaining_overweight = current_ind_w - target_w
        for ticker, w, _score in positions_in_industry:
            if remaining_overweight <= max_industry_difference + 0.0001:
                break
            if w <= round(remaining_overweight, 4):
                post_filter_weights.pop(ticker, None)
                remaining_overweight -= w
            else:
                post_filter_weights[ticker] = max(0.0, w - remaining_overweight)
                remaining_overweight = 0.0

    # Step 7: Bottom-Up Purchases — buy highest-scoring allowed securities into underweight
    # industries. The spec only skips ethics/reuters failures here (not quantile), so we
    # use the `allowed` universe rather than `eligible`.
    industry_weights = _compute_industry_weights()

    allowed_by_industry: dict[str, list[SecurityCandidate]] = defaultdict(list)
    for candidate in allowed:
        if candidate.industry:
            allowed_by_industry[candidate.industry].append(candidate)
    for ind in allowed_by_industry:
        allowed_by_industry[ind].sort(key=lambda c: c.score, reverse=True)

    for industry, target_w in industry_targets.items():
        current_ind_w = industry_weights.get(industry, 0.0)
        if target_w - current_ind_w < max_industry_difference:
            continue

        remaining_underweight = target_w - current_ind_w
        for candidate in allowed_by_industry.get(industry, []):
            if remaining_underweight < max_industry_difference:
                break
            weight_to_buy = min(neutral_position, remaining_underweight)
            existing_w = post_filter_weights.get(candidate.ticker, 0.0)
            weight_to_add = weight_to_buy - existing_w
            if weight_to_add > 0:
                post_filter_weights[candidate.ticker] = existing_w + weight_to_add
                remaining_underweight -= weight_to_add

    # Build TargetPosition list. Use allowed_by_ticker as the source of metadata because
    # Step 7 may have added securities that were excluded by the quantile filter.
    positions = [
        TargetPosition(
            ticker=ticker,
            target_weight=w,
            score=allowed_by_ticker[ticker].score,
            name=allowed_by_ticker[ticker].name,
            sector=allowed_by_ticker[ticker].sector,
            industry=allowed_by_ticker[ticker].industry,
        )
        for ticker, w in post_filter_weights.items()
        if w > _WEIGHT_EPSILON and ticker in allowed_by_ticker
    ]
    positions.sort(key=lambda p: p.target_weight, reverse=True)
    return positions, diagnostics

