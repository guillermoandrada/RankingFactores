"""Long/short portfolio construction."""

from __future__ import annotations

import math

from modules.portfolio.models import PortfolioDiagnostics, SecurityCandidate, TargetPosition


def _side_weights(
    selected: list[SecurityCandidate],
    total_weight: float,
    weighting: str,
) -> dict[str, float]:
    if not selected or total_weight <= 0:
        return {}
    if weighting == "score":
        min_score = min(item.score for item in selected)
        signals = {
            item.ticker: float(item.score - min_score + 1e-6)
            for item in selected
        }
        denom = sum(signals.values()) or 1.0
        return {ticker: total_weight * value / denom for ticker, value in signals.items()}
    equal = total_weight / len(selected)
    return {item.ticker: equal for item in selected}


def build_long_short_portfolio(
    candidates: list[SecurityCandidate],
    *,
    bucket_count: int = 10,
    long_bucket_count: int = 1,
    short_bucket_count: int = 1,
    weighting: str = "equal",
    gross_exposure: float = 1.0,
    net_exposure: float = 0.0,
) -> tuple[list[TargetPosition], PortfolioDiagnostics]:
    """Build a long/short portfolio from score-ranked buckets."""
    diagnostics = PortfolioDiagnostics()
    eligible = [candidate for candidate in candidates if candidate.is_allowed]
    for candidate in candidates:
        if not candidate.is_allowed:
            diagnostics.excluded.append({
                "ticker": candidate.ticker,
                "reason": "blocked_by_filter",
            })
    eligible = sorted(eligible, key=lambda item: item.score, reverse=True)
    if not eligible:
        return [], diagnostics

    bucket_count = max(2, int(bucket_count))
    long_bucket_count = max(1, int(long_bucket_count))
    short_bucket_count = max(1, int(short_bucket_count))
    bucket_size = max(1, math.ceil(len(eligible) / bucket_count))

    long_selection = eligible[: bucket_size * long_bucket_count]
    short_selection = eligible[-bucket_size * short_bucket_count :]

    long_total = max(0.0, (gross_exposure + net_exposure) / 2.0)
    short_total = max(0.0, (gross_exposure - net_exposure) / 2.0)

    long_weights = _side_weights(long_selection, long_total, weighting)
    short_weights = _side_weights(short_selection, short_total, weighting)

    combined: dict[str, float] = {}
    for ticker, weight in long_weights.items():
        combined[ticker] = combined.get(ticker, 0.0) + weight
    for ticker, weight in short_weights.items():
        combined[ticker] = combined.get(ticker, 0.0) - weight

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
        for ticker, weight in combined.items()
        if abs(weight) > 0
    ]
    diagnostics.notes.append(
        f"Long/short built with {bucket_count} buckets, weighting={weighting}."
    )
    return positions, diagnostics

