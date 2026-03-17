"""Portfolio construction service."""

from __future__ import annotations

from typing import Any

import pandas as pd

from api.schemas.portfolios import PortfolioBuildBody
from api.services.ranking_service import compute_ranking
from modules.db import FinancialDatabase
from modules.portfolio import (
    build_legacy_rebalance_target,
    build_long_short_portfolio,
    build_smart_beta_portfolio,
    parse_ethical_filter_rows,
    parse_holdings_rows,
)
from modules.portfolio.constraints import compute_group_weights
from modules.portfolio.models import (
    HoldingPosition,
    PortfolioDiagnostics,
    PortfolioTrade,
    SecurityCandidate,
    TargetPosition,
)


CASH_TICKER = "CASH_USD"
CASH_NAME = "Cash"


def _first_present(df: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"Required column not found. Expected one of: {names}")


class PortfolioService:
    """Build portfolios from scored universes."""

    def __init__(self, *, db: FinancialDatabase) -> None:
        self._db = db

    def construct_portfolio(
        self,
        period: str,
        request: PortfolioBuildBody,
    ) -> dict[str, Any]:
        df_ranked = compute_ranking(
            quarter=period,
            industry=request.industry,
            sector=request.sector,
            index=request.index,
            scoring_profile=request.scoring_profile,
        )
        candidates = self._build_candidates(period, df_ranked, request)
        target_positions, diagnostics = self._run_strategy(candidates, request)

        current_positions: list[HoldingPosition] = []
        cash = 0.0
        total_capital = float(request.capital_base)
        if request.construction_mode == "rebalance_existing":
            if not request.current_holdings:
                raise ValueError(
                    "current_holdings is required when construction_mode is 'rebalance_existing'."
                )
            current_positions, cash = parse_holdings_rows(
                [row.model_dump() for row in request.current_holdings]
            )
            self._enrich_current_positions(current_positions, candidates, period=period)
            total_capital = cash + sum(position.amount() for position in current_positions)
            if total_capital <= 0:
                raise ValueError("Existing portfolio value must be positive.")

        target_cash_weight = self._target_cash_weight(
            strategy=request.strategy,
            target_positions=target_positions,
        )
        notes = list(diagnostics.notes)
        if target_cash_weight > 0:
            if target_positions:
                notes.append(
                    f"Residual weight of {round(target_cash_weight, 6)} was assigned to cash."
                )
            else:
                notes.append("No target securities were generated; portfolio was allocated to cash.")

        positions_payload = self._build_positions_payload(
            target_positions,
            total_capital=total_capital,
            current_positions=current_positions,
            current_cash=cash,
            target_cash_weight=target_cash_weight,
        )
        current_portfolio_payload = self._build_current_portfolio_payload(
            current_positions,
            total_capital=total_capital,
            cash=cash,
        )
        trades_payload = self._build_trade_payload(
            target_positions,
            current_positions=current_positions,
            total_capital=total_capital,
            min_trade_weight=float(request.min_trade_weight),
            current_cash=cash,
            target_cash_weight=target_cash_weight,
        )
        constraint_diagnostics = self._build_constraint_diagnostics(
            target_positions,
            constraint_type=request.constraint_type,
            sector_targets=request.sector_targets,
            industry_targets=request.industry_targets,
        )
        return {
            "strategy": request.strategy,
            "construction_mode": request.construction_mode,
            "input_scope": {
                "period": period,
                "sector": request.sector.strip() or None,
                "industry": request.industry.strip() or None,
                "index": request.index.strip() or None,
                "scoring_profile": request.scoring_profile,
            },
            "summary": {
                "total_capital": round(total_capital, 6),
                "position_count": len(positions_payload),
                "excluded_count": len(diagnostics.excluded),
                "trade_count": len(trades_payload),
            },
            "portfolio": positions_payload,
            "current_portfolio": current_portfolio_payload,
            "trades": trades_payload,
            "excluded": diagnostics.excluded,
            "constraint_diagnostics": constraint_diagnostics,
            "notes": notes,
            "source_count": len(candidates),
        }

    def _build_candidates(
        self,
        period: str,
        df_ranked: pd.DataFrame,
        request: PortfolioBuildBody,
    ) -> list[SecurityCandidate]:
        ticker_col = _first_present(df_ranked, ["Ticker", "ticker"])
        name_col = next((col for col in ["Name", "name"] if col in df_ranked.columns), "")
        score_col = _first_present(df_ranked, ["Scoring", "scoring", "Score", "score"])

        metadata_rows = self._db.get_security_metadata(
            period,
            tickers=df_ranked[ticker_col].astype(str).tolist(),
        )
        meta_by_ticker = {row["ticker"]: row for row in metadata_rows}
        blocked = parse_ethical_filter_rows(
            [row.model_dump() for row in request.ethical_filter_rows]
        )

        candidates: list[SecurityCandidate] = []
        for _, row in df_ranked.iterrows():
            ticker = str(row[ticker_col]).strip().upper()
            meta = meta_by_ticker.get(ticker, {})
            candidates.append(
                SecurityCandidate(
                    ticker=ticker,
                    score=float(row[score_col]),
                    name=str(row[name_col]) if name_col else str(meta.get("name") or ""),
                    sector=str(meta.get("sector") or ""),
                    industry=str(meta.get("industry") or ""),
                    ethical_allowed=ticker not in blocked,
                )
            )
        return candidates

    def _run_strategy(
        self,
        candidates: list[SecurityCandidate],
        request: PortfolioBuildBody,
    ) -> tuple[list[TargetPosition], PortfolioDiagnostics]:
        if request.strategy == "legacy_rebalance":
            return build_legacy_rebalance_target(
                candidates,
                max_position=float(request.max_position),
                neutral_position=float(request.neutral_position),
                score_quantile_cutoff=float(request.score_quantile_cutoff),
                constraint_type=request.constraint_type,
                sector_targets=request.sector_targets,
                industry_targets=request.industry_targets,
            )
        if request.strategy == "smart_beta":
            return build_smart_beta_portfolio(
                candidates,
                top_n=request.top_n,
                max_weight=float(request.smart_beta_max_weight),
                sector_targets=request.sector_targets,
                industry_targets=request.industry_targets,
            )
        return build_long_short_portfolio(
            candidates,
            bucket_count=int(request.bucket_count),
            long_bucket_count=int(request.long_bucket_count),
            short_bucket_count=int(request.short_bucket_count),
            weighting=request.long_short_weighting,
            gross_exposure=float(request.gross_exposure),
            net_exposure=float(request.net_exposure),
        )

    def _enrich_current_positions(
        self,
        positions: list[HoldingPosition],
        candidates: list[SecurityCandidate],
        *,
        period: str,
    ) -> None:
        candidate_by_ticker = {candidate.ticker: candidate for candidate in candidates}
        metadata = self._db.get_security_metadata(
            period=period,
            tickers=[position.ticker for position in positions],
        )
        meta_by_ticker = {row["ticker"]: row for row in metadata}
        for position in positions:
            candidate = candidate_by_ticker.get(position.ticker)
            meta = meta_by_ticker.get(position.ticker, {})
            position.name = position.name or (candidate.name if candidate else meta.get("name") or "")
            position.sector = candidate.sector if candidate else str(meta.get("sector") or "")
            position.industry = candidate.industry if candidate else str(meta.get("industry") or "")
            position.score = candidate.score if candidate else None

    def _build_positions_payload(
        self,
        target_positions: list[TargetPosition],
        *,
        total_capital: float,
        current_positions: list[HoldingPosition],
        current_cash: float,
        target_cash_weight: float,
    ) -> list[dict[str, Any]]:
        current_weights = self._current_weight_map(current_positions, total_capital)
        payload = []
        for position in sorted(
            target_positions,
            key=lambda item: item.target_weight,
            reverse=True,
        ):
            payload.append({
                "ticker": position.ticker,
                "name": position.name,
                "sector": position.sector,
                "industry": position.industry,
                "score": position.score,
                "current_weight": round(current_weights.get(position.ticker, 0.0), 6),
                "target_weight": round(position.target_weight, 6),
                "target_amount": round(position.target_weight * total_capital, 6),
            })
        if target_cash_weight > 0:
            payload.append({
                "ticker": CASH_TICKER,
                "name": CASH_NAME,
                "sector": "",
                "industry": "",
                "score": None,
                "current_weight": round(self._cash_weight(current_cash, total_capital), 6),
                "target_weight": round(target_cash_weight, 6),
                "target_amount": round(target_cash_weight * total_capital, 6),
            })
        return sorted(payload, key=lambda item: item["target_weight"], reverse=True)

    def _build_current_portfolio_payload(
        self,
        current_positions: list[HoldingPosition],
        *,
        total_capital: float,
        cash: float,
    ) -> list[dict[str, Any]]:
        if not current_positions and cash <= 0:
            return []
        payload = []
        for position in current_positions:
            amount = position.amount()
            payload.append({
                "ticker": position.ticker,
                "name": position.name,
                "sector": position.sector,
                "industry": position.industry,
                "score": position.score,
                "quantity": position.quantity,
                "price": position.price,
                "amount": round(amount, 6),
                "weight": round(amount / total_capital, 6),
            })
        if cash:
            payload.append({
                "ticker": CASH_TICKER,
                "name": CASH_NAME,
                "sector": "",
                "industry": "",
                "score": None,
                "quantity": 0.0,
                "price": None,
                "amount": round(cash, 6),
                "weight": round(cash / total_capital, 6),
            })
        return payload

    def _build_trade_payload(
        self,
        target_positions: list[TargetPosition],
        *,
        current_positions: list[HoldingPosition],
        total_capital: float,
        min_trade_weight: float,
        current_cash: float,
        target_cash_weight: float,
    ) -> list[dict[str, Any]]:
        if not current_positions and current_cash <= 0:
            return []
        target_map = {position.ticker: position for position in target_positions}
        current_map = {position.ticker: position for position in current_positions}
        current_weights = self._current_weight_map(current_positions, total_capital)

        trades: list[PortfolioTrade] = []
        for ticker in sorted(set(target_map) | set(current_map)):
            target = target_map.get(ticker)
            current = current_map.get(ticker)
            current_weight = current_weights.get(ticker, 0.0)
            target_weight = target.target_weight if target else 0.0
            weight_delta = target_weight - current_weight
            if abs(weight_delta) < min_trade_weight:
                continue
            price = current.price if current else None
            quantity_delta = None
            if price:
                quantity_delta = round(weight_delta * total_capital / price, 4)
            trades.append(
                PortfolioTrade(
                    action="buy" if weight_delta > 0 else "sell",
                    ticker=ticker,
                    weight_delta=weight_delta,
                    current_weight=current_weight,
                    target_weight=target_weight,
                    reason="rebalance_to_target",
                    quantity_delta=quantity_delta,
                    price=price,
                    sector=(target.sector if target else current.sector if current else ""),
                    industry=(target.industry if target else current.industry if current else ""),
                )
            )
        current_cash_weight = self._cash_weight(current_cash, total_capital)
        cash_delta = target_cash_weight - current_cash_weight
        if abs(cash_delta) >= min_trade_weight:
            trades.append(
                PortfolioTrade(
                    action="raise_cash" if cash_delta > 0 else "deploy_cash",
                    ticker=CASH_TICKER,
                    weight_delta=cash_delta,
                    current_weight=current_cash_weight,
                    target_weight=target_cash_weight,
                    reason="cash_rebalance",
                    quantity_delta=None,
                    price=None,
                    sector="",
                    industry="",
                )
            )
        return [
            {
                "action": trade.action,
                "ticker": trade.ticker,
                "weight_delta": round(trade.weight_delta, 6),
                "current_weight": round(trade.current_weight, 6),
                "target_weight": round(trade.target_weight, 6),
                "reason": trade.reason,
                "quantity_delta": trade.quantity_delta,
                "price": trade.price,
                "sector": trade.sector,
                "industry": trade.industry,
            }
            for trade in trades
        ]

    def _build_constraint_diagnostics(
        self,
        target_positions: list[TargetPosition],
        *,
        constraint_type: str,
        sector_targets: dict[str, float],
        industry_targets: dict[str, float],
    ) -> dict[str, list[dict[str, Any]]]:
        sector_actual = compute_group_weights(target_positions, "sector")
        industry_actual = compute_group_weights(target_positions, "industry")
        return {
            "sector": self._diagnostic_rows(sector_targets, sector_actual)
            if constraint_type == "sector"
            else [],
            "industry": self._diagnostic_rows(industry_targets, industry_actual)
            if constraint_type == "industry"
            else [],
        }

    @staticmethod
    def _diagnostic_rows(
        targets: dict[str, float],
        actual: dict[str, float],
    ) -> list[dict[str, Any]]:
        rows = []
        for key in sorted(set(targets) | set(actual)):
            is_specified = key in targets
            target = float(targets[key]) if is_specified else None
            actual_value = float(actual.get(key, 0.0))
            rows.append({
                "group": key,
                "target_weight": round(target, 6) if target is not None else None,
                "actual_weight": round(actual_value, 6),
                "difference": round(actual_value - target, 6) if target is not None else None,
                "specified": is_specified,
            })
        return rows

    @staticmethod
    def _current_weight_map(
        positions: list[HoldingPosition],
        total_capital: float,
    ) -> dict[str, float]:
        if total_capital <= 0:
            return {}
        return {
            position.ticker: position.amount() / total_capital
            for position in positions
        }

    @staticmethod
    def _cash_weight(
        cash: float,
        total_capital: float,
    ) -> float:
        if total_capital <= 0 or cash <= 0:
            return 0.0
        return max(0.0, float(cash) / float(total_capital))

    @staticmethod
    def _target_cash_weight(
        *,
        strategy: str,
        target_positions: list[TargetPosition],
    ) -> float:
        if strategy == "long_short":
            return 1.0 if not target_positions else 0.0
        allocated_weight = sum(
            max(0.0, float(position.target_weight))
            for position in target_positions
        )
        return max(0.0, 1.0 - allocated_weight)

