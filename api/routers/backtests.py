"""Backtest execution router."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

import api.dependencies as dependencies
from api.schemas.backtests import PortfolioBacktestBody, StrategyBacktestBody

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.post("/portfolio")
async def run_portfolio_backtest(request: PortfolioBacktestBody):
    """Backtest an already-built portfolio over a date range."""
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            None,
            lambda: dependencies.get_backtest_service().backtest_portfolio(request),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/strategy")
async def run_strategy_backtest(request: StrategyBacktestBody):
    """Backtest repeated portfolio builds across a manual schedule."""
    known_periods = set(dependencies.get_db().list_periods())
    missing_periods = sorted(
        {
            window.period
            for window in request.schedule
            if window.period not in known_periods
        }
    )
    if missing_periods:
        raise HTTPException(
            status_code=404,
            detail=f"Periods not found: {', '.join(missing_periods)}",
        )
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            None,
            lambda: dependencies.get_backtest_service().backtest_strategy(request),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
