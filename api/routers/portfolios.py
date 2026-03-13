"""Portfolio construction router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.dependencies import get_db, get_portfolio_service
from api.schemas.portfolios import PortfolioBuildBody

router = APIRouter(prefix="/portfolios", tags=["portfolios"])


@router.post("/{period:path}")
async def construct_portfolio(period: str, request: PortfolioBuildBody):
    """Construct a portfolio from the scored universe for a period."""
    periods = get_db().list_periods()
    if period not in periods:
        raise HTTPException(status_code=404, detail=f"Period '{period}' not found.")

    try:
        return get_portfolio_service().construct_portfolio(period, request)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

