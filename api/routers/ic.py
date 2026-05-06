"""IC analysis router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_ic_service
from api.schemas.ic import ICRequest
from api.services.ic_service import ICService

router = APIRouter(prefix="/ic", tags=["ic"])


@router.post("")
async def run_ic_analysis(
    request: ICRequest,
    service: ICService = Depends(get_ic_service),
):
    """Compute multivariate Rank IC vs forward returns and inter-factor Spearman correlation."""
    try:
        return service.analyze(
            metric_names=request.metric_names,
            forward_months=request.forward_months,
            periods=request.periods,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
