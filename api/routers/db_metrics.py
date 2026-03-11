"""DB metrics router: update higher_is_better and N/A treatment."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.dependencies import get_db
from api.schemas.metrics import MetricUpdateRequest

router = APIRouter(prefix="/db-metrics", tags=["metrics"])


@router.put("/{metric_id}")
async def update_db_metric(metric_id: int, request: MetricUpdateRequest):
    """
    Update DB metric parameters (higher_is_better, na_handling).

    This updates the metric definition globally across all periods.
    """
    if request.higher_is_better is None and request.na_handling is None:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of higher_is_better or na_handling.",
        )

    try:
        get_db().update_metric(
            metric_id,
            higher_is_better=request.higher_is_better,
            na_handling=request.na_handling,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"success": True, "metric_id": metric_id}

