"""DB metrics router: upload variable values, update higher_is_better and N/A treatment."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from api.dependencies import get_db, get_db_metric_service, invalidate_fundamentals_caches
from api.schemas.metrics import MetricUpdateRequest
from api.services.db_metric_service import DbMetricService

router = APIRouter(prefix="/db-metrics", tags=["metrics"])


@router.post("", status_code=201)
async def create_db_metric_from_file(
    file: UploadFile = File(...),
    sheet: str | None = Query(
        default=None,
        description="Sheet to read. Defaults to the first sheet; its name is the variable name.",
    ),
    service: DbMetricService = Depends(get_db_metric_service),
):
    """
    Create a DB metric from a Bloomberg individual-variable Excel file.

    The file holds one variable observed at several periods; its values replace the
    variable's existing values in every period the file covers.
    """
    if not file.filename or not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="File must be .xlsx or .xls.")
    content = await file.read()
    try:
        result = service.ingest_variable_file(content, file.filename, sheet_name=sheet)
        invalidate_fundamentals_caches()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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

