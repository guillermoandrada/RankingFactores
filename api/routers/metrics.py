"""Metrics router: derived metrics only (GET, POST, PUT, DELETE). DB metrics managed via periods router."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import get_metrics_service
from api.schemas.metrics import (
    DerivedMetricPutRequest,
    MetricPostRequest,
    MetricPreviewRequest,
)
from api.services.metrics_service import DuplicateMetricError, MetricNotFoundError

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
async def list_derived_metrics(
    metric_name: Optional[str] = Query(default=None),
):
    """List derived metrics. Filter by metric_name for single get."""
    all_metrics = get_metrics_service().list_derived_metrics()
    if metric_name and metric_name.strip():
        metric = next(
            (m for m in all_metrics if m.get("metric_name") == metric_name.strip()),
            None,
        )
        if not metric:
            raise HTTPException(status_code=404, detail=f"Derived metric '{metric_name}' not found.")
        return {"metric": metric}
    return {"metrics": all_metrics}


@router.post("", status_code=201)
async def create_derived_metric(request: MetricPostRequest):
    """
    Create a derived metric formula (stored in JSON, computed on the fly).

    Returns 409 when the name is taken: a create must never overwrite an existing
    formula. Use PUT to change one deliberately.
    """
    if not (request.metric_names and len(request.metric_names) >= 2 and request.operations and request.new_metric_name):
        raise HTTPException(
            status_code=400,
            detail="Provide metric_names, operations, and new_metric_name for derived metrics.",
        )
    if len(request.operations) != len(request.metric_names) - 1:
        raise HTTPException(
            status_code=400,
            detail=f"operations must have {len(request.metric_names) - 1} items for {len(request.metric_names)} metrics.",
        )
    try:
        result = get_metrics_service().create_derived_metric(
            metric_name=request.new_metric_name,
            metric_names=request.metric_names,
            operations=request.operations,
            higher_is_better=request.higher_is_better,
            na_handling=request.na_handling,
        )
        return {"success": True, **result}
    except DuplicateMetricError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/preview")
async def preview_derived_metric(request: MetricPreviewRequest):
    """
    Compute a candidate formula on one period without saving it.

    Returns the distribution and the extreme values, so a formula that is valid but
    wrong can be spotted before it reaches a scoring profile.
    """
    if len(request.operations) != len(request.metric_names) - 1:
        raise HTTPException(
            status_code=400,
            detail=f"operations must have {len(request.metric_names) - 1} items for {len(request.metric_names)} metrics.",
        )
    try:
        return get_metrics_service().preview_derived_metric(
            period=request.period,
            metric_names=request.metric_names,
            operations=request.operations,
            metric_name=request.metric_name,
            na_handling=request.na_handling,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.put("/{metric_name:path}")
async def update_derived_metric(metric_name: str, request: DerivedMetricPutRequest):
    """Edit a derived metric formula. The result must still resolve to base metrics."""
    if not any([
        request.metric_names is not None,
        request.operations is not None,
        request.higher_is_better is not None,
        request.na_handling is not None,
    ]):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of metric_names, operations, higher_is_better, or na_handling.",
        )
    try:
        get_metrics_service().update_derived_metric(
            metric_name=metric_name,
            metric_names=request.metric_names,
            operations=request.operations,
            higher_is_better=request.higher_is_better,
            na_handling=request.na_handling,
        )
        return {"success": True, "metric_name": metric_name}
    except MetricNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/{metric_name:path}", status_code=204)
async def delete_derived_metric(metric_name: str):
    """Delete a derived metric formula from JSON."""
    try:
        get_metrics_service().delete_derived_metric(metric_name)
    except MetricNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
