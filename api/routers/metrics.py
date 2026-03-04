"""Metrics router: derived metrics only (GET, POST, PUT, DELETE). DB metrics managed via periods router."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import get_derived_store
from api.schemas.metrics import MetricPostRequest, DerivedMetricPutRequest

router = APIRouter(prefix="/metrics", tags=["metrics"])


def _list_derived_metrics() -> list[dict]:
    """Return all derived metric formulas."""
    formulas = get_derived_store().list_formulas()
    return [
        {
            "metric_name": name,
            "higher_is_better": f.get("higher_is_better"),
            "na_handling": f.get("na_handling"),
            "metric_names": f.get("metric_names", []),
            "operations": f.get("operations", []),
        }
        for name, f in formulas.items()
    ]


@router.get("")
async def list_derived_metrics(
    metric_name: Optional[str] = Query(default=None),
):
    """List derived metrics. Filter by metric_name for single get."""
    all_metrics = _list_derived_metrics()
    if metric_name and metric_name.strip():
        metric = next(
            (m for m in all_metrics if m.get("metric_name") == metric_name.strip()),
            None,
        )
        if not metric:
            raise HTTPException(status_code=404, detail=f"Derived metric '{metric_name}' not found.")
        return {"metric": metric}
    return {"metrics": all_metrics}


@router.post("")
async def create_derived_metric(request: MetricPostRequest):
    """Create derived metric formula (stored in JSON, computed on the fly)."""
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
        result = get_derived_store().upsert_formula(
            metric_name=request.new_metric_name,
            metric_names=request.metric_names,
            operations=request.operations,
            higher_is_better=request.higher_is_better,
            na_handling=request.na_handling,
        )
        return {"success": True, **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/{metric_name:path}")
async def update_derived_metric(metric_name: str, request: DerivedMetricPutRequest):
    """Edit derived metric formula."""
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
        get_derived_store().update_formula(
            metric_name=metric_name,
            metric_names=request.metric_names,
            operations=request.operations,
            higher_is_better=request.higher_is_better,
            na_handling=request.na_handling,
        )
        return {"success": True, "metric_name": metric_name}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{metric_name:path}")
async def delete_derived_metric(metric_name: str):
    """Delete derived metric formula from JSON."""
    try:
        get_derived_store().delete_formula(metric_name)
        return {"success": True, "metric_name": metric_name}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
