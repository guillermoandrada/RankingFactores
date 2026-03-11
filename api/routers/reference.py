"""Reference data endpoints (sectors, industries, periods, DB metrics)."""

from __future__ import annotations

from fastapi import APIRouter

from api.dependencies import get_db, get_derived_store

router = APIRouter(prefix="/reference", tags=["reference"])


@router.get("/periods")
async def list_periods():
    """List all available periods."""
    return {"periods": get_db().list_periods()}


@router.get("/sectors")
async def list_sectors():
    return {"sectors": get_db().list_sectors()}


@router.get("/industries")
async def list_industries():
    return {"industries": get_db().list_industries()}


@router.get("/indices")
async def list_indices():
    """List all available indices."""
    return {"indices": get_db().list_indices()}


@router.get("/db/metrics")
async def list_db_metrics():
    """List DB metrics (from SQL). For full list including derived, use GET /db/metrics/available."""
    return {"metrics": get_db().list_metrics()}


@router.get("/db/metrics/available")
async def list_available_metrics():
    """List all metrics available for scoring: DB metrics + derived formulas."""
    db_metrics = get_db().list_metrics()
    formulas = get_derived_store().list_formulas()
    result = list(db_metrics)
    for name, f in formulas.items():
        result.append({
            "metric_id": None,
            "metric_name": name,
            "higher_is_better": f.get("higher_is_better"),
            "na_handling": f.get("na_handling"),
            "derived": True,
            "metric_names": f.get("metric_names", []),
            "operations": f.get("operations", []),
        })
    return {"metrics": result}
