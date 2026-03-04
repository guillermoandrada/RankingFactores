"""Periods router: GET content, POST create, PUT edit, DELETE remove period."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile

from api.dependencies import get_db, get_importer
from api.schemas.periods import PeriodEditBody

router = APIRouter(prefix="/periods", tags=["periods"])


@router.get("/{period:path}")
async def get_period_content(period: str):
    """Retrieve securities with their associated metrics for a period."""
    periods = get_db().list_periods()
    if period not in periods:
        raise HTTPException(status_code=404, detail=f"Period '{period}' not found.")
    return get_db().get_period_content(period)


@router.post("")
async def create_period(
    file: UploadFile = File(...),
    if_period_exists: str = Query(
        default="replace",
        description="If period exists: 'replace' (overwrite) or 'append' (merge new metrics/securities).",
    ),
):
    """
    Create a new period table from an imported Bloomberg Excel file.
    If period already exists: replace (overwrite) or append (merge) based on if_period_exists.
    """
    if not file.filename or not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=400,
            detail="File must be .xlsx or .xls (Bloomberg Excel format)",
        )
    if if_period_exists not in ("replace", "append"):
        raise HTTPException(status_code=400, detail="if_period_exists must be 'replace' or 'append'.")

    tmp_suffix = Path(file.filename).suffix or ".xlsx"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = get_importer().import_file(
            tmp_path,
            verbose=False,
            reader="bloomberg",
            if_period_exists=if_period_exists,
        )
        Path(tmp_path).unlink(missing_ok=True)

        return {
            "success": True,
            "period": result.period,
            "companies_count": result.companies_count,
            "metrics_count": result.metrics_count,
            "records_count": result.records_count,
            "index_code": result.index_code,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("/{period:path}")
async def update_period(
    period: str,
    file: UploadFile | None = File(default=None),
    body: PeriodEditBody | None = Body(default=None),
):
    """
    Edit the period table.
    - With file: replace period content by uploading a new file.
    - With body: remove metrics, remove securities, and/or update values.
    """
    periods = get_db().list_periods()
    if period not in periods:
        raise HTTPException(status_code=404, detail=f"Period '{period}' not found.")

    if body and (
        body.remove_metrics
        or body.remove_securities
        or body.delete_metrics
        or body.update_values
    ):
        db = get_db()
        removed_metrics_rows = 0
        removed_securities_rows = 0
        deleted_metric_ids: list[int] = []
        updated_values = 0

        if body.remove_metrics:
            removed_metrics_rows = db.remove_metrics_from_period(
                period, body.remove_metrics
            )
        if body.remove_securities:
            removed_securities_rows = db.remove_securities_from_period(
                period, body.remove_securities
            )
        if body.delete_metrics:
            for mid in body.delete_metrics:
                try:
                    db.delete_metric(mid)
                    deleted_metric_ids.append(mid)
                except ValueError:
                    pass
        if body.update_values:
            updates = []
            for u in body.update_values:
                d = u.model_dump(exclude_none=True)
                if "value" not in d:
                    continue
                updates.append({
                    "ticker": d.get("ticker"),
                    "metric_name": d.get("metric_name"),
                    "security_id": d.get("security_id"),
                    "metric_id": d.get("metric_id"),
                    "value": d["value"],
                })
            updated_values = db.update_period_values(period, updates)

        return {
            "success": True,
            "period": period,
            "action": "edit",
            "removed_metrics_rows": removed_metrics_rows,
            "removed_securities_rows": removed_securities_rows,
            "deleted_metric_ids": deleted_metric_ids,
            "updated_values": updated_values,
        }

    if file and file.filename and file.filename.lower().endswith((".xlsx", ".xls")):
        tmp_suffix = Path(file.filename).suffix or ".xlsx"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            result = get_importer().import_file(
                tmp_path,
                verbose=False,
                period_override=period,
                if_period_exists="replace",
            )
            Path(tmp_path).unlink(missing_ok=True)

            return {
                "success": True,
                "period": result.period,
                "companies_count": result.companies_count,
                "metrics_count": result.metrics_count,
                "records_count": result.records_count,
                "index_code": result.index_code,
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    raise HTTPException(
        status_code=400,
        detail="Provide either a file (to replace period content) or body with remove_metrics, remove_securities, and/or update_values.",
    )


@router.delete("/{period:path}")
async def delete_period(period: str):
    """Fully eliminate the period table (all fundamental values and index membership for that period)."""
    periods = get_db().list_periods()
    if period not in periods:
        raise HTTPException(status_code=404, detail=f"Period '{period}' not found.")
    get_db().delete_period(period)
    return {"success": True, "period": period}
