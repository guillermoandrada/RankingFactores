"""Periods router: GET content, POST create, PUT edit, DELETE remove period."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile

from api.dependencies import get_db, get_period_service
from api.schemas.periods import PeriodEditBody

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/periods", tags=["periods"])


@router.get("/{period:path}")
async def get_period_content(period: str):
    """Retrieve securities with their associated metrics for a period."""
    periods = get_db().list_periods()
    if period not in periods:
        raise HTTPException(status_code=404, detail=f"Period '{period}' not found.")
    return get_db().get_period_content(period)


@router.post("", status_code=201)
async def create_period(
    file: UploadFile = File(...),
    reader: str = Query(
        default="bloomberg",
        description="Reader name: 'bloomberg', 'bql', 'reuters_metrics', or 'auto'.",
    ),
    if_period_exists: str = Query(
        default="replace",
        description="If period exists: 'replace' (overwrite) or 'append' (merge new metrics/securities).",
    ),
    period: str | None = Query(
        default=None,
        description="Manual period override. Required for readers that do not embed the period in the file.",
    ),
    index_code: str | None = Query(
        default=None,
        description="Optional manual index code override.",
    ),
):
    """
    Create a new period table from an imported Excel file.
    If period already exists: replace (overwrite) or append (merge) based on if_period_exists.
    """
    if not file.filename or not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=400,
            detail="File must be .xlsx or .xls.",
        )
    if reader not in ("bloomberg", "bql", "reuters_metrics", "auto"):
        raise HTTPException(
            status_code=400,
            detail="reader must be 'bloomberg', 'bql', 'reuters_metrics', or 'auto'.",
        )
    if if_period_exists not in ("replace", "append"):
        raise HTTPException(
            status_code=400,
            detail="if_period_exists must be 'replace' or 'append'.",
        )
    if reader == "reuters_metrics" and not (period or "").strip():
        raise HTTPException(
            status_code=400,
            detail="period is required when reader='reuters_metrics'.",
        )
    try:
        contents = await file.read()
        return get_period_service().create_period_from_file(
            file_contents=contents,
            filename=file.filename,
            if_period_exists=if_period_exists,
            reader=reader,
            period=period,
            index_code=index_code,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="File not found") from exc
    except Exception:
        logger.exception("Unexpected error during period import (file=%s)", file.filename)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while importing the file.",
        ) from None


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

    file_contents = None
    filename = None
    if file and file.filename:
        file_contents = await file.read()
        filename = file.filename

    try:
        return get_period_service().update_period(
            period,
            file_contents=file_contents,
            filename=filename,
            body=body,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="File not found") from exc
    except Exception:
        logger.exception("Unexpected error during period update (period=%s)", period)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while updating the period.",
        ) from None


@router.delete("/{period:path}", status_code=204)
async def delete_period(period: str):
    """Fully eliminate the period table (all fundamental values and index membership for that period)."""
    periods = get_db().list_periods()
    if period not in periods:
        raise HTTPException(status_code=404, detail=f"Period '{period}' not found.")
    get_period_service().delete_period(period)
