"""Total-return data management router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from api.dependencies import get_total_return_service
from api.services.total_return_service import TotalReturnService

router = APIRouter(prefix="/total-returns", tags=["total-returns"])


@router.post("/upload", status_code=201)
async def upload_total_return_file(
    file: UploadFile = File(...),
    service: TotalReturnService = Depends(get_total_return_service),
):
    """Upload a Bloomberg block-format total-return Excel file and persist it."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    content = await file.read()
    try:
        return service.ingest_from_file(content, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/windows")
async def list_total_return_windows(
    service: TotalReturnService = Depends(get_total_return_service),
):
    """List every stored return window and how many tickers it covers."""
    return {"windows": service.list_windows()}


@router.delete("/windows")
async def delete_total_return_windows(
    horizon_months: int | None = Query(
        None,
        description="Restrict the deletion to one horizon, e.g. 3. Omit to delete all.",
    ),
    service: TotalReturnService = Depends(get_total_return_service),
):
    """Delete stored total returns, optionally only those of one horizon."""
    try:
        return service.delete_windows(horizon_months)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
