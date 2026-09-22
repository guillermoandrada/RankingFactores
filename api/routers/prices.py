"""Price data management router."""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile

from api.dependencies import get_price_service, invalidate_price_caches
from api.services.price_service import PriceService

router = APIRouter(prefix="/prices", tags=["prices"])


@router.post("/upload", status_code=201)
async def upload_price_file(
    file: UploadFile = File(...),
    service: PriceService = Depends(get_price_service),
):
    """Upload a Bloomberg wide-format Excel price file and persist the close prices."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    content = await file.read()
    try:
        result = service.ingest_from_file(content, file.filename)
        invalidate_price_caches()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/latest")
async def get_latest_closes(
    tickers: str = Query(..., description="Comma-separated tickers."),
    service: PriceService = Depends(get_price_service),
):
    """Return the latest adjusted close per ticker, cached prices taking priority."""
    requested = [item.strip() for item in tickers.split(",") if item.strip()]
    if not requested:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")
    return service.get_latest_closes(requested)


@router.get("/tickers")
async def list_cached_tickers(
    service: PriceService = Depends(get_price_service),
):
    """List all tickers with cached price data and their date ranges."""
    return {"tickers": service.list_cached_tickers()}


@router.delete("/tickers")
async def delete_cached_tickers(
    tickers: list[str] = Body(...),
    source: str | None = Query(
        None,
        description="Restrict the deletion to one source, e.g. 'yfinance'. Omit to delete all.",
    ),
    service: PriceService = Depends(get_price_service),
):
    """Delete cached price data for specific tickers (JSON body: list of ticker strings)."""
    if not tickers:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")
    try:
        result = service.delete_tickers(tickers, source)
        invalidate_price_caches()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
