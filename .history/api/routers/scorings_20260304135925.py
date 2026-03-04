"""Scorings router: compute period scoring."""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from api.dependencies import get_db
from api.services.ranking_service import compute_ranking

router = APIRouter(prefix="/scorings", tags=["scorings"])


class ScopeItem(BaseModel):
    """Single scope for batch: sector and/or industry."""

    sector: str = ""
    industry: str = ""


class BatchScoringBody(BaseModel):
    """Body for POST /scorings/{period}/batch - compute multiple rankings in parallel."""

    scoring_profile: str = Field(..., description="Scoring profile to run.")
    scopes: list[ScopeItem] = Field(
        ...,
        description="List of sector/industry pairs. Empty sector and industry = full universe.",
    )


class ComputePeriodScoringBody(BaseModel):
    """Body for POST /scorings/{period} - compute period scoring."""

    scoring_profile: str = Field(..., description="Scoring profile to run.")
    industry: str = Field(default="", description="Industry filter; empty means all.")
    sector: str = Field(default="", description="Sector filter; empty means all.")
    export: bool = Field(
        default=False,
        description="If true, return XLSX file instead of JSON.",
    )


@router.post("/{period:path}/batch")
async def compute_period_scoring_batch(period: str, request: BatchScoringBody):
    """Compute rankings for multiple sectors/industries in parallel. Returns dict of scope_key -> result."""
    periods = get_db().list_periods()
    if period not in periods:
        raise HTTPException(status_code=404, detail=f"Period '{period}' not found.")

    def _run_one(scope_item: ScopeItem, scope_key: str) -> tuple[str, dict]:
        try:
            df_ranked = compute_ranking(
                quarter=period,
                industry=scope_item.industry or "",
                sector=scope_item.sector or "",
                scoring_profile=request.scoring_profile,
            )
            records = json.loads(df_ranked.to_json(orient="records", date_format="iso"))
            return scope_key, {
                "period": period,
                "industry": scope_item.industry.strip() or None,
                "sector": scope_item.sector.strip() or None,
                "scoring_profile": request.scoring_profile,
                "count": len(records),
                "ranking": records,
            }
        except (ValueError, KeyError) as exc:
            return scope_key, {"error": str(exc), "ranking": []}

    results: dict[str, dict] = {}
    scopes_with_keys: list[tuple[ScopeItem, str]] = []
    for s in request.scopes:
        key = s.sector.strip() or s.industry.strip() or "All"
        scopes_with_keys.append((s, key))

    max_workers = min(8, max(1, len(scopes_with_keys)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one, item, key): key for item, key in scopes_with_keys}
        for future in as_completed(futures):
            scope_key, result = future.result()
            results[scope_key] = result

    return {"results": results}


@router.post("/{period:path}")
async def compute_period_scoring(period: str, request: ComputePeriodScoringBody):
    """Compute period scoring. Set export=true for XLSX download."""
    periods = get_db().list_periods()
    if period not in periods:
        raise HTTPException(status_code=404, detail=f"Period '{period}' not found.")

    try:
        df_ranked = compute_ranking(
            quarter=period,
            industry=request.industry,
            sector=request.sector,
            scoring_profile=request.scoring_profile,
        )
        records = json.loads(
            df_ranked.to_json(orient="records", date_format="iso")
        )

        if request.export:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_ranked.to_excel(writer, index=False, sheet_name="Ranking")
            buffer.seek(0)
            period_safe = period.replace(" ", "").replace("/", "-")
            scope = request.industry.strip() or request.sector.strip() or "ALL"
            scope_safe = scope.replace(" ", "_")
            filename = f"Ranking_{period_safe}_{scope_safe}.xlsx"
            return Response(
                content=buffer.getvalue(),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )

        return {
            "period": period,
            "industry": request.industry.strip() or None,
            "sector": request.sector.strip() or None,
            "scoring_profile": request.scoring_profile,
            "count": len(records),
            "ranking": records,
        }
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
