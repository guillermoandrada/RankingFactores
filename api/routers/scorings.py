"""Scorings router: compute period scoring."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from api.dependencies import get_db
from api.schemas.scorings import BatchScoringBody, ComputePeriodScoringBody, ScopeItem
from api.services.ranking_service import compute_ranking, export_ranking_to_xlsx

router = APIRouter(prefix="/scorings", tags=["scorings"])


@router.post("/{period:path}/batch")
async def compute_period_scoring_batch(period: str, request: BatchScoringBody):
    """Compute rankings for multiple sectors/industries in parallel. Returns dict of scope_key -> result."""
    periods = get_db().list_periods()
    if period not in periods:
        raise HTTPException(status_code=404, detail=f"Period '{period}' not found.")

    def _run_one(scope_item: ScopeItem, scope_key: str) -> tuple[str, dict]:
        try:
            profile_name = scope_item.scoring_profile or request.scoring_profile
            df_ranked = compute_ranking(
                quarter=period,
                index=request.index,
                industry=scope_item.industry or "",
                sector=scope_item.sector or "",
                scoring_profile=profile_name,
            )
            records = json.loads(df_ranked.to_json(orient="records", date_format="iso"))
            return scope_key, {
                "period": period,
                "industry": scope_item.industry.strip() or None,
                "sector": scope_item.sector.strip() or None,
                "scoring_profile": profile_name,
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
            index=request.index,
            scoring_profile=request.scoring_profile,
        )
        records = json.loads(
            df_ranked.to_json(orient="records", date_format="iso")
        )

        if request.export:
            scope = request.industry.strip() or request.sector.strip() or "ALL"
            content, filename = export_ranking_to_xlsx(df_ranked, period, scope)
            return Response(
                content=content,
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
