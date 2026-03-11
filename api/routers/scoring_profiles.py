"""Scoring Profiles router: GET (list / get by name / resolve), POST (create), PUT (update), DELETE."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import get_profile_store
from api.schemas.profiles import ProfileCreateRequest, ProfileUpsertRequest

router = APIRouter(prefix="/scoring-profiles", tags=["scoring-profiles"])


@router.get("")
async def get_scoring_profiles(
    profile_name: Optional[str] = Query(default=None),
):
    """Get Scoring Profiles. If profile_name is empty, returns all profiles; otherwise returns the single profile."""
    if profile_name and profile_name.strip():
        try:
            return {"profile": get_profile_store().get_profile(profile_name.strip())}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"profiles": get_profile_store().list_profiles()}


@router.post("", status_code=201)
async def create_scoring_profile(request: ProfileCreateRequest):
    """Create a new profile. Same as PUT but name in body."""
    store = get_profile_store()
    store.upsert_scoring_profile(request.profile_name, request.profile)
    profile = store.get_profile(request.profile_name)
    return {
        "success": True,
        "profile_name": request.profile_name,
        "profile": profile,
    }


@router.put("/{profile_name}")
async def upsert_scoring_profile(profile_name: str, request: ProfileUpsertRequest):
    """Update or create a scoring profile. Returns success envelope with profile."""
    store = get_profile_store()
    store.upsert_scoring_profile(profile_name, request.profile)
    profile = store.get_profile(profile_name)
    return {
        "success": True,
        "profile_name": profile_name,
        "profile": profile,
    }


@router.delete("/{profile_name}", status_code=204)
async def delete_scoring_profile(profile_name: str):
    """Delete a scoring profile."""
    try:
        get_profile_store().delete_scoring_profile(profile_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
