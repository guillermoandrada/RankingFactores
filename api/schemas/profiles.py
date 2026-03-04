from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ProfileUpsertRequest(BaseModel):
    profile: dict[str, Any]


class ProfileCreateRequest(BaseModel):
    profile_name: str
    profile: dict[str, Any]


class ProfileResolveRequest(BaseModel):
    scoring_profile: str
    industry: str = ""
    sector: str = ""
