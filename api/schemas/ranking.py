from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RankingBody(BaseModel):
    industry: str = Field(default="", description="Industry filter; empty means all.")
    sector: str = Field(default="", description="Sector filter; empty means all.")
    scoring_profile: str = Field(
        ...,
        description="Scoring methodology/profile to run (required).",
    )


class MultiScoreBody(BaseModel):
    industry: str = Field(default="", description="Industry filter; empty means all.")
    sector: str = Field(default="", description="Sector filter; empty means all.")
    scoring_profiles: Optional[list[str]] = Field(
        default=None,
        description="Scoring profiles to execute.",
    )
