"""Request schemas for scorings router."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScopeItem(BaseModel):
    """Single scope for batch: sector and/or industry."""

    sector: str = ""
    industry: str = ""
    scoring_profile: str = Field(
        default="",
        description="Optional scoring profile for this scope; falls back to batch-level scoring_profile when empty.",
    )


class BatchScoringBody(BaseModel):
    """Body for POST /scorings/{period}/batch - compute multiple rankings in parallel."""

    scoring_profile: str = Field(..., description="Scoring profile to run.")
    index: str = Field(
        default="",
        description="Index filter shared across all scopes in this batch; empty means all indices.",
    )
    scopes: list[ScopeItem] = Field(
        ...,
        description="List of sector/industry pairs. Empty sector and industry = full universe.",
    )


class ComputePeriodScoringBody(BaseModel):
    """Body for POST /scorings/{period} - compute period scoring."""

    scoring_profile: str = Field(..., description="Scoring profile to run.")
    industry: str = Field(default="", description="Industry filter; empty means all.")
    sector: str = Field(default="", description="Sector filter; empty means all.")
    index: str = Field(
        default="",
        description="Index filter; empty means all indices.",
    )
    export: bool = Field(
        default=False,
        description="If true, return XLSX file instead of JSON.",
    )
