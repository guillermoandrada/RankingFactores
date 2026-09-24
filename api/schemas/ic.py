"""Request schema for IC analysis endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ICRequest(BaseModel):
    metric_names: list[str] = Field(..., min_length=2, description="At least two metric names (DB metrics or derived formulas).")
    forward_months: int = Field(..., gt=0, description="Forward return horizon in months.")
    periods: list[str] | None = Field(
        default=None,
        description="Optional period filter. None uses all available periods per metric.",
    )
    group_by: Literal["sector", "industry"] | None = Field(
        default=None,
        description="Optional breakdown: also compute Rank IC within each sector or industry.",
    )
