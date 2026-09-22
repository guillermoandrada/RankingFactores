"""Request schema for IC analysis endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ICRequest(BaseModel):
    metric_names: list[str] = Field(..., min_length=2, description="At least two metric names (DB metrics or derived formulas).")
    forward_months: int = Field(..., gt=0, description="Forward return horizon in months.")
    periods: list[str] | None = Field(
        default=None,
        description="Optional period filter. None uses all available periods per metric.",
    )
