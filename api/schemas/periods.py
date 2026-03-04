"""Request schemas for periods router."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ValueUpdate(BaseModel):
    """Single value update: use (ticker, metric_name) or (security_id, metric_id) with value."""

    ticker: Optional[str] = None
    metric_name: Optional[str] = None
    security_id: Optional[int] = None
    metric_id: Optional[int] = None
    value: float = Field(..., description="New value.")

    @model_validator(mode="after")
    def check_identifiers(self):
        has_ticker = bool(self.ticker and self.metric_name)
        has_ids = self.security_id is not None and self.metric_id is not None
        if not has_ticker and not has_ids:
            raise ValueError("Provide (ticker, metric_name) or (security_id, metric_id).")
        return self


class PeriodEditBody(BaseModel):
    """Body for PUT /periods/{period} when editing the period table."""

    remove_metrics: list[int] = Field(
        default_factory=list,
        description="Metric IDs to remove from this period (deletes their values in this period only).",
    )
    remove_securities: list[int] = Field(
        default_factory=list,
        description="Security IDs to remove from this period (deletes their values in this period only).",
    )
    delete_metrics: list[int] = Field(
        default_factory=list,
        description="Metric IDs to delete from the database entirely (removes metric and all its values).",
    )
    update_values: list[ValueUpdate] = Field(
        default_factory=list,
        description="Value updates: use (ticker, metric_name) or (security_id, metric_id) with value.",
    )


