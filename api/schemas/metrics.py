from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MetricCreateRequest(BaseModel):
    metric_name: str = Field(..., description="Name of the new metric.")
    higher_is_better: Optional[bool] = Field(
        default=None,
        description="True if higher values are better.",
    )
    na_handling: Optional[str] = Field(
        default=None,
        description="How to handle N/A values: replace_with_zero, replace_with_high, replace_with_low, eliminate.",
    )


class MetricUpdateRequest(BaseModel):
    higher_is_better: Optional[bool] = Field(
        default=None,
        description="True if higher values are better.",
    )
    na_handling: Optional[str] = Field(
        default=None,
        description="How to handle N/A values: replace_with_zero, replace_with_high, replace_with_low, eliminate.",
    )


class DerivedMetricPutRequest(BaseModel):
    """Update derived metric formula. All fields optional."""

    metric_names: Optional[list[str]] = None
    operations: Optional[list[str]] = None
    higher_is_better: Optional[bool] = None
    na_handling: Optional[str] = None


class MetricOperationRequest(BaseModel):
    """Create derived metric from multiple metrics. Operations applied left-to-right."""

    metric_names: list[str] = Field(
        ...,
        min_length=2,
        description="Ordered list of metric names (e.g. ['Debt', 'Assets', 'Equity']).",
    )
    operations: list[str] = Field(
        ...,
        description="Operations between metrics: +, -, *, /. Length must be len(metric_names)-1.",
    )
    new_metric_name: str = Field(..., description="Name of the derived metric.")
    higher_is_better: Optional[bool] = None
    na_handling: Optional[str] = None


class MetricPostRequest(BaseModel):
    """Union: simple create (metric_name) or derived (metric_names, operations)."""

    metric_name: Optional[str] = None
    higher_is_better: Optional[bool] = None
    na_handling: Optional[str] = None
    metric_names: Optional[list[str]] = None
    operations: Optional[list[str]] = None
    new_metric_name: Optional[str] = None
