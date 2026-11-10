"""How metrics are named in selectors: derived formulas show their recipe."""

from __future__ import annotations

from typing import Any


def is_derived_metric(metric: dict[str, Any] | None) -> bool:
    """A derived metric is a JSON formula: it has no DB id, only a recipe."""
    return bool(metric and metric.get("derived"))


def formula_text(metric: dict[str, Any]) -> str:
    """Render a derived metric's recipe, for example 'EBIT / EV'."""
    inputs = [str(name) for name in metric.get("metric_names") or []]
    operations = [str(op) for op in metric.get("operations") or []]
    if not inputs:
        return "?"
    parts = [inputs[0]]
    for index, operation in enumerate(operations):
        if index + 1 < len(inputs):
            parts.extend([operation, inputs[index + 1]])
    return " ".join(parts)


def metric_label(metric_index: dict[str, dict[str, Any]], metric_name: str) -> str:
    """Selection label; derived metrics carry their formula so the recipe is visible."""
    metric = metric_index.get(metric_name)
    if not is_derived_metric(metric):
        return metric_name
    return f"{metric_name}  ·  derived: {formula_text(metric or {})}"
