"""JSON-backed store for derived metric formulas (recipes)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DERIVED_METRICS_PATH = Path(__file__).with_name("derived_metrics.json")


class DerivedMetricStore:
    """CRUD for derived metric formulas. Does not write to the database."""

    def __init__(self, path: Path = DERIVED_METRICS_PATH) -> None:
        self.path = path

    def _ensure_file(self) -> None:
        if not self.path.exists():
            self.path.write_text('{"formulas": {}}', encoding="utf-8")

    def load(self) -> dict[str, Any]:
        self._ensure_file()
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, data: dict[str, Any]) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def list_formulas(self) -> dict[str, dict[str, Any]]:
        """Return all derived metric formulas: {metric_name: {metric_names, operations, ...}}."""
        data = self.load()
        return data.get("formulas", {})

    def get_formula(self, metric_name: str) -> dict[str, Any] | None:
        return self.list_formulas().get(metric_name)

    def upsert_formula(
        self,
        metric_name: str,
        metric_names: list[str],
        operations: list[str],
        higher_is_better: bool | None = None,
        na_handling: str | None = None,
    ) -> dict[str, Any]:
        """Add or update a derived metric formula."""
        if len(metric_names) < 2:
            raise ValueError("metric_names must have at least 2 metrics.")
        if len(operations) != len(metric_names) - 1:
            raise ValueError(
                f"operations must have {len(metric_names) - 1} items for {len(metric_names)} metrics."
            )
        for op in operations:
            if str(op).strip() not in {"+", "-", "*", "/"}:
                raise ValueError("operation must be one of: +, -, *, /")
        if not metric_name.strip():
            raise ValueError("metric_name cannot be empty.")

        data = self.load()
        data.setdefault("formulas", {})[metric_name.strip()] = {
            "metric_names": metric_names,
            "operations": operations,
            "higher_is_better": higher_is_better,
            "na_handling": na_handling,
        }
        self.save(data)
        return {
            "metric_name": metric_name.strip(),
            "metric_names": metric_names,
            "operations": operations,
        }

    def update_formula(
        self,
        metric_name: str,
        metric_names: list[str] | None = None,
        operations: list[str] | None = None,
        higher_is_better: bool | None = None,
        na_handling: str | None = None,
    ) -> None:
        """Update an existing derived metric formula. Raises ValueError if not found."""
        data = self.load()
        formulas = data.get("formulas", {})
        if metric_name not in formulas:
            raise ValueError(f"Derived metric '{metric_name}' not found.")
        current = formulas[metric_name]
        if metric_names is not None:
            if len(metric_names) < 2:
                raise ValueError("metric_names must have at least 2 metrics.")
            current["metric_names"] = metric_names
        if operations is not None:
            m_names = current.get("metric_names", [])
            if len(operations) != len(m_names) - 1:
                raise ValueError(
                    f"operations must have {len(m_names) - 1} items for {len(m_names)} metrics."
                )
            for op in operations:
                if str(op).strip() not in {"+", "-", "*", "/"}:
                    raise ValueError("operation must be one of: +, -, *, /")
            current["operations"] = operations
        if higher_is_better is not None:
            current["higher_is_better"] = higher_is_better
        if na_handling is not None:
            current["na_handling"] = na_handling
        self.save(data)

    def delete_formula(self, metric_name: str) -> None:
        data = self.load()
        formulas = data.get("formulas", {})
        if metric_name not in formulas:
            raise ValueError(f"Derived metric '{metric_name}' not found.")
        formulas.pop(metric_name)
        self.save(data)
