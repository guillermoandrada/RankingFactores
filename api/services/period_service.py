"""Period service: create, update, delete period orchestration."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from modules.models import ImportResult


class PeriodService:
    """Orchestrates period create, update, delete. Delegates to importer and db."""

    def __init__(
        self,
        *,
        db: "FinancialDatabase",  # type: ignore[name-defined]
        importer: "DataImporter",  # type: ignore[name-defined]
    ) -> None:
        self._db = db
        self._importer = importer

    def create_period_from_file(
        self,
        file_contents: bytes,
        filename: str,
        if_period_exists: str,
    ) -> dict[str, Any]:
        """
        Create a period from uploaded file contents.
        Returns structured result with period name and counts.
        Raises ValueError for invalid input or import errors.
        """
        if not filename or not filename.lower().endswith((".xlsx", ".xls")):
            raise ValueError("File must be .xlsx or .xls (Bloomberg Excel format)")
        if if_period_exists not in ("replace", "append"):
            raise ValueError("if_period_exists must be 'replace' or 'append'.")

        tmp_suffix = Path(filename).suffix or ".xlsx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix) as tmp:
            tmp.write(file_contents)
            tmp_path = tmp.name

        try:
            result = self._importer.import_file(
                tmp_path,
                verbose=False,
                reader="bloomberg",
                if_period_exists=if_period_exists,
            )
            return _import_result_to_dict(result)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def update_period(
        self,
        period: str,
        *,
        file_contents: bytes | None = None,
        filename: str | None = None,
        body: Any | None = None,
    ) -> dict[str, Any]:
        """
        Update a period: either via file upload or body (remove metrics/securities, update values).
        Returns structured result.
        """
        if body and (
            body.remove_metrics
            or body.remove_securities
            or body.delete_metrics
            or body.update_values
        ):
            return self._update_period_from_body(period, body)

        if file_contents is not None and filename and filename.lower().endswith((".xlsx", ".xls")):
            return self._replace_period_from_file(period, file_contents, filename)

        raise ValueError(
            "Provide either a file (to replace period content) or body with "
            "remove_metrics, remove_securities, and/or update_values."
        )

    def _update_period_from_body(self, period: str, body: Any) -> dict[str, Any]:
        """Handle PUT body: remove metrics, remove securities, delete metrics, update values."""
        db = self._db
        removed_metrics_rows = 0
        removed_securities_rows = 0
        deleted_metric_ids: list[int] = []
        updated_values = 0

        if body.remove_metrics:
            removed_metrics_rows = db.remove_metrics_from_period(
                period, body.remove_metrics
            )
        if body.remove_securities:
            removed_securities_rows = db.remove_securities_from_period(
                period, body.remove_securities
            )
        if body.delete_metrics:
            for mid in body.delete_metrics:
                try:
                    db.delete_metric(mid)
                    deleted_metric_ids.append(mid)
                except ValueError:
                    pass
        if body.update_values:
            updates = []
            for u in body.update_values:
                d = u.model_dump(exclude_none=True)
                if "value" not in d:
                    continue
                updates.append({
                    "ticker": d.get("ticker"),
                    "metric_name": d.get("metric_name"),
                    "security_id": d.get("security_id"),
                    "metric_id": d.get("metric_id"),
                    "value": d["value"],
                })
            updated_values = db.update_period_values(period, updates)

        return {
            "success": True,
            "period": period,
            "action": "edit",
            "removed_metrics_rows": removed_metrics_rows,
            "removed_securities_rows": removed_securities_rows,
            "deleted_metric_ids": deleted_metric_ids,
            "updated_values": updated_values,
        }

    def _replace_period_from_file(
        self, period: str, file_contents: bytes, filename: str
    ) -> dict[str, Any]:
        """Replace period content from uploaded file."""
        tmp_suffix = Path(filename).suffix or ".xlsx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix) as tmp:
            tmp.write(file_contents)
            tmp_path = tmp.name

        try:
            result = self._importer.import_file(
                tmp_path,
                verbose=False,
                period_override=period,
                if_period_exists="replace",
            )
            return _import_result_to_dict(result)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def delete_period(self, period: str) -> None:
        """Remove the period table entirely."""
        self._db.delete_period(period)


def _import_result_to_dict(result: ImportResult) -> dict[str, Any]:
    """Convert ImportResult to API response dict."""
    return {
        "success": True,
        "period": result.period,
        "companies_count": result.companies_count,
        "metrics_count": result.metrics_count,
        "records_count": result.records_count,
        "index_code": result.index_code,
    }
