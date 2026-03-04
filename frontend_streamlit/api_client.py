from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx


class ApiError(RuntimeError):
    pass


class RankingApiClient:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.request(method, url, **kwargs)
        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise ApiError(f"{method} {path} failed: {response.status_code} {detail}")

        if "application/json" in response.headers.get("content-type", ""):
            return response.json()
        return response.content

    def list_periods(self) -> list[str]:
        payload = self._request("GET", "/periods")
        return payload.get("periods", [])

    def get_period_content(self, period: str) -> dict[str, Any]:
        payload = self._request("GET", f"/periods/{quote(period, safe='')}")
        return payload

    def create_period(
        self,
        file_content: bytes,
        filename: str,
        if_period_exists: str = "replace",
    ) -> dict[str, Any]:
        """Create period from uploaded Bloomberg Excel file. if_period_exists: replace|append."""
        url = f"{self.base_url.rstrip('/')}/periods"
        params = {"if_period_exists": if_period_exists}
        files = {"file": (filename, file_content)}
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(url, params=params, files=files)
        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise ApiError(f"POST /periods failed: {response.status_code} {detail}")
        return response.json()

    def update_period_with_file(
        self,
        period: str,
        file_content: bytes,
        filename: str,
    ) -> dict[str, Any]:
        """Replace period content by uploading a new file."""
        url = f"{self.base_url.rstrip('/')}/periods/{quote(period, safe='')}"
        files = {"file": (filename, file_content)}
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.put(url, files=files)
        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise ApiError(f"PUT /periods/{period} failed: {response.status_code} {detail}")
        return response.json()

    def delete_period(self, period: str) -> dict[str, Any]:
        """Fully delete a period (all fundamental values and index membership)."""
        return self._request("DELETE", f"/periods/{quote(period, safe='')}")

    def edit_period(
        self,
        period: str,
        remove_metrics: list[int] | None = None,
        remove_securities: list[int] | None = None,
        delete_metrics: list[int] | None = None,
        update_values: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Edit period: remove metrics/securities, delete metrics, update values."""
        body: dict[str, Any] = {}
        if remove_metrics:
            body["remove_metrics"] = remove_metrics
        if remove_securities:
            body["remove_securities"] = remove_securities
        if delete_metrics:
            body["delete_metrics"] = delete_metrics
        if update_values:
            body["update_values"] = update_values
        if not body:
            raise ValueError("Provide at least one edit action.")
        return self._request("PUT", f"/periods/{quote(period, safe='')}", json=body)

    def list_metrics(self) -> list[dict[str, Any]]:
        """List all available metrics (DB + derived) for scoring profiles and metric chains."""
        payload = self._request("GET", "/db/metrics/available")
        return payload.get("metrics", [])

    def list_derived_metrics(self) -> list[dict[str, Any]]:
        """List derived metrics only."""
        payload = self._request("GET", "/metrics")
        return payload.get("metrics", [])

    def list_db_metrics(self) -> list[dict[str, Any]]:
        """List DB metrics only (from SQL)."""
        payload = self._request("GET", "/db/metrics")
        return payload.get("metrics", [])

    def get_derived_metric(self, metric_name: str) -> dict[str, Any]:
        payload = self._request("GET", f"/metrics?metric_name={quote(metric_name, safe='')}")
        return payload.get("metric", {})

    def update_derived_metric(
        self,
        metric_name: str,
        metric_names: list[str] | None = None,
        operations: list[str] | None = None,
        higher_is_better: bool | None = None,
        na_handling: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if metric_names is not None:
            body["metric_names"] = metric_names
        if operations is not None:
            body["operations"] = operations
        if higher_is_better is not None:
            body["higher_is_better"] = higher_is_better
        if na_handling is not None:
            body["na_handling"] = na_handling
        return self._request("PUT", f"/metrics/{quote(metric_name, safe='')}", json=body)

    def delete_derived_metric(self, metric_name: str) -> dict[str, Any]:
        return self._request("DELETE", f"/metrics/{quote(metric_name, safe='')}")

    def delete_db_metrics(self, period: str, metric_ids: list[int]) -> dict[str, Any]:
        """Delete metrics from the SQL database entirely via periods PUT (edit)."""
        return self._request(
            "PUT",
            f"/periods/{quote(period, safe='')}",
            json={"delete_metrics": metric_ids},
        )

    def list_sectors(self) -> list[str]:
        payload = self._request("GET", "/sectors")
        return payload.get("sectors", [])

    def list_industries(self) -> list[str]:
        payload = self._request("GET", "/industries")
        return payload.get("industries", [])

    def list_scoring_profiles(self) -> dict[str, Any]:
        payload = self._request("GET", "/scoring-profiles")
        return payload.get("profiles", {})

    def get_scoring_profile(self, profile_name: str) -> dict[str, Any]:
        payload = self._request(
            "GET",
            f"/scoring-profiles?profile_name={quote(profile_name, safe='')}",
        )
        return payload.get("profile", {})

    def upsert_scoring_profile(self, name: str, profile: dict[str, Any]) -> dict[str, Any]:
        return self._request("PUT", f"/scoring-profiles/{name}", json={"profile": profile})

    def delete_scoring_profile(self, name: str) -> dict[str, Any]:
        return self._request("DELETE", f"/scoring-profiles/{name}")

    def create_metric_operation(
        self,
        *,
        metric_names: list[str],
        operations: list[str],
        new_metric_name: str,
        higher_is_better: bool | None = None,
        na_handling: str | None = None,
    ) -> dict[str, Any]:
        """Create derived metric formula. Operations applied left-to-right."""
        body: dict[str, Any] = {
            "metric_names": metric_names,
            "operations": operations,
            "new_metric_name": new_metric_name,
        }
        if higher_is_better is not None:
            body["higher_is_better"] = higher_is_better
        if na_handling is not None:
            body["na_handling"] = na_handling
        return self._request("POST", "/metrics", json=body)

    def run_ranking(
        self,
        period: str,
        scoring_profile: str,
        industry: str = "",
        sector: str = "",
    ) -> dict[str, Any]:
        body = {
            "industry": industry,
            "sector": sector,
            "scoring_profile": scoring_profile,
        }
        path = f"/scorings/{quote(period, safe='')}"
        return self._request("POST", path, json=body)

    def run_ranking_batch(
        self,
        period: str,
        scoring_profile: str,
        scopes: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """Run rankings for multiple sector/industry pairs in parallel. scopes: [(sector, industry), ...]."""
        body = {
            "scoring_profile": scoring_profile,
            "scopes": [{"sector": s, "industry": i} for s, i in scopes],
        }
        path = f"/scorings/{quote(period, safe='')}/batch"
        return self._request("POST", path, json=body)

    def export_ranking_xlsx(
        self,
        period: str,
        scoring_profile: str,
        industry: str = "",
        sector: str = "",
    ) -> bytes:
        body = {
            "industry": industry,
            "sector": sector,
            "scoring_profile": scoring_profile,
            "export": True,
        }
        path = f"/scorings/{quote(period, safe='')}"
        content = self._request("POST", path, json=body)
        if not isinstance(content, (bytes, bytearray)):
            raise ApiError("Expected binary XLSX payload.")
        return bytes(content)
