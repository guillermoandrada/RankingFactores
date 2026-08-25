from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

# Reused across Streamlit reruns (same process) to avoid new TCP/TLS per request.
_HTTP_CLIENTS: dict[tuple[str, float], httpx.Client] = {}


def _pooled_http_client(base_url: str, timeout_seconds: float) -> httpx.Client:
    key = (base_url, timeout_seconds)
    if key not in _HTTP_CLIENTS:
        _HTTP_CLIENTS[key] = httpx.Client(
            base_url=base_url,
            timeout=timeout_seconds,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
    return _HTTP_CLIENTS[key]


def _read_timeout(client: httpx.Client) -> float:
    """Read timeout of a client, for error messages. 0 when it is unbounded."""
    return client.timeout.read or 0.0


class ApiError(RuntimeError):
    pass


class RankingApiClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 30.0,
        long_timeout_seconds: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._http = _pooled_http_client(self.base_url, timeout_seconds)
        self._http_long = _pooled_http_client(self.base_url, long_timeout_seconds)

    def _request(self, method: str, path: str, *, long: bool = False, **kwargs: Any) -> Any:
        """
        Send a request and translate every failure into ApiError.

        `long=True` selects the long-timeout client, for endpoints that legitimately run
        for minutes: file imports, backtests and IC analysis. Transport failures are
        wrapped too, so callers only ever have to catch ApiError.
        """
        client = self._http_long if long else self._http
        try:
            response = client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiError(
                f"{method} {path} timed out after {_read_timeout(client):.0f}s. "
                "The server may still be finishing the request — check the result "
                "before retrying."
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"{method} {path} could not reach the API: {exc}") from exc

        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise ApiError(f"{method} {path} failed: {response.status_code} {detail}")

        if response.status_code == 204:
            return {}
        if "application/json" in response.headers.get("content-type", ""):
            try:
                return response.json()
            except Exception:
                return {}
        return response.content

    def get_stats(self) -> dict[str, Any]:
        """Return KPI summary stats for the Home page dashboard."""
        return self._request("GET", "/reference/stats")

    def list_periods(self) -> list[str]:
        payload = self._request("GET", "/reference/periods")
        return payload.get("periods", [])

    def get_period_content(self, period: str) -> dict[str, Any]:
        payload = self._request("GET", f"/periods/{quote(period, safe='')}")
        return payload

    def create_period(
        self,
        file_content: bytes,
        filename: str,
        if_period_exists: str = "replace",
        *,
        reader: str = "bloomberg",
        period: str | None = None,
        index_code: str | None = None,
    ) -> dict[str, Any]:
        """Create period from uploaded file using the selected reader."""
        params: dict[str, Any] = {
            "if_period_exists": if_period_exists,
            "reader": reader,
        }
        if period:
            params["period"] = period
        if index_code:
            params["index_code"] = index_code
        return self._request(
            "POST",
            "/periods",
            params=params,
            files={"file": (filename, file_content)},
            long=True,
        )

    def update_period_with_file(
        self,
        period: str,
        file_content: bytes,
        filename: str,
    ) -> dict[str, Any]:
        """Replace period content by uploading a new file."""
        return self._request(
            "PUT",
            f"/periods/{quote(period, safe='')}",
            files={"file": (filename, file_content)},
            long=True,
        )

    def delete_period(self, period: str) -> dict[str, Any]:
        """Fully delete a period (all fundamental values and index membership). Returns {} on success (204)."""
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
        payload = self._request("GET", "/reference/db/metrics/available")
        return payload.get("metrics", [])

    def list_derived_metrics(self) -> list[dict[str, Any]]:
        """List derived metrics only."""
        payload = self._request("GET", "/metrics")
        return payload.get("metrics", [])

    def list_db_metrics(self) -> list[dict[str, Any]]:
        """List DB metrics only (from SQL)."""
        payload = self._request("GET", "/reference/db/metrics")
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

    def update_db_metric(
        self,
        metric_id: int,
        *,
        higher_is_better: bool | None = None,
        na_handling: str | None = None,
    ) -> dict[str, Any]:
        """
        Update DB metric parameters (higher_is_better, na_handling).

        This updates the metric definition globally across all periods.
        """
        body: dict[str, Any] = {}
        if higher_is_better is not None:
            body["higher_is_better"] = higher_is_better
        if na_handling is not None:
            body["na_handling"] = na_handling
        if not body:
            raise ValueError("Provide at least one of higher_is_better or na_handling.")
        return self._request("PUT", f"/db-metrics/{metric_id}", json=body)

    def upload_variable_file(
        self,
        file_content: bytes,
        filename: str,
        *,
        sheet: str | None = None,
    ) -> dict[str, Any]:
        """
        Upload a Bloomberg individual-variable Excel file (one variable, several periods).

        Uses the long timeout: the server imports every period in the file, which takes
        roughly a second per period.
        """
        return self._request(
            "POST",
            "/db-metrics",
            params={"sheet": sheet} if sheet else None,
            files={"file": (filename, file_content, "application/octet-stream")},
            long=True,
        )

    def list_sectors(self) -> list[str]:
        payload = self._request("GET", "/reference/sectors")
        return payload.get("sectors", [])

    def list_industries(self) -> list[str]:
        payload = self._request("GET", "/reference/industries")
        return payload.get("industries", [])

    def list_indices(self) -> list[str]:
        payload = self._request("GET", "/reference/indices")
        return payload.get("indices", [])

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
        index: str = "",
    ) -> dict[str, Any]:
        body = {
            "industry": industry,
            "sector": sector,
            "index": index,
            "scoring_profile": scoring_profile,
        }
        path = f"/scorings/{quote(period, safe='')}"
        return self._request("POST", path, json=body)

    def construct_portfolio(
        self,
        period: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Run portfolio construction for a period."""
        path = f"/portfolios/{quote(period, safe='')}"
        return self._request("POST", path, json=body)

    def run_portfolio_backtest(
        self,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a backtest for an already-built portfolio."""
        return self._request("POST", "/backtests/portfolio", json=body, long=True)

    def run_strategy_backtest(
        self,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a historical strategy backtest over multiple periods."""
        return self._request("POST", "/backtests/strategy", json=body, long=True)

    def run_ranking_batch(
        self,
        period: str,
        scoring_profile: str,
        scopes: list[tuple[str, str]],
        index: str = "",
    ) -> dict[str, Any]:
        """Run rankings for multiple sector/industry pairs in parallel.

        scopes: list of (sector, industry) tuples. The same scoring_profile
        is applied to all scopes.
        """
        body = {
            "scoring_profile": scoring_profile,
            "index": index,
            "scopes": [{"sector": s, "industry": i} for s, i in scopes],
        }
        path = f"/scorings/{quote(period, safe='')}/batch"
        return self._request("POST", path, json=body)

    def run_ranking_batch_with_profiles(
        self,
        period: str,
        default_scoring_profile: str,
        scopes: list[dict[str, Any]],
        index: str = "",
    ) -> dict[str, Any]:
        """Run rankings for multiple scopes, optionally overriding the
        scoring profile per scope.

        scopes: list of dicts like
        {\"sector\": str, \"industry\": str, \"scoring_profile\": Optional[str]}.
        """
        body = {
            "scoring_profile": default_scoring_profile,
            "index": index,
            "scopes": scopes,
        }
        path = f"/scorings/{quote(period, safe='')}/batch"
        return self._request("POST", path, json=body)

    def run_ic_analysis(
        self,
        metric_names: list[str],
        forward_months: int,
        periods: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run multivariate IC analysis (Rank IC + inter-factor Spearman correlation)."""
        body: dict[str, Any] = {
            "metric_names": metric_names,
            "forward_months": forward_months,
        }
        if periods is not None:
            body["periods"] = periods
        return self._request("POST", "/ic", json=body, long=True)

    def upload_price_file(self, file_content: bytes, filename: str) -> dict[str, Any]:
        """Upload a Bloomberg wide-format Excel price file."""
        return self._request(
            "POST",
            "/prices/upload",
            files={"file": (filename, file_content, "application/octet-stream")},
            long=True,
        )

    def get_latest_prices(self, tickers: list[str]) -> dict[str, Any]:
        """
        Return the latest adjusted close per ticker.

        Resolution happens server-side through the hybrid provider, so uploaded
        Bloomberg prices take priority over Yahoo Finance.
        """
        if not tickers:
            raise ValueError("Provide at least one ticker.")
        return self._request(
            "GET",
            "/prices/latest",
            params={"tickers": ",".join(tickers)},
        )

    def list_cached_price_tickers(self) -> list[dict[str, Any]]:
        """List all tickers with cached price data and their date ranges."""
        data = self._request("GET", "/prices/tickers")
        return data.get("tickers", [])

    def delete_cached_price_tickers(self, tickers: list[str]) -> dict[str, Any]:
        """Delete cached price data for the given tickers."""
        return self._request("DELETE", "/prices/tickers", json=tickers)

    def export_ranking_xlsx(
        self,
        period: str,
        scoring_profile: str,
        industry: str = "",
        sector: str = "",
        index: str = "",
    ) -> bytes:
        body = {
            "industry": industry,
            "sector": sector,
            "index": index,
            "scoring_profile": scoring_profile,
            "export": True,
        }
        path = f"/scorings/{quote(period, safe='')}"
        content = self._request("POST", path, json=body)
        if not isinstance(content, (bytes, bytearray)):
            raise ApiError("Expected binary XLSX payload.")
        return bytes(content)
