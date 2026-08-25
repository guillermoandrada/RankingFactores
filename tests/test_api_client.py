"""Tests for the Streamlit API client's error translation and timeout selection."""

from __future__ import annotations

import httpx
import pytest

import streamlit_app.client.api_client as api_client_module
from streamlit_app.client.api_client import ApiError, RankingApiClient


@pytest.fixture(scope="module", autouse=True)
def _isolate_client_pool():
    """
    Drop the module-level pool around this file so no mocked transport leaks out.

    Scoped to the module on purpose: every test assigns its own transport before
    calling, and rebuilding the pooled clients per test costs ~2s of SSL setup each.
    """
    api_client_module._HTTP_CLIENTS.clear()
    yield
    api_client_module._HTTP_CLIENTS.clear()


def _client_with(handler) -> RankingApiClient:
    client = RankingApiClient("http://testserver")
    transport = httpx.MockTransport(handler)
    client._http._transport = transport
    client._http_long._transport = transport
    return client


def test_upload_variable_file_uses_the_long_timeout_client() -> None:
    """A 40-period file takes over a minute server-side; 30s would always time out."""
    client = RankingApiClient("http://testserver")

    assert client._http.timeout.read == 30.0
    assert client._http_long.timeout.read == 600.0

    used: list[float | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        used.append(request.extensions.get("timeout", {}).get("read"))
        return httpx.Response(201, json={"variable": "Volatility 12m", "periods": []})

    transport = httpx.MockTransport(handler)
    client._http._transport = transport
    client._http_long._transport = transport

    client.upload_variable_file(b"x", "v.xlsx")

    assert used == [600.0]


def test_read_timeout_is_reported_as_a_handled_api_error() -> None:
    """The page catches ApiError; a raw httpx exception would crash the script."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = _client_with(handler)

    with pytest.raises(ApiError) as exc_info:
        client.upload_variable_file(b"x", "v.xlsx")

    message = str(exc_info.value)
    assert "timed out after 600s" in message
    assert "may still be finishing" in message


def test_connection_failure_is_reported_as_a_handled_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client_with(handler)

    with pytest.raises(ApiError, match="could not reach the API"):
        client.list_periods()


def test_error_status_keeps_the_server_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "No valid variable values found in 'v.xlsx'."})

    client = _client_with(handler)

    with pytest.raises(ApiError, match="No valid variable values"):
        client.upload_variable_file(b"x", "v.xlsx")


def test_successful_upload_returns_the_parsed_payload() -> None:
    payload = {
        "variable": "Volatility 12m",
        "periods": [{"period": "2026/06/30", "records_count": 740}],
        "records_count": 740,
    }

    client = _client_with(lambda request: httpx.Response(201, json=payload))

    assert client.upload_variable_file(b"x", "v.xlsx") == payload


@pytest.mark.parametrize(
    ("call", "expected_long"),
    [
        (lambda c: c.create_period(b"x", "f.xlsx"), True),
        (lambda c: c.update_period_with_file("2026/06/30", b"x", "f.xlsx"), True),
        (lambda c: c.upload_variable_file(b"x", "v.xlsx"), True),
        (lambda c: c.upload_price_file(b"x", "p.xlsx"), True),
        (lambda c: c.list_periods(), False),
        (lambda c: c.delete_cached_price_tickers(["AAPL"]), False),
    ],
)
def test_file_uploads_run_on_the_long_timeout_and_reads_do_not(call, expected_long: bool) -> None:
    """Every import path can outlast 30s; plain reads must not hold the UI for 10 minutes."""
    client = RankingApiClient("http://testserver")
    observed: list[float | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request.extensions.get("timeout", {}).get("read"))
        return httpx.Response(200, json={"periods": []})

    transport = httpx.MockTransport(handler)
    client._http._transport = transport
    client._http_long._transport = transport

    call(client)

    assert observed == [600.0 if expected_long else 30.0]
