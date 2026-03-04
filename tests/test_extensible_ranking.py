from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

import api.main as api_main
import api.routers.metrics as metrics_router
import api.routers.scorings as scorings_router
from modules.analytics.ranking import RankingEngine
from modules.analytics.transforms import build_default_transform_registry
from modules.config.ranking_profiles import RankingProfileResolver, RankingProfileStore


def test_profile_precedence(tmp_path: Path) -> None:
    profile_data = {
        "pipelines": {
            "standard_z": [
                {"winsor": {"lower": 0.01, "upper": 0.99}},
                {"zscore": {}},
            ]
        },
        "defaults": {"method": "linear", "pipeline": "standard_z"},
        "profiles": {
            "core": {
                "nodes": {
                    "base": {"inputs": {"M1": 1.0}},
                    "score": {"inputs": {"base": 1.0}},
                },
                "overrides": {
                    "sector": {
                        "Tech": {"patch": {"nodes.base.inputs": {"M1": 2.0}}},
                    },
                    "industry": {
                        "Software": {"patch": {"nodes.base.inputs": {"M1": 3.0}}},
                    },
                },
            }
        },
    }
    config_path = tmp_path / "profiles.json"
    config_path.write_text(json.dumps(profile_data), encoding="utf-8")

    resolver = RankingProfileResolver(RankingProfileStore(path=config_path))
    resolved = resolver.resolve("core", industry="Software", sector="Tech")
    assert resolved["factors"][0]["weights"]["M1"] == 3.0


def test_transform_chain_handles_na() -> None:
    registry = build_default_transform_registry()
    series = pd.Series([1.0, 2.0, None, 100.0])
    chain = [
        {"name": "winsor", "params": {"lower": 0.05, "upper": 0.95}},
        {"name": "zscore", "params": {}},
    ]
    out = registry.apply_chain(series, chain)
    assert len(out) == 4
    assert not out.isna().any()


def test_combiner_methods() -> None:
    df = pd.DataFrame(
        {
            "A_zscore": [1.0, 0.0],
            "B_zscore": [0.5, -0.5],
        }
    )
    direction = {"A": True, "B": False}
    weights = {"A": 1.0, "B": 2.0}
    engine = RankingEngine(zsuffix="_zscore")

    linear = engine.compute(df, weights, direction, method="linear", out_col="score")
    softplus = engine.compute(df, weights, direction, method="softplus", out_col="score")

    assert "score" in linear.columns
    assert "score" in softplus.columns
    assert linear["score"].iloc[0] != linear["score"].iloc[1]


def test_export_endpoint_returns_valid_xlsx(monkeypatch) -> None:
    def _fake_compute(**_kwargs):
        return pd.DataFrame(
            {
                "security_id": [1, 2],
                "ticker": ["AAA", "BBB"],
                "score": [1.2, 0.8],
            }
        )

    monkeypatch.setattr(scorings_router, "compute_ranking", _fake_compute)
    client = TestClient(api_main.app)

    response = client.post(
        "/scorings/2023%20Q3",
        json={
            "industry": "",
            "sector": "",
            "scoring_profile": "quality_scoring",
            "export": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    assert response.content[:2] == b"PK"


def test_compute_period_scoring_endpoint(monkeypatch) -> None:
    def _fake_compute(**_kwargs):
        return pd.DataFrame(
            {
                "security_id": [1, 2],
                "ticker": ["AAA", "BBB"],
                "score": [1.0, 0.5],
            }
        )

    monkeypatch.setattr(scorings_router, "compute_ranking", _fake_compute)
    client = TestClient(api_main.app)

    response = client.post(
        "/scorings/2023%20Q3",
        json={
            "scoring_profile": "quality_scoring",
            "industry": "",
            "sector": "",
            "export": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["scoring_profile"] == "quality_scoring"


def test_metric_operation_endpoint(monkeypatch) -> None:
    class _FakeDerivedStore:
        def upsert_formula(self, **kwargs):
            return {
                "metric_name": kwargs["metric_name"],
                "metric_names": kwargs["metric_names"],
                "operations": kwargs["operations"],
            }

    monkeypatch.setattr(metrics_router, "get_derived_store", lambda: _FakeDerivedStore())
    client = TestClient(api_main.app)

    response = client.post(
        "/metrics",
        json={
            "metric_names": ["Debt", "Assets"],
            "operations": ["/"],
            "new_metric_name": "Debt/Assets",
            "higher_is_better": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["metric_name"] == "Debt/Assets"
    assert payload["operations"] == ["/"]


def test_scoring_profile_get_and_delete(monkeypatch) -> None:
    client = TestClient(api_main.app)

    response = client.get("/scoring-profiles?profile_name=Value%20Test")
    assert response.status_code == 200
    payload = response.json()
    assert "profile" in payload
    assert "nodes" in payload["profile"]

    response = client.get("/scoring-profiles?profile_name=nonexistent")
    assert response.status_code == 404

    response = client.delete("/scoring-profiles/nonexistent")
    assert response.status_code == 404
