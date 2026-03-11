"""JSON-backed ranking profile repository and resolver."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

PROFILE_PATH = Path(__file__).with_name("ranking_profiles.json")


def _profile_to_transform_chain(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build transform chain from profile's normalization and winsorization.
    normalization: "zscore" | "normalized_zscore" | "percentile"
    winsorization: false | true | {lower, upper}
    """
    chain: list[dict[str, Any]] = []
    winsor = profile.get("winsorization")
    winsor_mode = profile.get("winsor_mode", "quantile")
    if winsor:
        if winsor_mode == "semi":
            params = winsor if isinstance(winsor, dict) else {}
            k = float(params.get("k", 3.0))
            chain.append({
                "name": "semi_winsor",
                "params": {
                    "k": k,
                },
            })
        else:
            params = winsor if isinstance(winsor, dict) else {}
            chain.append({
                "name": "winsor",
                "params": {
                    "lower": params.get("lower", 0.01),
                    "upper": params.get("upper", 0.99),
                },
            })
    norm = profile.get("normalization", "zscore")
    chain.append({"name": norm, "params": {}})
    return chain


def normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized copy of a profile dict with defaults applied.

    This is the canonical place to ensure that profiles have consistent
    fields (normalization, winsorization, winsor_mode) and no deprecated
    keys before execution.
    """
    p = deepcopy(profile)

    # Remove historical overrides field if still present
    p.pop("overrides", None)

    # Normalization default
    if "normalization" not in p:
        p["normalization"] = "zscore"

    # Winsorization: keep existing values, but default to False when missing
    if "winsorization" not in p:
        p["winsorization"] = False

    # Winsorization mode: quantile or semi (default quantile to match legacy behaviour)
    if "winsor_mode" not in p:
        p["winsor_mode"] = "quantile"

    return p


def _nodes_to_factors_and_layers(
    nodes: dict[str, Any],
    method: str = "linear",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Convert nodes format to factors + layers.
    Leaf nodes (inputs are metrics only) -> factors.
    Non-leaf nodes -> layers, in topological order.
    """
    node_names = set(nodes.keys())
    factors: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []

    def is_metric(name: str) -> bool:
        return name not in node_names

    def get_order() -> list[str]:
        """Topological order: dependencies first."""
        order: list[str] = []
        seen: set[str] = set()

        def visit(n: str) -> None:
            if n in seen:
                return
            seen.add(n)
            inputs = nodes.get(n, {}).get("inputs", {})
            for inp in inputs:
                if not is_metric(inp):
                    visit(inp)
            order.append(n)

        for name in nodes:
            visit(name)
        return order

    order = get_order()
    available = set()

    for node_name in order:
        inputs = nodes.get(node_name, {}).get("inputs", {})
        if not inputs:
            continue
        spec = {"name": node_name, "method": method, "weights": dict(inputs)}

        if all(is_metric(k) for k in inputs):
            factors.append(spec)
            available.add(node_name)
        else:
            layers.append(spec)
            available.add(node_name)

    return factors, layers


def _convert_profile_to_legacy(
    profile: dict[str, Any],
    data: dict[str, Any],  # kept for compatibility with existing call sites
) -> dict[str, Any]:
    """Convert profile schema (nodes, normalization, winsorization) to legacy
    format (factors, layers, metric_transforms).

    The legacy ranking engine expects:
    - factors: leaf nodes where inputs are metrics only
    - layers: composition nodes that combine factors or other layers
    - metric_transforms: transform chain derived from normalization/winsorization.
    """
    nodes = deepcopy(profile.get("nodes", {}))
    if not nodes:
        raise ValueError("Profile must have non-empty 'nodes'.")

    method = profile.get("method", "linear")
    metric_transforms = _profile_to_transform_chain(profile)

    factors, layers = _nodes_to_factors_and_layers(nodes, method=method)
    if not factors:
        raise ValueError("Profile must have at least one factor (leaf node).")
    if not layers:
        # Single root node (e.g. "Scoring"): add synthetic score layer for legacy format
        root_name = factors[-1]["name"]
        layers = [{"name": "score", "method": method, "weights": {root_name: 1.0}}]

    return {
        "factors": factors,
        "layers": layers,
        "metric_transforms": metric_transforms,
        "method": method,
    }


class RankingProfileStore:
    """CRUD operations for ranking profiles stored in JSON."""

    def __init__(self, path: Path = PROFILE_PATH) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Ranking profile file not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, data: dict[str, Any]) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def list_profiles(self) -> dict[str, Any]:
        data = self.load()
        return data.get("profiles", {})

    def get_profile(self, profile_name: str) -> dict[str, Any]:
        data = self.load()
        profile = data.get("profiles", {}).get(profile_name)
        if profile is None:
            raise ValueError(f"Scoring profile '{profile_name}' not found.")
        return profile

    def upsert_scoring_profile(
        self, profile_name: str, profile: dict[str, Any]
    ) -> dict[str, Any]:
        data = self.load()
        data.setdefault("profiles", {})[profile_name] = profile
        self.save(data)
        return data

    def delete_scoring_profile(self, profile_name: str) -> dict[str, Any]:
        data = self.load()
        profiles = data.setdefault("profiles", {})
        if profile_name not in profiles:
            raise ValueError(f"Scoring profile '{profile_name}' not found.")
        profiles.pop(profile_name)
        self.save(data)
        return data


class RankingProfileResolver:
    """Resolve a scoring profile and convert it to the legacy format used by the
    ranking engine.

    The resolver ignores any historical ``overrides`` keys that might still be
    present in stored profiles. Sector/industry–specific behavior is now modeled
    by choosing different profiles at scoring time instead of in-profile
    overrides.
    """

    def __init__(self, store: RankingProfileStore | None = None) -> None:
        self.store = store or RankingProfileStore()

    def resolve(
        self,
        scoring_profile: str,
        industry: str | None = None,  # kept for backwards compatibility
        sector: str | None = None,  # kept for backwards compatibility
    ) -> dict[str, Any]:
        data = self.store.load()
        profile_block = data.get("profiles", {}).get(scoring_profile)
        if profile_block is None:
            raise ValueError(f"Scoring profile '{scoring_profile}' not found.")

        # Industry/sector are kept for backwards-compatible call sites, but
        # profile resolution no longer depends on them. Selection of different
        # profiles per sector/industry is handled at a higher level.
        resolved = normalize_profile(profile_block)
        return _convert_profile_to_legacy(resolved, data)
