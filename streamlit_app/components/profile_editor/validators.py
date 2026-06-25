"""Global and per-node validation. Issues have node_id for navigation."""

from __future__ import annotations

from typing import Any

NodeId = str
ValidationIssue = dict[str, Any]  # {node_id, kind, message}


def validate_nodes(nodes: dict[str, dict], root_id: str) -> list[dict[str, Any]]:
    """Run all validations. Returns list of issues. Each issue has node_id, kind, message."""
    issues: list[dict[str, Any]] = []

    for nid, node in nodes.items():
        children = node.get("children", [])
        enabled = [c for c in children if c.get("enabled", True)]
        total = sum(c.get("weight", 0) for c in enabled)

        if not enabled and node.get("type") != "metric":
            issues.append({"node_id": nid, "kind": "empty_node", "message": "No enabled inputs"})

        if enabled and abs(total - 1.0) > 0.001:
            issues.append({
                "node_id": nid,
                "kind": "weight_sum",
                "message": f"Weights sum to {total:.4f} (expected 1.0)",
            })

    return issues
