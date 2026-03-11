"""Flat store for scoring profiles. Node = {id, type, name, method, params, children}.

ARCHITECTURE:
- nodes: Dict[node_id, Node] - all node definitions
- Node.children: [{child_id, weight, enabled}] - child_id is node_id (subfactor) or metric name (leaf)
- root_id: str - id of root node (e.g. "Scoring")
- Selection and state live in st.session_state["selected_node_id"], etc.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any

# Type aliases
NodeId = str
ChildEntry = dict[str, Any]  # {child_id, weight, enabled}
Node = dict[str, Any]  # {id, type, name, method, params, children}


def _generate_id() -> str:
    """Generate a unique node id."""
    return str(uuid.uuid4())[:8]


def migrate_legacy_to_flat(profile: dict[str, Any]) -> tuple[dict[NodeId, Node], NodeId]:
    """Convert legacy schema (nodes: {name: {inputs: {child: weight}}}) to flat store.

    Returns (nodes_dict, root_id).
    Legacy nodes use name as id; we assign root as first "Scoring"/"score" or first node.
    """
    legacy_nodes = profile.get("nodes") or {}
    if not legacy_nodes:
        return {}, ""

    root_name = "Scoring" if "Scoring" in legacy_nodes else ("score" if "score" in legacy_nodes else next(iter(legacy_nodes)))
    node_names = set(legacy_nodes.keys())
    nodes: dict[NodeId, Node] = {}

    def visit(name: str) -> NodeId:
        if name in nodes:
            return name
        data = legacy_nodes.get(name, {})
        inputs = (data or {}).get("inputs") or {}
        children: list[ChildEntry] = []
        for child_name, w in inputs.items():
            child_id = child_name
            if child_name in node_names:
                visit(child_name)
            children.append({"child_id": child_id, "weight": float(w), "enabled": True})
        nodes[name] = {
            "id": name,
            "type": "root" if name == root_name else ("subfactor" if any(c["child_id"] in node_names for c in children) else "subfactor"),
            "name": name,
            "method": (data or {}).get("method", "linear"),
            "params": {},
            "children": children,
        }
        if name == root_name:
            nodes[name]["type"] = "root"
        return name

    visit(root_name)
    for name in legacy_nodes:
        if name not in nodes:
            visit(name)

    return nodes, root_name


def flat_to_export_payload(
    nodes: dict[NodeId, Node],
    root_id: NodeId,  # root_id kept for symmetry with other helpers
    normalization: str,
    winsorization: Any,
    winsor_mode: str = "quantile",
    method: str = "linear",
) -> dict[str, Any]:
    """Convert flat store back to API payload (nodes dict with inputs plus
    profile-level normalization, winsorization, and aggregation method)."""
    export_nodes: dict[str, Any] = {}

    def to_inputs(node: Node) -> dict[str, float]:
        result: dict[str, float] = {}
        for c in node.get("children", []):
            if not c.get("enabled", True):
                continue
            cid = c.get("child_id", "")
            if not cid:
                continue
            result[cid] = float(c.get("weight", 0))
        return result

    for nid, node in nodes.items():
        export_nodes[nid] = {
            "inputs": to_inputs(node),
            "method": node.get("method", "linear"),
        }

    return {
        "nodes": export_nodes,
        "normalization": normalization,
        "winsorization": winsorization,
        "winsor_mode": winsor_mode,
        "method": method,
    }


def flat_store_to_factors_layers(
    nodes: dict[NodeId, Node],
    root_id: NodeId,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert flat store to factors + layers (for wizard payload)."""
    node_names = set(nodes.keys())
    name_by_id: dict[str, str] = {nid: nd.get("name", nid) for nid, nd in nodes.items()}

    def to_inputs(node: Node) -> dict[str, float]:
        result: dict[str, float] = {}
        for c in node.get("children", []):
            if not c.get("enabled", True):
                continue
            cid = c.get("child_id", "")
            if not cid:
                continue
            key = name_by_id.get(cid, cid)
            result[key] = float(c.get("weight", 0))
        return result

    nodes_by_name: dict[str, dict[str, Any]] = {}
    for nid, node in nodes.items():
        name = node.get("name", nid)
        nodes_by_name[name] = {"inputs": to_inputs(node), "method": node.get("method", "linear")}

    def get_order() -> list[str]:
        order: list[str] = []
        seen: set[str] = set()

        def visit(n: str) -> None:
            if n in seen:
                return
            seen.add(n)
            for inp in nodes_by_name.get(n, {}).get("inputs", {}):
                if inp in nodes_by_name:
                    visit(inp)
            order.append(n)

        for name in nodes_by_name:
            visit(name)
        return order

    order = get_order()
    factors: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []

    for name in order:
        data = nodes_by_name.get(name, {})
        inputs = data.get("inputs", {})
        if not inputs:
            continue
        method = data.get("method", "linear")
        spec = {"name": name, "method": method, "weights": dict(inputs)}
        if all(k not in nodes_by_name for k in inputs):
            factors.append(spec)
        else:
            layers.append(spec)

    if not factors and not layers:
        return [], []
    if not layers and factors:
        layers = [{"name": "score", "method": "linear", "weights": {factors[-1]["name"]: 1.0}}]
    elif layers and layers[-1].get("name") != "score":
        layers.append({"name": "score", "method": "linear", "weights": {layers[-1]["name"]: 1.0}})
    return factors, layers


class ProfileStore:
    """CRUD for flat profile store. Used in-memory; sync to API on save."""

    def __init__(self) -> None:
        self._nodes: dict[NodeId, Node] = {}
        self._root_id: NodeId = ""

    def load_from_profile(self, profile: dict[str, Any]) -> None:
        """Load from profile dict in nodes format."""
        self._nodes, self._root_id = migrate_legacy_to_flat(profile)

    def get_nodes(self) -> dict[NodeId, Node]:
        return self._nodes

    def get_root_id(self) -> NodeId:
        return self._root_id

    def get_node(self, node_id: NodeId) -> Node | None:
        return self._nodes.get(node_id)

    def set_node(self, node_id: NodeId, node: Node) -> None:
        self._nodes[node_id] = node

    def add_child(
        self,
        parent_id: NodeId,
        child_id: NodeId,
        child_type: str,
        weight: float = 0.0,
    ) -> None:
        """Add a child to a node. child_type: 'metric' | 'subfactor'."""
        node = self._nodes.get(parent_id)
        if not node:
            return
        children = node.setdefault("children", [])
        children.append({"child_id": child_id, "weight": weight, "enabled": True})

    def add_subfactor(self, parent_id: NodeId, name: str) -> NodeId:
        """Create a new subfactor node and add as child. Returns new node id."""
        new_id = _generate_id()
        self._nodes[new_id] = {
            "id": new_id,
            "type": "subfactor",
            "name": name,
            "method": "linear",
            "params": {},
            "children": [],
        }
        self.add_child(parent_id, new_id, "subfactor", weight=0.0)
        return new_id

    def add_metric_child(self, parent_id: NodeId, metric_name: str, weight: float = 0.0) -> None:
        """Add a metric as child (leaf)."""
        self.add_child(parent_id, metric_name, "metric", weight)

    def delete_node(self, node_id: NodeId) -> None:
        """Remove node and its subtree. Also remove from all parents' children."""
        if node_id == self._root_id:
            return
        to_remove = set()
        stack = [node_id]
        while stack:
            n = stack.pop()
            to_remove.add(n)
            node = self._nodes.get(n)
            if node:
                for c in node.get("children", []):
                    cid = c.get("child_id")
                    if cid and cid in self._nodes:
                        stack.append(cid)
        for n in to_remove:
            self._nodes.pop(n, None)
        for node in self._nodes.values():
            node["children"] = [c for c in node.get("children", []) if c.get("child_id") not in to_remove]

    def remove_child(self, parent_id: NodeId, child_id: str) -> None:
        """Remove a child from parent."""
        node = self._nodes.get(parent_id)
        if not node:
            return
        node["children"] = [c for c in node.get("children", []) if c.get("child_id") != child_id]

    def get_parent(self, node_id: NodeId) -> NodeId | None:
        """Return parent node id, or None if root."""
        for nid, n in self._nodes.items():
            for c in n.get("children", []):
                if c.get("child_id") == node_id:
                    return nid
        return None

    def duplicate_subtree(self, node_id: NodeId, new_name: str) -> NodeId | None:
        """Duplicate a node and its subtree, add as sibling. Returns new root id of duplicate."""
        node = self._nodes.get(node_id)
        if not node:
            return None
        parent_id = self.get_parent(node_id)
        id_map: dict[str, str] = {}

        def copy_rec(n: Node) -> Node:
            new_id = _generate_id()
            id_map[n["id"]] = new_id
            new_children: list[ChildEntry] = []
            for c in n.get("children", []):
                cid = c["child_id"]
                if cid in self._nodes:
                    child_copy = copy_rec(self._nodes[cid])
                    self._nodes[child_copy["id"]] = child_copy
                    new_children.append({
                        "child_id": child_copy["id"],
                        "weight": c.get("weight", 0),
                        "enabled": c.get("enabled", True),
                    })
                else:
                    new_children.append({
                        "child_id": cid,
                        "weight": c.get("weight", 0),
                        "enabled": c.get("enabled", True),
                    })
            return {
                "id": new_id,
                "type": n["type"],
                "name": new_name if n["id"] == node_id else n["name"],
                "method": n.get("method", "linear"),
                "params": deepcopy(n.get("params", {})),
                "children": new_children,
            }

        new_node = copy_rec(node)
        new_node["name"] = new_name
        new_id = new_node["id"]
        self._nodes[new_id] = new_node
        if parent_id and parent_id in self._nodes:
            w = 0.0
            for c in self._nodes[parent_id].get("children", []):
                if c.get("child_id") == node_id:
                    w = c.get("weight", 0)
                    break
            self.add_child(parent_id, new_id, "subfactor", w)
        return new_id

    def normalize_weights(self, node_id: NodeId) -> None:
        """Rescale enabled children weights to sum to 1."""
        node = self._nodes.get(node_id)
        if not node:
            return
        children = node.get("children", [])
        enabled = [c for c in children if c.get("enabled", True)]
        if not enabled:
            return
        total = sum(c.get("weight", 0) for c in enabled)
        if total <= 0:
            for c in enabled:
                c["weight"] = 1.0 / len(enabled)
        else:
            for c in enabled:
                c["weight"] = c.get("weight", 0) / total

    def equal_weights(self, node_id: NodeId) -> None:
        """Set all enabled children to equal weight."""
        node = self._nodes.get(node_id)
        if not node:
            return
        enabled = [c for c in node.get("children", []) if c.get("enabled", True)]
        if not enabled:
            return
        w = 1.0 / len(enabled)
        for c in enabled:
            c["weight"] = w

    def sort_children_by_weight(self, node_id: NodeId) -> None:
        """Sort children by weight descending."""
        node = self._nodes.get(node_id)
        if not node:
            return
        node["children"] = sorted(node.get("children", []), key=lambda c: -c.get("weight", 0))
