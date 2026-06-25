from __future__ import annotations

from typing import Any

import streamlit as st

TOP_NODE_NAME = "Scoring"


# Terminal transform: (display label, API value)
_TERMINAL_OPTIONS = [
    ("Standardize (z-score)", "zscore"),
    ("Normalized z-score", "normalized_zscore"),
    ("Percentile rank", "percentile"),
]


def render_transforms(key_prefix: str, title: str = "Transform Chain") -> list[dict[str, Any]]:
    st.subheader(title)
    use_winsor = st.checkbox(
        "Use winsorization",
        value=True,
        key=f"{key_prefix}_use_winsor",
        help="Clamp extreme values before normalization using either quantiles or mean±k·σ.",
    )

    winsor_mode = "quantile"
    lower, upper, k = 0.01, 0.99, 3.0
    if use_winsor:
        mode_labels = ["Quantile winsorization (percentiles)", "Semi winsorization (mean ± k·σ)"]
        mode_values = ["quantile", "semi"]
        mode_choice = st.radio(
            "Winsorization method",
            options=mode_labels,
            key=f"{key_prefix}_winsor_mode",
        )
        winsor_mode = mode_values[mode_labels.index(mode_choice)]

        if winsor_mode == "quantile":
            lower_col, upper_col = st.columns(2)
            with lower_col:
                lower = st.number_input(
                    "Winsor lower quantile",
                    min_value=0.0,
                    max_value=0.49,
                    value=0.01,
                    step=0.01,
                    key=f"{key_prefix}_winsor_lower",
                    help="Lower percentile below which values are replaced (e.g. 0.01 = 1st percentile).",
                )
            with upper_col:
                upper = st.number_input(
                    "Winsor upper quantile",
                    min_value=0.51,
                    max_value=1.0,
                    value=0.99,
                    step=0.01,
                    key=f"{key_prefix}_winsor_upper",
                    help="Upper percentile above which values are replaced (e.g. 0.99 = 99th percentile).",
                )
        else:
            k = st.number_input(
                "k (std dev multiplier)",
                min_value=0.1,
                max_value=10.0,
                value=3.0,
                step=0.1,
                key=f"{key_prefix}_semiwinsor_k",
                help="Values outside mean ± k·σ are clipped to the boundary.",
            )

    st.divider()
    terminal_labels = [x[0] for x in _TERMINAL_OPTIONS]
    terminal_values = [x[1] for x in _TERMINAL_OPTIONS]
    terminal_choice = st.selectbox(
        "Terminal transform",
        options=terminal_labels,
        index=0,
        key=f"{key_prefix}_terminal_transform",
        help="Final normalization: z-score (standardize), percentile (rank), or normalized z-score.",
    )
    terminal = terminal_values[terminal_labels.index(terminal_choice)]

    chain: list[dict[str, Any]] = []
    if use_winsor:
        if winsor_mode == "quantile":
            chain.append({"name": "winsor", "params": {"lower": float(lower), "upper": float(upper)}})
        else:
            chain.append({"name": "semi_winsor", "params": {"k": float(k)}})
    chain.append({"name": terminal, "params": {}})

    summary_parts = []
    if use_winsor:
        if winsor_mode == "quantile":
            summary_parts.append(f"Winsor ({lower:.2f}–{upper:.2f})")
        else:
            summary_parts.append(f"Semi winsor (k={k:.2f})")
    summary_parts.append(terminal_choice)
    st.info(f"**Transform chain:** {' → '.join(summary_parts)}")
    return chain


def _weight_inputs(
    title: str,
    input_names: list[str],
    key_prefix: str,
) -> dict[str, float]:
    st.markdown(f"**{title}**")
    selected = st.multiselect(
        "Select inputs",
        options=input_names,
        default=[],
        key=f"{key_prefix}_selected",
    )
    weights: dict[str, float] = {}
    if selected:
        default_w = round(1.0 / len(selected), 4)
        for name in selected:
            weights[name] = float(
                st.number_input(
                    f"Weight - {name}",
                    value=default_w,
                    step=0.01,
                    format="%.4f",
                    key=f"{key_prefix}_w_{name}",
                )
            )
    return weights


def _unique_factor_name(name: str, existing: set[str]) -> str:
    """Ensure factor/layer name is unique."""
    out = name
    n = 0
    while out in existing:
        n += 1
        out = f"{name}_{n}"
    return out


def _flatten_box_to_factors_layers(
    box: dict[str, Any],
    factors: list[dict[str, Any]],
    layers: list[dict[str, Any]],
    name_counter: list[int],
    used_names: set[str],
) -> str | None:
    """
    Recursively flatten a box tree to factors + layers.
    Returns the output node name for this box, or None if empty.
    """
    base_name = (box.get("name") or "factor").strip() or "factor"
    name = _unique_factor_name(base_name, used_names)
    used_names.add(name)
    method = box.get("method", "linear")
    metrics = box.get("metrics", {})  # {metric_name: weight}
    subfactors = box.get("subfactors", [])  # [{"weight": float, "box": {...}}]

    metrics = {k: float(v) for k, v in metrics.items() if k and v is not None}
    subfactors = [s for s in subfactors if s.get("box")]

    if not metrics and not subfactors:
        return None

    if not subfactors:
        factors.append({"name": name, "method": method, "weights": metrics})
        return name

    sub_outputs: dict[str, float] = {}
    for sf in subfactors:
        sub_name = _flatten_box_to_factors_layers(
            sf["box"], factors, layers, name_counter, used_names
        )
        w = float(sf.get("weight", 0))
        if sub_name and w:
            sub_outputs[sub_name] = w

    if metrics:
        # Inline primary metrics into this layer instead of creating a _direct factor
        for m, w in metrics.items():
            sub_outputs[m] = float(w)

    if sub_outputs:
        layers.append({"name": name, "method": method, "weights": sub_outputs})
    return name


def _render_recursive_box(
    metrics: list[str],
    key_prefix: str,
    depth: int = 0,
    max_depth: int = 5,
    box_label: str = "Box",
) -> dict[str, Any]:
    """Render a single box with Add metric / Add subfactor. Returns box structure."""
    if depth >= max_depth:
        st.warning("Maximum nesting depth reached.")
        return {"name": "", "method": "linear", "metrics": {}, "subfactors": []}

    num_metrics_key = f"{key_prefix}_num_metrics"
    num_subfactors_key = f"{key_prefix}_num_subfactors"
    if num_metrics_key not in st.session_state:
        st.session_state[num_metrics_key] = 0
    if num_subfactors_key not in st.session_state:
        st.session_state[num_subfactors_key] = 0

    with st.container(border=True):
        st.markdown(f"**{box_label}**")
        if depth == 0:
            st.caption(f"Root node: {TOP_NODE_NAME}")
            name = TOP_NODE_NAME
        else:
            name = st.text_input(
                "Name",
                value="subfactor",
                key=f"{key_prefix}_name",
            ).strip() or "factor"
        method = st.selectbox(
            "Method",
            ["linear", "softplus"],
            index=0,
            key=f"{key_prefix}_method",
        )

        col_add_m, col_add_s = st.columns(2)
        with col_add_m:
            if st.button("Add metric", key=f"{key_prefix}_add_metric"):
                st.session_state[num_metrics_key] += 1
                st.rerun()
        with col_add_s:
            if st.button("Add subfactor", key=f"{key_prefix}_add_subfactor"):
                st.session_state[num_subfactors_key] += 1
                st.rerun()

        metric_weights: dict[str, float] = {}
        for i in range(st.session_state[num_metrics_key]):
            c1, c2 = st.columns([2, 1])
            with c1:
                m = st.selectbox(
                    "Metric",
                    options=["— Select —"] + sorted(metrics),
                    index=0,
                    key=f"{key_prefix}_m_{i}",
                )
            with c2:
                w = st.number_input(
                    "Weight",
                    value=0.5,
                    min_value=0.0,
                    step=0.01,
                    format="%.2f",
                    key=f"{key_prefix}_w_{i}",
                )
            if m and m != "— Select —":
                metric_weights[m] = float(w)

        subfactor_list: list[dict[str, Any]] = []
        for i in range(st.session_state[num_subfactors_key]):
            with st.container(border=True):
                st.markdown(f"##### Subfactor {i + 1}")
                sf_weight = st.number_input(
                    "Weight for this subfactor",
                    value=0.5,
                    min_value=0.0,
                    step=0.01,
                    format="%.2f",
                    key=f"{key_prefix}_sf_w_{i}",
                )
                sf_box = _render_recursive_box(
                    metrics=metrics,
                    key_prefix=f"{key_prefix}_sub_{i}",
                    depth=depth + 1,
                    max_depth=max_depth,
                    box_label=f"Subfactor {i + 1}",
                )
                if sf_box.get("name"):
                    subfactor_list.append({"weight": float(sf_weight), "box": sf_box})

        return {
            "name": name,
            "method": method,
            "metrics": metric_weights,
            "subfactors": subfactor_list,
        }


def build_multistage_config(
    metrics: list[str],
    key_prefix: str,
    title: str = "Composition",
    max_depth: int = 5,
) -> dict[str, Any]:
    st.subheader(title)
    box = _render_recursive_box(
        metrics=metrics,
        key_prefix=key_prefix,
        depth=0,
        max_depth=max_depth,
        box_label="Composition",
    )

    factors: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    _flatten_box_to_factors_layers(box, factors, layers, [0], set())

    factors = [f for f in factors if f.get("weights")]
    if not factors:
        return {"factors": [], "layers": [{"name": "score", "method": "linear", "weights": {}}]}
    if not layers:
        layers = [{"name": "score", "method": "linear", "weights": {f["name"]: 1.0 for f in factors}}]
    elif layers[-1].get("name") != "score":
        layers.append({"name": "score", "method": "linear", "weights": {layers[-1]["name"]: 1.0}})

    return {"factors": factors, "layers": layers}


def _convert_factors_layers_to_nodes(
    factors: list[dict[str, Any]],
    layers: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Convert legacy factors+layers format to new nodes format.
    Merges 'score' and its sole child into one node 'Scoring' with direct factor inputs.
    """
    nodes: dict[str, dict[str, Any]] = {}
    for f in factors:
        if f.get("weights"):
            nodes[f["name"]] = {
                "inputs": dict(f["weights"]),
                "method": f.get("method", "linear"),
            }
    for layer in layers:
        if layer.get("weights"):
            nodes[layer["name"]] = {
                "inputs": dict(layer["weights"]),
                "method": layer.get("method", "linear"),
            }

    # Merge score + sole child into single "Scoring" node
    score_node = nodes.get("score", {})
    score_inputs = score_node.get("inputs") or {}
    if len(score_inputs) == 1:
        top_name = next(iter(score_inputs))
        top_node = nodes.pop(top_name, {"inputs": {}, "method": "linear"})
        nodes.pop("score", None)
        # Use "Scoring" as the merged node name, keep its inputs, use score layer's method
        nodes[TOP_NODE_NAME] = {
            "inputs": top_node.get("inputs", {}),
            "method": score_node.get("method", "linear"),
        }
    return nodes


def _transforms_to_normalization_winsor(transforms: list[dict[str, Any]]) -> tuple[str, Any, str]:
    """Extract normalization, winsorization, and winsor_mode from transform chain."""
    normalization = "zscore"
    winsorization: Any = False
    winsor_mode = "quantile"
    for step in transforms:
        name = step.get("name", "")
        params = step.get("params") or {}
        if name == "winsor":
            winsor_mode = "quantile"
            winsorization = {"lower": params.get("lower", 0.01), "upper": params.get("upper", 0.99)}
        elif name == "semi_winsor":
            winsor_mode = "semi"
            winsorization = {"k": params.get("k", 3.0)}
        elif name in ("zscore", "normalized_zscore", "percentile"):
            normalization = name
    return normalization, winsorization, winsor_mode


def convert_wizard_to_profile(
    base: dict[str, Any],
    transforms: list[dict[str, Any]] | None = None,
    method: str = "linear",
) -> dict[str, Any]:
    """Convert wizard output to the profile schema (nodes, normalization,
    winsorization, winsor_mode, method)."""
    factors = base.get("factors", [])
    layers = base.get("layers", [])
    nodes = _convert_factors_layers_to_nodes(factors, layers)
    if not nodes:
        return {
            "nodes": {},
            "normalization": "zscore",
            "winsorization": {"lower": 0.01, "upper": 0.99},
            "winsor_mode": "quantile",
            "method": method,
        }

    normalization, winsorization, winsor_mode = "zscore", {"lower": 0.01, "upper": 0.99}, "quantile"
    if transforms:
        normalization, winsorization, winsor_mode = _transforms_to_normalization_winsor(transforms)

    return {
        "nodes": nodes,
        "normalization": normalization,
        "winsorization": winsorization,
        "winsor_mode": winsor_mode,
        "method": method,
    }


def validate_profile_payload(payload: dict[str, Any]) -> list[str]:
    """Validate wizard payload (factors/layers) before conversion."""
    warnings: list[str] = []

    base = payload.get("base", {})
    factors = base.get("factors", [])
    layers = base.get("layers", [])

    if not factors:
        warnings.append("Base has no factors.")
    for factor in factors:
        if not factor.get("weights"):
            warnings.append(f"Factor '{factor.get('name', '?')}' has empty weights.")
    if not layers:
        warnings.append("Base has no layers.")
    for layer in layers:
        if not layer.get("weights"):
            warnings.append(f"Layer '{layer.get('name', '?')}' has empty weights.")

    return warnings


def validate_nodes_profile(profile: dict[str, Any]) -> list[str]:
    """Validate profile in nodes format."""
    warnings: list[str] = []
    nodes = profile.get("nodes", {})
    if not nodes:
        warnings.append("Profile has no nodes.")
    for name, node in nodes.items():
        inputs = node.get("inputs", {})
        if not inputs:
            warnings.append(f"Node '{name}' has empty inputs.")
    if "score" not in nodes and "Scoring" not in nodes:
        warnings.append("Profile should have a 'score' or 'Scoring' node.")
    return warnings
