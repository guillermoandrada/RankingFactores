from __future__ import annotations

from typing import Any

import streamlit as st

TOP_NODE_NAME = "Scoring"


def render_transforms(key_prefix: str, title: str = "Transform Chain") -> list[dict[str, Any]]:
    st.subheader(title)
    use_winsor = st.checkbox("Use winsor transform", value=True, key=f"{key_prefix}_use_winsor")
    lower = st.number_input(
        "Winsor lower quantile",
        min_value=0.0,
        max_value=0.49,
        value=0.01,
        step=0.01,
        key=f"{key_prefix}_winsor_lower",
    )
    upper = st.number_input(
        "Winsor upper quantile",
        min_value=0.51,
        max_value=1.0,
        value=0.99,
        step=0.01,
        key=f"{key_prefix}_winsor_upper",
    )
    terminal = st.selectbox(
        "Terminal transform",
        ["zscore", "normalized_zscore", "identity"],
        index=0,
        key=f"{key_prefix}_terminal_transform",
    )

    chain: list[dict[str, Any]] = []
    if use_winsor:
        chain.append({"name": "winsor", "params": {"lower": float(lower), "upper": float(upper)}})
    chain.append({"name": terminal, "params": {}})
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


def build_overrides(
    scope_name: str,
    metrics: list[str],
    key_prefix: str,
    max_depth: int = 3,
    options: list[str] | None = None,
) -> dict[str, Any]:
    st.subheader(f"{scope_name.capitalize()} Overrides")
    override_count = int(
        st.number_input(
            f"Number of {scope_name} override boxes",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key=f"{key_prefix}_{scope_name}_count",
        )
    )
    overrides: dict[str, Any] = {}
    for i in range(override_count):
        with st.expander(f"{scope_name.capitalize()} override #{i + 1}", expanded=False):
            if options:
                select_options = ["— Select one —"] + sorted(options)
                selected = st.selectbox(
                    f"Select {scope_name}",
                    options=select_options,
                    index=0,
                    key=f"{key_prefix}_{scope_name}_name_{i}",
                )
                target_name = "" if selected == "— Select one —" else selected.strip()
            else:
                target_name = st.text_input(
                    f"{scope_name.capitalize()} name",
                    value="",
                    key=f"{key_prefix}_{scope_name}_name_{i}",
                ).strip()
            if not target_name:
                st.info("Select or enter a target name to keep this override.")
                continue

            transforms = render_transforms(
                key_prefix=f"{key_prefix}_{scope_name}_{i}_transforms",
                title=f"{scope_name.capitalize()} '{target_name}' transform chain",
            )
            profile = build_multistage_config(
                metrics=metrics,
                key_prefix=f"{key_prefix}_{scope_name}_{i}_multi",
                title=f"{scope_name.capitalize()} '{target_name}' composition",
                max_depth=max_depth,
            )
            profile["metric_transforms"] = transforms
            overrides[target_name] = profile
    return overrides


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
            nodes[f["name"]] = {"inputs": dict(f["weights"])}
    for layer in layers:
        if layer.get("weights"):
            nodes[layer["name"]] = {"inputs": dict(layer["weights"])}

    # Merge score + sole child into single "Scoring" node
    score_inputs = nodes.get("score", {}).get("inputs") or {}
    if len(score_inputs) == 1:
        top_name = next(iter(score_inputs))
        top_node = nodes.pop(top_name, {"inputs": {}})
        nodes.pop("score", None)
        # Use "Scoring" as the merged node name, keep its inputs (factors)
        nodes[TOP_NODE_NAME] = top_node
    return nodes


def _override_to_patch(override: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a single override (factors, layers) to patch format.
    Patch keys are dot paths like "nodes.Value.inputs", values are the new inputs dict.
    """
    factors = override.get("factors", [])
    layers = override.get("layers", [])
    override_nodes = _convert_factors_layers_to_nodes(factors, layers)
    patch: dict[str, Any] = {}
    for node_name, node_data in override_nodes.items():
        inputs = node_data.get("inputs")
        if inputs is not None:
            patch[f"nodes.{node_name}.inputs"] = dict(inputs)
    return patch


def convert_wizard_to_profile(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert wizard output (factors, layers, metric_transforms) to new profile schema (nodes, overrides).
    Uses default pipeline; overrides are converted to patch format from their composition.
    """
    factors = base.get("factors", [])
    layers = base.get("layers", [])
    nodes = _convert_factors_layers_to_nodes(factors, layers)
    if not nodes:
        return {"nodes": {}, "overrides": {"sector": {}, "industry": {}}}

    sector_overrides: dict[str, Any] = {}
    industry_overrides: dict[str, Any] = {}
    for name, override in overrides.get("sector", {}).items():
        if name:
            patch = _override_to_patch(override)
            sector_overrides[name] = {"patch": patch}
    for name, override in overrides.get("industry", {}).items():
        if name:
            patch = _override_to_patch(override)
            industry_overrides[name] = {"patch": patch}

    return {
        "nodes": nodes,
        "overrides": {"sector": sector_overrides, "industry": industry_overrides},
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
