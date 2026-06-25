"""Sector/industry constraint UI: explicit widgets instead of st.data_editor (avoids cell reset bugs)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def targets_from_dataframe(df: pd.DataFrame) -> dict[str, float]:
    """Build API map group -> weight fraction (0–1) for enabled rows only."""
    result: dict[str, float] = {}
    if df.empty:
        return result
    for _, row in df.iterrows():
        if not bool(row.get("enabled", False)):
            continue
        group = str(row.get("group") or "").strip()
        if not group:
            continue
        try:
            weight = float(row.get("weight") or 0.0) / 100.0
        except (TypeError, ValueError):
            continue
        result[group] = weight
    return result


def _prefilled_targets(options: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"enabled": False, "group": option, "weight": 0.0} for option in sorted(options)],
        columns=["enabled", "group", "weight"],
    )


def ensure_targets_state(state_key: str, options: list[str]) -> pd.DataFrame:
    expected = _prefilled_targets(options)
    current = st.session_state.get(state_key)
    if not isinstance(current, pd.DataFrame):
        st.session_state[state_key] = expected
        return expected

    current_map = {
        str(row.get("group") or "").strip(): row
        for _, row in current.iterrows()
        if str(row.get("group") or "").strip()
    }
    rows: list[dict[str, Any]] = []
    for group in expected["group"].tolist():
        existing = current_map.get(group, {})
        rows.append(
            {
                "enabled": bool(existing.get("enabled", False)),
                "group": group,
                "weight": float(existing.get("weight", 0.0) or 0.0),
            }
        )
    refreshed = pd.DataFrame(rows, columns=["enabled", "group", "weight"])
    st.session_state[state_key] = refreshed
    return refreshed


def set_all_targets_enabled(state_key: str, enabled: bool) -> None:
    current = st.session_state.get(state_key)
    if not isinstance(current, pd.DataFrame) or current.empty:
        return
    updated = current.copy()
    updated["enabled"] = enabled
    st.session_state[state_key] = updated
    epoch_key = f"{state_key}_widget_epoch"
    st.session_state[epoch_key] = int(st.session_state.get(epoch_key, 0)) + 1


def render_constraint_target_fields(
    state_key: str,
    key_prefix: str,
    options: list[str],
) -> pd.DataFrame:
    """
    Render one row per group: label, enable checkbox, weight (%) number input.
    Persists into session_state[state_key].
    """
    ensure_targets_state(state_key, options)
    df = st.session_state[state_key]

    sig_key = f"{state_key}_options_sig"
    opts_sig = tuple(sorted(options))
    if st.session_state.get(sig_key) != opts_sig:
        st.session_state[sig_key] = opts_sig
        epoch_key = f"{state_key}_widget_epoch"
        st.session_state[epoch_key] = int(st.session_state.get(epoch_key, 0)) + 1

    epoch = int(st.session_state.get(f"{state_key}_widget_epoch", 0))

    header_on, header_group, header_weight = st.columns([1, 5, 2])
    header_on.caption("On")
    header_group.caption("Group")
    header_weight.caption("Weight (%)")

    new_rows: list[dict[str, Any]] = []
    for i in range(len(df)):
        row = df.iloc[i]
        group = str(row["group"])
        col_on, col_group, col_weight = st.columns([1, 5, 2])
        with col_on:
            enabled = st.checkbox(
                "Enable restriction",
                value=bool(row["enabled"]),
                key=f"{key_prefix}_en_{i}_e{epoch}",
                label_visibility="collapsed",
            )
        with col_group:
            st.text_input(
                "Group",
                value=group,
                disabled=True,
                key=f"{key_prefix}_grp_{i}_e{epoch}",
                label_visibility="collapsed",
            )
        with col_weight:
            weight_pct = st.number_input(
                "Weight percent",
                min_value=0.0,
                max_value=100.0,
                value=float(row["weight"]),
                step=0.01,
                format="%.2f",
                key=f"{key_prefix}_wt_{i}_e{epoch}",
                label_visibility="collapsed",
            )
        new_rows.append({"enabled": enabled, "group": group, "weight": weight_pct})

    out = pd.DataFrame(new_rows, columns=["enabled", "group", "weight"])
    st.session_state[state_key] = out
    return out
