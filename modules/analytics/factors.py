"""Factor composition engine. All profiles use factors + layers structure."""

from __future__ import annotations

from typing import Any

import pandas as pd

from modules.analytics.ranking import RankingEngine
from modules.analytics.zscore import ZScoreCalculator


class FactorScoringService:
    """Builds final rankings from profile execution specs."""

    def __init__(self, calculator: ZScoreCalculator, ranking_engine: RankingEngine) -> None:
        self.calculator = calculator
        self.ranking_engine = ranking_engine

    def run(
        self,
        *,
        period: str,
        metric_names: list[str],
        weights: dict[str, float] | None,
        method: str | None,
        industry: str | None,
        sector: str | None,
        profile: dict[str, Any],
    ) -> pd.DataFrame:
        factors = profile.get("factors", [])
        layers = profile.get("layers", [])
        if not factors:
            raise ValueError("Profile must have non-empty 'factors'.")
        if not layers:
            raise ValueError("Profile must have non-empty 'layers'.")

        transform_chain = profile.get("metric_transforms")
        if transform_chain is None:
            transform_chain = profile.get("transforms")

        df_scores, direction_map = self.calculator.compute(
            period=period,
            metric_names=metric_names,
            industry_name=industry,
            sector_name=sector,
            transform_chain=transform_chain,
            out_suffix="_zscore",
        )

        return self._run_multistage(
            df_scores=df_scores,
            direction_map=direction_map,
            profile=profile,
        )

    def _run_multistage(
        self,
        *,
        df_scores: pd.DataFrame,
        direction_map: dict[str, bool],
        profile: dict[str, Any],
    ) -> pd.DataFrame:
        factors = profile.get("factors", [])
        if not factors:
            raise ValueError("Multistage profile requires non-empty 'factors'.")

        stage_df = df_scores.copy()
        factor_names: list[str] = []
        for factor in factors:
            factor_name = str(factor.get("name", "")).strip()
            if not factor_name:
                raise ValueError("Each factor requires a non-empty 'name'.")
            factor_method = factor.get("method", "linear")
            factor_weights = factor.get("weights", {})
            if not factor_weights:
                raise ValueError(f"Factor '{factor_name}' has empty weights.")

            tmp = self.ranking_engine.compute(
                df_z=stage_df,
                weights=factor_weights,
                direction_map=direction_map,
                method=factor_method,
                out_col=factor_name,
            )
            stage_df[f"{factor_name}_zscore"] = tmp[factor_name]
            factor_names.append(factor_name)

        layer_specs = profile.get("layers", [])

        out = stage_df.copy()
        available_nodes = list(factor_names)
        for idx, layer in enumerate(layer_specs):
            is_last_layer = idx == len(layer_specs) - 1
            layer_name = str(layer.get("name", "")).strip() or f"layer_{idx + 1}"
            layer_method = layer.get("method", "linear")
            layer_weights = layer.get("weights", {})
            if not layer_weights:
                raise ValueError(f"Layer '{layer_name}' has empty weights.")

            # Nodes: higher-is-better; metrics: use direction_map
            layer_direction = {name: True for name in available_nodes}
            for w in layer_weights:
                if w not in layer_direction:
                    layer_direction[w] = direction_map.get(w, True)
            out_col = "scoring" if is_last_layer else layer_name
            tmp = self.ranking_engine.compute(
                df_z=out,
                weights=layer_weights,
                direction_map=layer_direction,
                method=layer_method,
                out_col=out_col,
            )
            out[out_col] = tmp[out_col]
            if not is_last_layer:
                out[f"{layer_name}_zscore"] = out[out_col]
                available_nodes.append(layer_name)
            # Last layer: output is "scoring"; no z-score on top of final result

        # Drop redundant columns that duplicate "scoring"
        for c in ("score", "score_zscore", "Scoring_zscore"):
            if c in out.columns:
                out = out.drop(columns=[c], errors="ignore")
        return out
