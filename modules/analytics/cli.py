"""
Interactive CLI for ranking computation.
"""

from typing import Optional

from modules.analytics.ranking import RankingEngine, export_to_excel
from modules.analytics.zscore import ZScoreCalculator
from modules.config import DB_URL
from modules.db import FinancialDatabase


def _prompt_non_empty(prompt: str) -> str:
    value = input(prompt).strip()
    while not value:
        value = input("Required. Try again: ").strip()
    return value


def _parse_metric_ids(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("At least one metric_id required.")
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise ValueError("metric_ids must be integers separated by commas.") from e


def _prompt_float(prompt: str, default: float) -> float:
    raw = input(prompt).strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print("Invalid value, using default.")
        return default


def main(
    db_url: Optional[str] = None,
) -> None:
    """Run interactive ranking CLI."""
    db = FinancialDatabase(db_url or DB_URL)
    calculator = ZScoreCalculator(engine=db.engine)
    engine = RankingEngine()

    print("=== Security Ranking (Analytics CLI) ===")

    period = _prompt_non_empty("Period (e.g. 2024 Q4): ")
    index_name = input(
        "Index name (e.g. B500, empty = no filter): "
    ).strip() or None
    industry_name = input(
        "Industry name (empty = no filter): "
    ).strip() or None

    metric_ids_str = _prompt_non_empty(
        "Metric IDs (e.g. 1,2,3) from metrics.metric_id: "
    )
    try:
        metric_ids = _parse_metric_ids(metric_ids_str)
    except ValueError as e:
        print(f"Error: {e}")
        return

    try:
        df_z, direction_map = calculator.compute(
            period=period,
            metric_ids=metric_ids,
            index_name=index_name,
            industry_name=industry_name,
        )
    except Exception as e:
        print(f"\nError computing z-scores: {e}")
        return

    metric_names = sorted(direction_map.keys())
    print("\nMetrics available for weighting:")
    for name in metric_names:
        direction = "↑ better" if direction_map.get(name, True) else "↓ better"
        print(f"  - {name} ({direction})")

    print("\nEnter weights (ENTER = 1.0, 0 = ignore):")
    weights: dict[str, float] = {}
    for name in metric_names:
        w = _prompt_float(f"  Weight for {name}: ", default=1.0)
        if w != 0.0:
            weights[name] = w

    if not weights:
        print("No non-zero weights. Aborting.")
        return

    print("\nCombination method:")
    print("  1) Linear")
    print("  2) Softplus")
    method = ""
    while method not in {"1", "2"}:
        method = input("Select 1 or 2: ").strip()

    method_name = "linear" if method == "1" else "softplus"
    df_scored = engine.compute(
        df_z=df_z,
        weights=weights,
        direction_map=direction_map,
        method=method_name,
        out_col="score",
    )
    df_ranked = df_scored.sort_values("score", ascending=False)

    period_safe = period.replace(" ", "")
    index_part = (index_name or "ALL").replace(" ", "")
    filename = f"Ranking_{period_safe}_{index_part}.xlsx"

    export_to_excel(df_ranked, filename, index=True)
    print(f"\nRanking saved to: {filename}")
    print("Top rows:")
    print(df_ranked.head())


if __name__ == "__main__":
    main()
