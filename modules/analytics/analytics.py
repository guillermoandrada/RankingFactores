"""
Analytics.py

Cálculo de z-scores winsorizados y rankings de acciones
(lineales y tipo softplus) sobre la base de datos financial_data.db.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Conexión a la base de datos (este fichero debe estar junto a financial_data.db)
ENGINE = create_engine("sqlite:///financial_data.db")


# ---------------------------------------------------------------------------
# 1. Cálculo de Z-Scores (con winsorization) + filtros por índice / industria
# ---------------------------------------------------------------------------
def get_zscore_ranking(
    as_of_date: str,
    metric_ids: List[int],
    index_name: Optional[str] = None,
    industry_name: Optional[str] = None,
    engine=ENGINE,
) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    """
    Calcula z-scores por métrica para cada acción en una fecha dada,
    con winsorization 1%-99%, y filtros opcionales por índice / industria.

    Parameters
    ----------
    as_of_date:
        Valor de la columna 'period' en fundamental_values (ej. '2024 Q4').
    metric_ids:
        Lista de metric_id a incluir en el cálculo.
    index_name:
        Nombre del índice (tabla indices.name) para filtrar.
        Si se proporciona, se hace JOIN con index_memberships e indices,
        filtrando index_memberships.period == as_of_date.
    industry_name:
        Nombre de la industria (tabla industries.name) para filtrar.

    Returns
    -------
    df_zscores:
        DataFrame indexado por (security_id, ticker), con columnas de valores
        winsorizados y columnas *_zscore.
    direction_map:
        Diccionario {metric_name: higher_is_better_bool}.
    """
    if not metric_ids:
        raise ValueError("metric_ids no puede estar vacío.")

    metric_placeholders = ", ".join(f":m{i}" for i in range(len(metric_ids)))

    joins: List[str] = [
        "JOIN securities s ON s.id = fv.security_id",
        "JOIN metrics m ON m.metric_id = fv.metric_id",
    ]
    where_clauses: List[str] = [
        "fv.period = :as_of_date",
        f"fv.metric_id IN ({metric_placeholders})",
    ]

    params: Dict[str, object] = {"as_of_date": as_of_date}
    params.update({f"m{i}": mid for i, mid in enumerate(metric_ids)})

    if index_name:
        joins.append(
            "JOIN index_memberships im "
            "ON im.security_id = fv.security_id "
            "AND im.period = :as_of_date"
        )
        joins.append("JOIN indices idx ON idx.index_id = im.index_id")
        where_clauses.append("idx.name = :index_name")
        params["index_name"] = index_name

    if industry_name:
        joins.append("JOIN industries ind ON ind.id = s.industry_id")
        where_clauses.append("ind.name = :industry_name")
        params["industry_name"] = industry_name

    sql = text(
        f"""
        SELECT
            fv.security_id,
            s.ticker,
            fv.metric_id,
            m.metric_name AS metric_name,
            m.higher_is_better AS higher_is_better,
            fv.value
        FROM fundamental_values fv
        {' '.join(joins)}
        WHERE {' AND '.join(where_clauses)}
        """
    )

    df_long = pd.read_sql_query(sql, con=engine, params=params)

    if df_long.empty:
        raise ValueError(
            "No se han encontrado datos para esa fecha, métricas y filtros."
        )

    # Construimos el mapa de direccionalidad a partir de metrics.higher_is_better
    dir_rows = (
        df_long[["metric_name", "higher_is_better"]]
        .drop_duplicates(subset=["metric_name"])
        .copy()
    )
    direction_map: Dict[str, bool] = {}
    for _, row in dir_rows.iterrows():
        metric_name = str(row["metric_name"])
        hib_raw = row["higher_is_better"]
        # higher_is_better: 1 => True, 0 => False, None => True por defecto
        direction_map[metric_name] = bool(hib_raw) if hib_raw is not None else True

    # Pivot: filas = (security_id, ticker), columnas = metric_name
    df_wide = (
        df_long.pivot_table(
            index=["security_id", "ticker"],
            columns="metric_name",
            values="value",
        )
        .sort_index()
    )
    df_wide.columns = [str(c) for c in df_wide.columns]
    metric_cols = df_wide.columns.tolist()

    # Winsorization 1% - 99%
    df_wins = df_wide.copy()
    for col in metric_cols:
        series = df_wins[col]
        lower = series.quantile(0.01)
        upper = series.quantile(0.99)
        df_wins[col] = series.clip(lower=lower, upper=upper)

    # Cálculo de z-score + relleno de NaN con la media
    for col in metric_cols:
        col_series = df_wins[col]
        mean = col_series.mean()
        std = col_series.std(ddof=0)

        col_filled = col_series.fillna(mean)

        if std == 0 or pd.isna(std):
            z_scores = pd.Series(0.0, index=df_wins.index)
        else:
            z_scores = (col_filled - mean) / std

        df_wins[f"{col}_zscore"] = z_scores

    return df_wins, direction_map


# ---------------------------------------------------------------------------
# 2. Combinación lineal (weighted average) con direccionalidad
# ---------------------------------------------------------------------------
def apply_weighted_average(
    df_z: pd.DataFrame,
    weights: Dict[str, float],
    direction_map: Dict[str, bool],
    zsuffix: str = "_zscore",
    out_col: str = "score_linear",
) -> pd.DataFrame:
    """
    Calcula un score lineal:

        Score = sum_i ( sign_i * peso_i * z_i )

    donde sign_i = +1 si higher_is_better, -1 en caso contrario.
    """
    if not weights:
        raise ValueError("El diccionario de pesos no puede estar vacío.")

    df = df_z.copy()
    score = pd.Series(0.0, index=df.index, dtype="float64")

    for metric_name, w in weights.items():
        col_name = f"{metric_name}{zsuffix}"
        if col_name not in df.columns:
            raise KeyError(
                f"No se encuentra la columna de z-score '{col_name}' en el DataFrame."
            )

        sign = 1.0 if direction_map.get(metric_name, True) else -1.0
        score += sign * float(w) * df[col_name].fillna(0.0)

    df[out_col] = score
    return df


# ---------------------------------------------------------------------------
# 3. Combinación tipo "softplus product" con direccionalidad
# ---------------------------------------------------------------------------
def _softplus(x: np.ndarray) -> np.ndarray:
    """
    Implementación numéricamente estable de softplus(x) = ln(1 + e^x).
    """
    x_clipped = np.clip(x, -50, 50)
    return np.log1p(np.exp(x_clipped))


def apply_softplus_product(
    df_z: pd.DataFrame,
    weights: Dict[str, float],
    direction_map: Dict[str, bool],
    zsuffix: str = "_zscore",
    out_col: str = "score_softplus",
) -> pd.DataFrame:
    """
    Calcula el score:

        ln(Score) = sum_i ( sign_i * peso_i * ln(softplus(z_i)) )

    donde sign_i = +1 si higher_is_better, -1 en caso contrario.

    Equivalente a:

        Score = prod_i ( softplus(z_i) ** (sign_i * peso_i) )
    """
    if not weights:
        raise ValueError("El diccionario de pesos no puede estar vacío.")

    df = df_z.copy()
    log_score = pd.Series(0.0, index=df.index, dtype="float64")

    for metric_name, w in weights.items():
        col_name = f"{metric_name}{zsuffix}"
        if col_name not in df.columns:
            raise KeyError(
                f"No se encuentra la columna de z-score '{col_name}' en el DataFrame."
            )

        sign = 1.0 if direction_map.get(metric_name, True) else -1.0

        z = df[col_name].fillna(0.0).to_numpy(dtype="float64")
        s = _softplus(z)
        s = np.where(s <= 0, 1e-12, s)

        log_score += sign * float(w) * np.log(s)

    df[out_col] = np.exp(log_score)
    return df


# ---------------------------------------------------------------------------
# 4. Utilidad de exportación a Excel
# ---------------------------------------------------------------------------
def export_to_excel(df: pd.DataFrame, filepath: str, index: bool = True) -> None:
    """Guarda un DataFrame en un fichero Excel."""
    df.to_excel(filepath, index=index)


# ---------------------------------------------------------------------------
# 5. CLI interactivo
# ---------------------------------------------------------------------------
def _prompt_non_empty(prompt: str) -> str:
    value = input(prompt).strip()
    while not value:
        value = input("Valor obligatorio. Intenta de nuevo: ").strip()
    return value


def _parse_metric_ids(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Debes introducir al menos un metric_id.")
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError("Los metric_id deben ser enteros separados por comas.") from exc


def _prompt_float(prompt: str, default: float) -> float:
    raw = input(prompt).strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print("Valor no válido, usando el valor por defecto.")
        return default


def main() -> None:
    print("=== Ranking de acciones (Analytics CLI) ===")

    as_of_date = _prompt_non_empty("Periodo (ej. 2024 Q4): ")

    index_name_raw = input(
        "Nombre del índice (ej. S&P 500, vacío = sin filtro índice): "
    ).strip()
    index_name = index_name_raw or None

    industry_name_raw = input(
        "Nombre de la industria (vacío = sin filtro industria): "
    ).strip()
    industry_name = industry_name_raw or None

    metric_ids_str = _prompt_non_empty(
        "IDs de métricas (ej. 1,2,3) según 'metrics.metric_id': "
    )
    try:
        metric_ids = _parse_metric_ids(metric_ids_str)
    except ValueError as exc:
        print(f"Error en metric_ids: {exc}")
        return

    try:
        df_z, direction_map = get_zscore_ranking(
            as_of_date=as_of_date,
            metric_ids=metric_ids,
            index_name=index_name,
            industry_name=industry_name,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\nError al obtener z-scores: {exc}")
        return

    # Preguntar pesos para cada métrica disponible
    metric_names = sorted(direction_map.keys())
    print("\nMétricas disponibles para ponderar:")
    for name in metric_names:
        direction = "↑ mejor" if direction_map.get(name, True) else "↓ mejor"
        print(f"  - {name} ({direction})")

    print("\nIntroduce los pesos para cada métrica (ENTER = 1.0, 0 = ignorar):")
    weights: Dict[str, float] = {}
    for name in metric_names:
        w = _prompt_float(f"  Peso para {name}: ", default=1.0)
        if w != 0.0:
            weights[name] = w

    if not weights:
        print("No se ha definido ningún peso distinto de cero. Abortando.")
        return

    # Elegir método
    print("\nMétodo de combinación:")
    print("  1) Lineal")
    print("  2) Softplus")
    method = ""
    while method not in {"1", "2"}:
        method = input("Selecciona 1 o 2: ").strip()

    if method == "1":
        df_scored = apply_weighted_average(
            df_z=df_z,
            weights=weights,
            direction_map=direction_map,
            out_col="score",
        )
    else:
        df_scored = apply_softplus_product(
            df_z=df_z,
            weights=weights,
            direction_map=direction_map,
            out_col="score",
        )

    df_ranked = df_scored.sort_values("score", ascending=False)

    # Exportar a Excel
    period_safe = as_of_date.replace(" ", "")
    index_part = index_name.replace(" ", "") if index_name else "ALL"
    filename = f"Ranking_{period_safe}_{index_part}.xlsx"

    export_to_excel(df_ranked, filename, index=True)
    print(f"\nRanking generado y guardado en: {filename}")
    print("Primeras filas:")
    print(df_ranked.head())


if __name__ == "__main__":
    main()