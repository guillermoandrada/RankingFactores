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

# Conexión a la base de datos
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
    Calcula z-scores por métrica para cada acción en una fecha dada.
    """
    if not metric_ids:
        raise ValueError("metric_ids no puede estar vacío.")

    metric_placeholders = ", ".join(f":m{i}" for i in range(len(metric_ids)))

    # JOINS BÁSICOS
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

    # FILTRO OPCIONAL: ÍNDICE
    if index_name:
        joins.append(
            "JOIN index_membership im "  # <-- CORREGIDO: sin la 's'
            "ON im.security_id = fv.security_id "
            "AND im.period = :as_of_date"
        )
        joins.append("JOIN indices idx ON idx.index_id = im.index_id")
        where_clauses.append("idx.name = :index_name")
        params["index_name"] = index_name

    # FILTRO OPCIONAL: INDUSTRIA
    if industry_name:
        joins.append("JOIN industries ind ON ind.industry_id = s.industry_id")
        where_clauses.append("ind.industry_name = :industry_name")
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
            f"No se han encontrado datos para esa fecha ({as_of_date}), métricas ({metric_ids}) y filtros."
        )

    # Construimos el mapa de direccionalidad
    dir_rows = (
        df_long[["metric_name", "higher_is_better"]]
        .drop_duplicates(subset=["metric_name"])
        .copy()
    )
    direction_map: Dict[str, bool] = {}
    for _, row in dir_rows.iterrows():
        metric_name = str(row["metric_name"])
        hib_raw = row["higher_is_better"]
        # 1 => True, 0 => False, None => True por defecto
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

    # Cálculo de z-score
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
    Score = sum ( sign * peso * z_score )
    """
    if not weights:
        raise ValueError("El diccionario de pesos no puede estar vacío.")

    df = df_z.copy()
    score = pd.Series(0.0, index=df.index, dtype="float64")

    for metric_name, w in weights.items():
        col_name = f"{metric_name}{zsuffix}"
        # Si la métrica existe en el df
        if col_name in df.columns:
            # Si higher_is_better es True -> +1, si es False -> -1
            sign = 1.0 if direction_map.get(metric_name, True) else -1.0
            score += sign * float(w) * df[col_name].fillna(0.0)

    df[out_col] = score
    return df


# ---------------------------------------------------------------------------
# 3. Combinación tipo "softplus product" con direccionalidad
# ---------------------------------------------------------------------------
def _softplus(x: np.ndarray) -> np.ndarray:
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
    ln(Score) = sum ( sign * peso * ln(softplus(z)) )
    """
    if not weights:
        raise ValueError("El diccionario de pesos no puede estar vacío.")

    df = df_z.copy()
    log_score = pd.Series(0.0, index=df.index, dtype="float64")

    for metric_name, w in weights.items():
        col_name = f"{metric_name}{zsuffix}"
        if col_name in df.columns:
            sign = 1.0 if direction_map.get(metric_name, True) else -1.0
            z = df[col_name].fillna(0.0).to_numpy(dtype="float64")
            s = _softplus(z)
            s = np.where(s <= 0, 1e-12, s)

            log_score += sign * float(w) * np.log(s)

    df[out_col] = np.exp(log_score)
    return df


# ---------------------------------------------------------------------------
# 4. Utilidad de exportación
# ---------------------------------------------------------------------------
def export_to_excel(df: pd.DataFrame, filepath: str, index: bool = True) -> None:
    df.to_excel(filepath, index=index)


# ---------------------------------------------------------------------------
# 5. Pipeline principal (preparado para API)
# ---------------------------------------------------------------------------
def run_analytics_pipeline(payload: Dict) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de análisis de rankings a partir de un diccionario
    de configuración (payload).
    """
    # Validación de campos obligatorios
    required_fields = ["as_of_date", "metric_ids", "weights", "method"]
    for field in required_fields:
        if field not in payload:
            raise ValueError(f"Campo obligatorio faltante en payload: '{field}'")

    as_of_date = payload["as_of_date"]
    metric_ids = payload["metric_ids"]
    weights = payload["weights"]
    method = payload["method"].lower()
    index_name = payload.get("index_name")
    industry_name = payload.get("industry_name")
    export_excel = payload.get("export_excel", False)

    # Validación de tipos
    if not isinstance(metric_ids, list) or not all(isinstance(x, int) for x in metric_ids):
        raise ValueError("'metric_ids' debe ser una lista de enteros.")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("'weights' debe ser un diccionario no vacío.")
    if method not in ("linear", "softplus"):
        raise ValueError("'method' debe ser 'linear' o 'softplus'.")

    # Paso 1: Obtener z-scores y direction_map
    df_z, direction_map = get_zscore_ranking(
        as_of_date=as_of_date,
        metric_ids=metric_ids,
        index_name=index_name,
        industry_name=industry_name,
    )

    # Paso 2: Aplicar método de combinación según el método seleccionado
    if method == "linear":
        df_scored = apply_weighted_average(
            df_z=df_z,
            weights=weights,
            direction_map=direction_map,
            out_col="score",
        )
    else:  # method == "softplus"
        df_scored = apply_softplus_product(
            df_z=df_z,
            weights=weights,
            direction_map=direction_map,
            out_col="score",
        )

    # Paso 3: Ordenar por score (descendente)
    df_ranked = df_scored.sort_values("score", ascending=False)

    # Paso 4: Exportar a Excel si se solicita
    if export_excel:
        period_safe = as_of_date.replace(" ", "")
        index_part = index_name.replace(" ", "") if index_name else "ALL"
        filename = f"Ranking_{period_safe}_{index_part}.xlsx"
        export_to_excel(df_ranked, filename, index=True)

    return df_ranked


# ---------------------------------------------------------------------------
# 6. Ejemplo de uso local (para testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Payload actualizado con los datos reales de la base de datos
    example_payload = {
        "as_of_date": "2024 Q4",
        "index_name": "B500",
        "industry_name": None,
        "metric_ids": [1, 4],
        "weights": {
            "Operating Margin": 0.5,
            "Current Profit Margin": 0.5
        },
        "method": "linear",
        "export_excel": True
    }

    try:
        result_df = run_analytics_pipeline(example_payload)
        print("✅ Pipeline ejecutado correctamente")
        print("\nPrimeras 10 filas del ranking:")

        # Filtramos las columnas que queremos ver para que quede más limpio en la consola
        cols_to_print = ['ticker', 'Operating Margin', 'Current Profit Margin', 'score']
        # Nos aseguramos de imprimir solo las columnas que realmente existen en el DataFrame
        existing_cols = [c for c in cols_to_print if c in result_df.columns]

        print(result_df[existing_cols].head(10))
        print(f"\nTotal de filas: {len(result_df)}")
        print("Archivo Excel generado con éxito.")
    except Exception as exc:
        print(f"❌ Error durante la ejecución: {exc}")
        raise