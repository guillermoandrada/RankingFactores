import pandas as pd
from sqlalchemy import create_engine, text

METRICAS_LOWER_IS_BETTER = [
    # ej: "P/E Ratio",
]

def configurar_metricas():
    engine = create_engine("sqlite:///financial_data.db")

    with engine.begin() as conn:
        # PASO 1: Poner 1 (True) a todo lo que esté vacío (NULL) por defecto
        query_default = text("UPDATE metrics SET higher_is_better = 1 WHERE higher_is_better IS NULL")
        res_default = conn.execute(query_default)
        print(f"-> {res_default.rowcount} métricas nuevas actualizadas a 1 por defecto.")

        # PASO 2: Forzar a 0 (False) las métricas que tú has puesto en la lista
        if METRICAS_LOWER_IS_BETTER:
            query_lower = text("UPDATE metrics SET higher_is_better = 0 WHERE metric_name = :nombre")

            for metrica in METRICAS_LOWER_IS_BETTER:
                res_lower = conn.execute(query_lower, {"nombre": metrica})
                if res_lower.rowcount > 0:
                    print(f"-> '{metrica}' forzada a 0 (Lower is better).")
                else:
                    print(f"-> Ojo: '{metrica}' no existe en la base de datos todavía.")


    df_metrics = pd.read_sql("SELECT * FROM metrics", engine)
    print(df_metrics.to_string(index=False))

if __name__ == "__main__":
    configurar_metricas()