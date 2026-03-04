import os
import re
import sys
import pandas as pd
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Float,
    ForeignKey, UniqueConstraint, select, delete
)
from sqlalchemy.engine import Engine

# --- CONFIGURACIÓN ---
DB_URL = "sqlite:///financial_data.db"
# Nombre exacto que salió en tu comando 'dir'
ARCHIVO_POR_DEFECTO = "2024 Q4.xlsx"

# Columnas que identifican a la empresa (las demás son métricas)
FIXED_COLUMNS = [
    "Ticker",
    "Long Name",
    "GICS Sector Name",
    "GICS Industry Group Name",
    "Market Cap (USD)",
]


# --- FUNCIONES ---
def get_period_from_name(filename: str) -> str:
    """Intenta sacar '2024 Q4' del nombre del archivo."""
    base = os.path.basename(filename)
    # Busca 4 digitos + Q + numero (ej: 2024 Q4)
    match = re.search(r"(\d{4}\s*Q[1-4])", base, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "DESCONOCIDO"


def init_db():
    """Crea la base de datos y las tablas si no existen."""
    engine = create_engine(DB_URL)
    metadata = MetaData()

    # Tabla de Métricas (metric_id, metric_name)
    metrics = Table(
        "metrics",
        metadata,
        Column("metric_id", Integer, primary_key=True, autoincrement=True),
        Column("metric_name", String, unique=True, nullable=False),
    )

    # Tabla de Sectores (sector_id, sector_name)
    sectors = Table(
        "sectors",
        metadata,
        Column("sector_id", Integer, primary_key=True, autoincrement=True),
        Column("sector_name", String, unique=True, nullable=False),
    )

    # Tabla de Industrias (industry_id, industry_name)
    industries = Table(
        "industries",
        metadata,
        Column("industry_id", Integer, primary_key=True, autoincrement=True),
        Column("industry_name", String, unique=True, nullable=False),
    )

    # Tabla 1: Empresas
    securities = Table(
        "securities",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("ticker", String, unique=True, nullable=False),
        Column("long_name", String),
        Column("sector_id", Integer, ForeignKey("sectors.sector_id")),
        Column("industry_id", Integer, ForeignKey("industries.industry_id")),
    )

    # Tabla 2: Valores Fundamentales (Formato Largo)
    fundamentals = Table(
        "fundamental_values",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("security_id", Integer, ForeignKey("securities.id"), nullable=False),
        Column("metric_id", Integer, ForeignKey("metrics.metric_id"), nullable=False),
        Column("value", Float),
        Column("period", String, nullable=False),
    )

    metadata.create_all(engine)
    return engine, securities, fundamentals, sectors, industries, metrics


def read_data(filepath):
    """Lee Excel o CSV de forma inteligente."""
    print(f"Leyendo archivo: {filepath} ...")

    # 1. Intentar como Excel
    try:
        # header=3 significa que los títulos están en la fila 4
        return pd.read_excel(filepath, header=3)
    except Exception as e_excel:
        print(f"No se pudo leer como Excel ({e_excel}). Probando como CSV...")

    # 2. Intentar como CSV (Plan B)
    try:
        return pd.read_csv(filepath, header=3)
    except Exception as e_csv:
        raise ValueError(f"CRÍTICO: No se pudo leer el archivo ni como Excel ni como CSV.")


def process_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo NO está en esta carpeta: {filepath}")

    # 1. Leer y Limpiar
    df = read_data(filepath)

    # Validar que tiene columna Ticker
    if "Ticker" not in df.columns:
        raise ValueError(f"El archivo no tiene la columna 'Ticker'. Columnas encontradas: {list(df.columns)}")

    # Eliminar filas vacías
    df = df.dropna(subset=["Ticker"])
    period = get_period_from_name(filepath)
    print(f"--> Periodo detectado: {period}")

    # 2. Conectar a Base de Datos
    engine, tbl_sec, tbl_fund, tbl_sector, tbl_industry, tbl_metric = init_db()

    # 3. Insertar Empresas (Upsert manual) + sectores/industrias normalizados
    # Seleccionamos solo las columnas de info de empresa
    companies = df[FIXED_COLUMNS].copy().drop_duplicates(subset=["Ticker"])

    print(f"Procesando {len(companies)} empresas...")
    ticker_map = {}      # Diccionario para guardar ticker -> id
    sector_cache = {}    # sector_name -> sector_id
    industry_cache = {}  # industry_name -> industry_id

    # OJO: usamos iterrows() para poder acceder a las columnas por nombre exacto
    # (con itertuples los espacios se convierten en guiones bajos y fallaban los getattr)
    with engine.begin() as conn:
        for _, row in companies.iterrows():
            ticker_val = row["Ticker"]
            long_name_val = row.get("Long Name")
            sector_name = row.get("GICS Sector Name")
            industry_name = row.get("GICS Industry Group Name")

            # --- Resolver / crear sector_id ---
            sector_id = None
            if sector_name:
                if sector_name in sector_cache:
                    sector_id = sector_cache[sector_name]
                else:
                    res_sector = conn.execute(
                        select(tbl_sector.c.sector_id).where(
                            tbl_sector.c.sector_name == sector_name
                        )
                    ).first()
                    if res_sector:
                        sector_id = res_sector[0]
                    else:
                        ins_sector = conn.execute(
                            tbl_sector.insert().values(sector_name=sector_name)
                        )
                        sector_id = ins_sector.inserted_primary_key[0]
                    sector_cache[sector_name] = sector_id

            # --- Resolver / crear industry_id ---
            industry_id = None
            if industry_name:
                if industry_name in industry_cache:
                    industry_id = industry_cache[industry_name]
                else:
                    res_ind = conn.execute(
                        select(tbl_industry.c.industry_id).where(
                            tbl_industry.c.industry_name == industry_name
                        )
                    ).first()
                    if res_ind:
                        industry_id = res_ind[0]
                    else:
                        ins_ind = conn.execute(
                            tbl_industry.insert().values(industry_name=industry_name)
                        )
                        industry_id = ins_ind.inserted_primary_key[0]
                    industry_cache[industry_name] = industry_id

            # Buscar si existe la security
            res = conn.execute(
                select(tbl_sec.c.id, tbl_sec.c.long_name, tbl_sec.c.sector_id, tbl_sec.c.industry_id).where(
                    tbl_sec.c.ticker == ticker_val
                )
            ).first()

            if res:
                # Ya existe: podemos completar datos que estén a NULL
                sec_id = res[0]
                update_values = {}
                if long_name_val and (res[1] is None or res[1] == ""):
                    update_values["long_name"] = long_name_val
                if sector_id is not None and res[2] is None:
                    update_values["sector_id"] = sector_id
                if industry_id is not None and res[3] is None:
                    update_values["industry_id"] = industry_id

                if update_values:
                    conn.execute(
                        tbl_sec.update()
                        .where(tbl_sec.c.id == sec_id)
                        .values(**update_values)
                    )
            else:
                # Insertar nueva security enlazada a sector/industry por ID
                ins = conn.execute(
                    tbl_sec.insert().values(
                        ticker=ticker_val,
                        long_name=long_name_val,
                        sector_id=sector_id,
                        industry_id=industry_id,
                    )
                )
                sec_id = ins.inserted_primary_key[0]

            ticker_map[ticker_val] = sec_id

    # 4. Transformar Métricas (De Ancho a Largo)
    # Todas las columnas que NO son fijas, son métricas
    metric_cols = [c for c in df.columns if c not in FIXED_COLUMNS and not c.startswith("Unnamed")]

    print(f"Transformando {len(metric_cols)} tipos de métricas...")

    # 'Melt' convierte las columnas en filas
    df_long = df.melt(
        id_vars=["Ticker"],
        value_vars=metric_cols,
        var_name="metric_name",
        value_name="value"
    )

    # Asignar el ID numérico de la security usando el mapa que creamos antes
    df_long["security_id"] = df_long["Ticker"].map(ticker_map)
    df_long["period"] = period

    # Mapear metric_name -> metric_id usando la tabla metrics
    metric_cache = {}
    with engine.begin() as conn:
        unique_metrics = df_long["metric_name"].dropna().unique().tolist()
        for m_name in unique_metrics:
            if m_name in metric_cache:
                continue
            res_metric = conn.execute(
                select(tbl_metric.c.metric_id).where(
                    tbl_metric.c.metric_name == m_name
                )
            ).first()
            if res_metric:
                m_id = res_metric[0]
            else:
                ins_metric = conn.execute(
                    tbl_metric.insert().values(metric_name=m_name)
                )
                m_id = ins_metric.inserted_primary_key[0]
            metric_cache[m_name] = m_id

    df_long["metric_id"] = df_long["metric_name"].map(metric_cache)

    # Limpiar datos
    df_long = df_long.dropna(subset=["security_id", "value", "metric_id"])  # Borrar si falta algo clave
    df_long["value"] = pd.to_numeric(df_long["value"], errors='coerce')  # Asegurar que son números

    # 5. Guardar en SQL
    print(f"Guardando {len(df_long)} registros en la base de datos...")

    with engine.begin() as conn:
        # Borrar datos anteriores de ESTE periodo para no duplicar
        conn.execute(delete(tbl_fund).where(tbl_fund.c.period == period))

    # Insertar masivo (es mucho más rápido)
    data_to_insert = df_long[["security_id", "metric_id", "value", "period"]]
    data_to_insert.to_sql("fundamental_values", engine, if_exists="append", index=False)

    print("¡ÉXITO TOTAL! Datos importados correctamente.")


# --- PUNTO DE ENTRADA ---
if __name__ == "__main__":
    # Si le pasas un archivo por comando lo usa, si no, usa el de defecto
    archivo = sys.argv[1] if len(sys.argv) > 1 else ARCHIVO_POR_DEFECTO

    try:
        process_file(archivo)
    except Exception as e:
        print("\n--- ERROR ---")
        print(e)
        print("-------------")
        # Pausa para que leas el error si se cierra rápido
        input("Presiona ENTER para salir...")