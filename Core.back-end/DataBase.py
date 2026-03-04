import os
import pandas as pd

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    ForeignKey,
    select,
    delete,
    text,
    and_,
)
from sqlalchemy.engine import Engine

# --- CONFIGURACIÓN DE BASE DE DATOS ---

DB_URL = "sqlite:///financial_data.db"

# Columnas que identifican a la empresa (las demás son métricas)
FIXED_COLUMNS = [
    "Ticker",
    "Long Name",
    "GICS Sector Name",
    "GICS Industry Group Name",
    "Market Cap (USD)",
]


# --- DEFINICIÓN DE TABLAS E INICIALIZACIÓN ---


def init_db() -> tuple[Engine, Table, Table, Table, Table, Table, Table, Table]:
    """Crea la base de datos y las tablas si no existen."""
    engine = create_engine(DB_URL)
    metadata = MetaData()

    # Tabla de Métricas (metric_id, metric_name, higher_is_better)
    metrics = Table(
        "metrics",
        metadata,
        Column("metric_id", Integer, primary_key=True, autoincrement=True),
        Column("metric_name", String, unique=True, nullable=False),
        # Nuevo campo: lo rellenarás tú a mano después (True/False)
        Column("higher_is_better", Boolean, nullable=True),
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

    # Nueva tabla: Índices (Index ID, Name)
    # Aquí el Name será el código que viene del Excel (ej. B500)
    indices = Table(
        "indices",
        metadata,
        Column("index_id", Integer, primary_key=True, autoincrement=True),
        Column("name", String, unique=True, nullable=False),
    )

    # Nueva tabla: pertenecía a índices por periodo
    # Usamos 'period' (ej. '2024 Q4') en lugar de fechas
    index_membership = Table(
        "index_membership",
        metadata,
        Column("index_id", Integer, ForeignKey("indices.index_id"), nullable=False),
        Column("security_id", Integer, ForeignKey("securities.id"), nullable=False),
        Column("period", String, nullable=False),
        # Podrías añadir un UniqueConstraint aquí si quisieras evitar duplicados a nivel SQL
        # UniqueConstraint("index_id", "security_id", "period", name="uq_index_membership"),
    )

    metadata.create_all(engine)

    # Si la BD ya existía sin la columna higher_is_better, la añadimos vía ALTER TABLE
    with engine.connect() as conn:
        try:
            conn.execute(
                text("ALTER TABLE metrics ADD COLUMN higher_is_better BOOLEAN")
            )
            conn.commit()
        except Exception:
            # Si falla (porque ya existe), lo ignoramos
            conn.rollback()

    return (
        engine,
        securities,
        fundamentals,
        sectors,
        industries,
        metrics,
        indices,
        index_membership,
    )


# --- FUNCIÓN PÚBLICA PARA GUARDAR DATAFRAME ---


def save_dataframe_to_sql(df: pd.DataFrame, period: str, index_code: str | None) -> None:
    """
    Recibe un DataFrame ya leído/validado y un periodo (por ejemplo '2024 Q4'),
    y guarda los datos en la base de datos SQL.

    - Usa FIXED_COLUMNS para separar la información de empresa de las métricas.
    - Normaliza sectores, industrias y métricas.
    - Hace 'melt' de las columnas de métricas.
    - Borra los datos del mismo periodo antes de insertar.
    - Si index_code no es None, crea/usa ese índice (ej. 'B500') y rellena la
      tabla index_membership para ese periodo.
    """
    if "Ticker" not in df.columns:
        raise ValueError(
            f"El DataFrame no tiene la columna 'Ticker'. Columnas encontradas: {list(df.columns)}"
        )

    # 1. Conectar a Base de Datos
    (
        engine,
        tbl_sec,
        tbl_fund,
        tbl_sector,
        tbl_industry,
        tbl_metric,
        tbl_indices,
        tbl_index_membership,
    ) = init_db()

    # 2. Insertar Empresas (Upsert manual) + sectores/industrias normalizados
    companies = df[FIXED_COLUMNS].copy().drop_duplicates(subset=["Ticker"])

    print(f"Procesando {len(companies)} empresas...")
    ticker_map: dict[str, int] = {}      # ticker -> id
    sector_cache: dict[str, int] = {}    # sector_name -> sector_id
    industry_cache: dict[str, int] = {}  # industry_name -> industry_id

    # OJO: usamos iterrows() para poder acceder a las columnas por nombre exacto
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
                select(
                    tbl_sec.c.id,
                    tbl_sec.c.long_name,
                    tbl_sec.c.sector_id,
                    tbl_sec.c.industry_id,
                ).where(tbl_sec.c.ticker == ticker_val)
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

    # 2b. Si tenemos código de índice desde el Excel, registrar pertenencia
    if index_code is not None and str(index_code).strip():
        code = str(index_code).strip()
        with engine.begin() as conn:
            # Crear / obtener índice
            res_idx = conn.execute(
                select(tbl_indices.c.index_id).where(tbl_indices.c.name == code)
            ).first()
            if res_idx:
                index_id = res_idx[0]
            else:
                ins_idx = conn.execute(
                    tbl_indices.insert().values(name=code)
                )
                index_id = ins_idx.inserted_primary_key[0]

            # Borrar membresía previa de este índice en este periodo (idempotente)
            conn.execute(
                delete(tbl_index_membership).where(
                    and_(
                        tbl_index_membership.c.index_id == index_id,
                        tbl_index_membership.c.period == period,
                    )
                )
            )

            # Insertar membresía para todas las securities del Excel
            for sec_id in ticker_map.values():
                conn.execute(
                    tbl_index_membership.insert().values(
                        index_id=index_id,
                        security_id=sec_id,
                        period=period,
                    )
                )

        print(
            f"Pertenencia al índice '{code}' actualizada para periodo {period} "
            f"({len(ticker_map)} securities)."
        )

    # 3. Transformar Métricas (De Ancho a Largo)
    metric_cols = [
        c for c in df.columns if c not in FIXED_COLUMNS and not str(c).startswith("Unnamed")
    ]

    print(f"Transformando {len(metric_cols)} tipos de métricas...")

    df_long = df.melt(
        id_vars=["Ticker"],
        value_vars=metric_cols,
        var_name="metric_name",
        value_name="value",
    )

    # Asignar el ID numérico de la security usando el mapa que creamos antes
    df_long["security_id"] = df_long["Ticker"].map(ticker_map)
    df_long["period"] = period

    # Mapear metric_name -> metric_id usando la tabla metrics
    metric_cache: dict[str, int] = {}
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
    df_long = df_long.dropna(subset=["security_id", "value", "metric_id"])
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

    # 4. Guardar en SQL
    print(f"Guardando {len(df_long)} registros en la base de datos...")

    with engine.begin() as conn:
        # Borrar datos anteriores de ESTE periodo para no duplicar
        conn.execute(delete(tbl_fund).where(tbl_fund.c.period == period))

        # Insertar masivo en la MISMA conexión/transacción
        data_to_insert = df_long[["security_id", "metric_id", "value", "period"]]
        data_to_insert.to_sql(
            "fundamental_values",
            con=conn,
            if_exists="append",
            index=False,
        )

    print("¡ÉXITO TOTAL! Datos importados correctamente.")