"""
SQLAlchemy table definitions and database initialization.
"""

from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    ForeignKey,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine

from modules.config import DB_URL


def create_tables(engine: Engine) -> None:
    """Create all tables if they do not exist."""
    metadata = MetaData()

    metrics = Table(
        "metrics",
        metadata,
        Column("metric_id", Integer, primary_key=True, autoincrement=True),
        Column("metric_name", String, unique=True, nullable=False),
        Column("description", String, nullable=True),
        Column("higher_is_better", Boolean, nullable=True),
        Column("n/a treatment", String, nullable=True),
    )

    sectors = Table(
        "sectors",
        metadata,
        Column("sector_id", Integer, primary_key=True, autoincrement=True),
        Column("sector_name", String, unique=True, nullable=False),
    )

    industries = Table(
        "industries",
        metadata,
        Column("industry_id", Integer, primary_key=True, autoincrement=True),
        Column("industry_name", String, unique=True, nullable=False),
    )

    securities = Table(
        "securities",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("ticker", String, unique=True, nullable=False),
        Column("long_name", String),
        Column("sector_id", Integer, ForeignKey("sectors.sector_id")),
        Column("industry_id", Integer, ForeignKey("industries.industry_id")),
    )

    fundamentals = Table(
        "fundamental_values",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("security_id", Integer, ForeignKey("securities.id"), nullable=False),
        Column("metric_id", Integer, ForeignKey("metrics.metric_id"), nullable=False),
        Column("value", Float),
        Column("period", String, nullable=False),
    )

    indices = Table(
        "indices",
        metadata,
        Column("index_id", Integer, primary_key=True, autoincrement=True),
        Column("name", String, unique=True, nullable=False),
    )

    index_membership = Table(
        "index_membership",
        metadata,
        Column("index_id", Integer, ForeignKey("indices.index_id"), nullable=False),
        Column("security_id", Integer, ForeignKey("securities.id"), nullable=False),
        Column("period", String, nullable=False),
    )

    metadata.create_all(engine)

    # Migrate: add higher_is_better if table existed without it
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE metrics ADD COLUMN higher_is_better BOOLEAN"))
            conn.commit()
        except Exception:
            conn.rollback()
