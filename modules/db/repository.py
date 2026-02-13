"""
Repository for financial data persistence.
Encapsulates all database operations.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, select, delete, update, and_, MetaData, Table
from sqlalchemy.engine import Engine

from modules.config import DB_URL, FIXED_COLUMNS
from modules.db.schema import create_tables
from modules.models import ImportResult


class FinancialDatabase:
    """
    Repository for storing and querying financial fundamental data.
    Handles normalization of sectors, industries, metrics, and index membership.
    """

    def __init__(self, db_url: str = DB_URL) -> None:
        self._engine = create_engine(db_url)
        create_tables(self._engine)
        self._metadata = MetaData()
        self._metadata.reflect(bind=self._engine)

    @property
    def engine(self) -> Engine:
        """Expose engine for read-only queries (e.g. analytics)."""
        return self._engine

    def _get_table(self, name: str) -> Table:
        return self._metadata.tables[name]

    def get_metric_ids_by_names(self, metric_names: list[str]) -> dict[str, int]:
        """
        Resolve metric names to metric_ids.
        Raises ValueError if any name is not found in the database.
        """
        if not metric_names:
            return {}
        tbl = self._get_table("metrics")
        result = {}
        with self._engine.connect() as conn:
            for name in metric_names:
                row = conn.execute(
                    select(tbl.c.metric_id).where(tbl.c.metric_name == name)
                ).first()
                if not row:
                    raise ValueError(
                        f"Metric '{name}' not found in database. "
                        "Ensure the metric exists (import data first)."
                    )
                result[name] = row[0]
        return result

    def list_metrics(self) -> list[dict]:
        """
        Return all metrics in the database.
        Each dict has: metric_id, metric_name, higher_is_better.
        """
        tbl = self._get_table("metrics")
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(
                    tbl.c.metric_id,
                    tbl.c.metric_name,
                    tbl.c.higher_is_better,
                )
            ).fetchall()

        return [
            {
                "metric_id": r[0],
                "metric_name": r[1],
                "higher_is_better": bool(r[2]) if r[2] is not None else None,
            }
            for r in rows
        ]

    def update_metric_higher_is_better(
        self, metric_id: int, higher_is_better: bool
    ) -> None:
        """
        Update the higher_is_better column for a metric.
        Raises ValueError if metric_id does not exist.
        """
        tbl = self._get_table("metrics")
        with self._engine.begin() as conn:
            exists = conn.execute(
                select(tbl.c.metric_id).where(tbl.c.metric_id == metric_id)
            ).first()
            if not exists:
                raise ValueError(f"Metric with id {metric_id} not found.")
            conn.execute(
                update(tbl).where(tbl.c.metric_id == metric_id).values(
                    higher_is_better=higher_is_better
                )
            )

    def save_fundamentals(
        self,
        df: pd.DataFrame,
        period: str,
        index_code: Optional[str] = None,
    ) -> ImportResult:
        """
        Save a DataFrame of fundamental data to the database.
        Normalizes sectors, industries, metrics; upserts securities;
        optionally registers index membership.
        """
        if "Ticker" not in df.columns:
            raise ValueError(
                f"DataFrame must have 'Ticker' column. Found: {list(df.columns)}"
            )

        tbl_sec = self._get_table("securities")
        tbl_fund = self._get_table("fundamental_values")
        tbl_sector = self._get_table("sectors")
        tbl_industry = self._get_table("industries")
        tbl_metric = self._get_table("metrics")
        tbl_indices = self._get_table("indices")
        tbl_index_membership = self._get_table("index_membership")

        companies = df[FIXED_COLUMNS].copy().drop_duplicates(subset=["Ticker"])
        ticker_map: dict[str, int] = {}
        sector_cache: dict[str, int] = {}
        industry_cache: dict[str, int] = {}

        with self._engine.begin() as conn:
            for _, row in companies.iterrows():
                ticker_val = row["Ticker"]
                long_name_val = row.get("Long Name")
                sector_name = row.get("GICS Sector Name")
                industry_name = row.get("GICS Industry Group Name")

                sector_id = self._resolve_sector(
                    conn, tbl_sector, sector_name, sector_cache
                )
                industry_id = self._resolve_industry(
                    conn, tbl_industry, industry_name, industry_cache
                )

                sec_id = self._upsert_security(
                    conn,
                    tbl_sec,
                    ticker_val,
                    long_name_val,
                    sector_id,
                    industry_id,
                )
                ticker_map[ticker_val] = sec_id

        if index_code and str(index_code).strip():
            code = str(index_code).strip()
            self._update_index_membership(
                tbl_indices,
                tbl_index_membership,
                code,
                period,
                list(ticker_map.values()),
            )

        metric_cols = [
            c
            for c in df.columns
            if c not in FIXED_COLUMNS and not str(c).startswith("Unnamed")
        ]

        df_long = df.melt(
            id_vars=["Ticker"],
            value_vars=metric_cols,
            var_name="metric_name",
            value_name="value",
        )
        df_long["security_id"] = df_long["Ticker"].map(ticker_map)
        df_long["period"] = period

        metric_cache: dict[str, int] = {}
        with self._engine.begin() as conn:
            for m_name in df_long["metric_name"].dropna().unique().tolist():
                if m_name not in metric_cache:
                    metric_cache[m_name] = self._resolve_metric(
                        conn, tbl_metric, m_name
                    )

        df_long["metric_id"] = df_long["metric_name"].map(metric_cache)
        df_long = df_long.dropna(subset=["security_id", "value", "metric_id"])
        df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

        with self._engine.begin() as conn:
            conn.execute(delete(tbl_fund).where(tbl_fund.c.period == period))
            data = df_long[["security_id", "metric_id", "value", "period"]]
            data.to_sql(
                "fundamental_values",
                con=conn,
                if_exists="append",
                index=False,
            )

        return ImportResult(
            period=period,
            companies_count=len(ticker_map),
            metrics_count=len(metric_cols),
            records_count=len(df_long),
            index_code=index_code,
        )

    def _resolve_sector(self, conn, tbl, name, cache):
        if not name:
            return None
        if name in cache:
            return cache[name]
        row = conn.execute(
            select(tbl.c.sector_id).where(tbl.c.sector_name == name)
        ).first()
        if row:
            cache[name] = row[0]
            return row[0]
        ins = conn.execute(tbl.insert().values(sector_name=name))
        cache[name] = ins.inserted_primary_key[0]
        return cache[name]

    def _resolve_industry(self, conn, tbl, name, cache):
        if not name:
            return None
        if name in cache:
            return cache[name]
        row = conn.execute(
            select(tbl.c.industry_id).where(tbl.c.industry_name == name)
        ).first()
        if row:
            cache[name] = row[0]
            return row[0]
        ins = conn.execute(tbl.insert().values(industry_name=name))
        cache[name] = ins.inserted_primary_key[0]
        return cache[name]

    def _resolve_metric(self, conn, tbl, name):
        row = conn.execute(
            select(tbl.c.metric_id).where(tbl.c.metric_name == name)
        ).first()
        if row:
            return row[0]
        ins = conn.execute(tbl.insert().values(metric_name=name))
        return ins.inserted_primary_key[0]

    def _upsert_security(
        self,
        conn,
        tbl,
        ticker: str,
        long_name,
        sector_id,
        industry_id,
    ) -> int:
        row = conn.execute(
            select(
                tbl.c.id,
                tbl.c.long_name,
                tbl.c.sector_id,
                tbl.c.industry_id,
            ).where(tbl.c.ticker == ticker)
        ).first()

        if row:
            sec_id = row[0]
            updates = {}
            if long_name and (row[1] is None or row[1] == ""):
                updates["long_name"] = long_name
            if sector_id is not None and row[2] is None:
                updates["sector_id"] = sector_id
            if industry_id is not None and row[3] is None:
                updates["industry_id"] = industry_id
            if updates:
                conn.execute(tbl.update().where(tbl.c.id == sec_id).values(**updates))
            return sec_id

        ins = conn.execute(
            tbl.insert().values(
                ticker=ticker,
                long_name=long_name,
                sector_id=sector_id,
                industry_id=industry_id,
            )
        )
        return ins.inserted_primary_key[0]

    def _update_index_membership(
        self,
        tbl_indices,
        tbl_membership,
        code: str,
        period: str,
        security_ids: list[int],
    ) -> None:
        with self._engine.begin() as conn:
            row = conn.execute(
                select(tbl_indices.c.index_id).where(tbl_indices.c.name == code)
            ).first()
            if row:
                index_id = row[0]
            else:
                ins = conn.execute(tbl_indices.insert().values(name=code))
                index_id = ins.inserted_primary_key[0]

            conn.execute(
                delete(tbl_membership).where(
                    and_(
                        tbl_membership.c.index_id == index_id,
                        tbl_membership.c.period == period,
                    )
                )
            )
            for sec_id in security_ids:
                conn.execute(
                    tbl_membership.insert().values(
                        index_id=index_id,
                        security_id=sec_id,
                        period=period,
                    )
                )
