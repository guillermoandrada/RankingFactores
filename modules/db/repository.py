"""
Repository for financial data persistence.
Encapsulates all database operations.
"""

from __future__ import annotations

import json
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

    def list_periods(self) -> list[str]:
        """Return all distinct periods in fundamental_values."""
        tbl = self._get_table("fundamental_values")
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(tbl.c.period).distinct().order_by(tbl.c.period)
            ).fetchall()
        return [r[0] for r in rows if r[0]]

    def get_period_content(self, period: str) -> dict:
        """Return securities with their metric values for a period (wide format)."""
        tbl_fund = self._get_table("fundamental_values")
        tbl_sec = self._get_table("securities")
        tbl_metric = self._get_table("metrics")
        with self._engine.connect() as conn:
            df = pd.read_sql_query(
                select(
                    tbl_fund.c.security_id,
                    tbl_fund.c.period,
                    tbl_fund.c.metric_id,
                    tbl_fund.c.value,
                    tbl_sec.c.ticker,
                    tbl_metric.c.metric_name,
                )
                .select_from(tbl_fund)
                .join(tbl_sec, tbl_sec.c.id == tbl_fund.c.security_id)
                .join(tbl_metric, tbl_metric.c.metric_id == tbl_fund.c.metric_id)
                .where(tbl_fund.c.period == period),
                con=conn,
            )
        if df.empty:
            return {"period": period, "metrics": [], "data": []}
        wide = df.pivot_table(
            index=["security_id", "ticker"],
            columns="metric_name",
            values="value",
        ).reset_index()
        metrics = list(wide.columns.drop(["security_id", "ticker"]))
        data = json.loads(wide.to_json(orient="records", date_format="iso"))
        return {
            "period": period,
            "metrics": metrics,
            "data": data,
        }

    def delete_period(self, period: str) -> None:
        """Delete all fundamental values, index membership, and classifications for a period."""
        tbl_fund = self._get_table("fundamental_values")
        tbl_membership = self._get_table("index_membership")
        tbl_classification = self._get_table("security_classification")
        with self._engine.begin() as conn:
            conn.execute(delete(tbl_fund).where(tbl_fund.c.period == period))
            conn.execute(
                delete(tbl_membership).where(tbl_membership.c.period == period)
            )
            conn.execute(
                delete(tbl_classification).where(tbl_classification.c.period == period)
            )

    def remove_metrics_from_period(self, period: str, metric_ids: list[int]) -> int:
        """Remove metric values from a period (scoped to period only). Returns count deleted."""
        if not metric_ids:
            return 0
        tbl_fund = self._get_table("fundamental_values")
        with self._engine.begin() as conn:
            result = conn.execute(
                delete(tbl_fund).where(
                    and_(
                        tbl_fund.c.period == period,
                        tbl_fund.c.metric_id.in_(metric_ids),
                    )
                )
            )
        return result.rowcount or 0

    def remove_securities_from_period(self, period: str, security_ids: list[int]) -> int:
        """Remove security values and classifications from a period. Returns count deleted."""
        if not security_ids:
            return 0
        tbl_fund = self._get_table("fundamental_values")
        tbl_classification = self._get_table("security_classification")
        with self._engine.begin() as conn:
            result = conn.execute(
                delete(tbl_fund).where(
                    and_(
                        tbl_fund.c.period == period,
                        tbl_fund.c.security_id.in_(security_ids),
                    )
                )
            )
            conn.execute(
                delete(tbl_classification).where(
                    and_(
                        tbl_classification.c.period == period,
                        tbl_classification.c.security_id.in_(security_ids),
                    )
                )
            )
        return result.rowcount or 0

    def update_period_values(
        self,
        period: str,
        updates: list[dict],
    ) -> int:
        """
        Update fundamental values in a period.
        Each update: {ticker, metric_name, value} or {security_id, metric_id, value}.
        Returns count updated.
        """
        if not updates:
            return 0
        tbl_fund = self._get_table("fundamental_values")
        tbl_sec = self._get_table("securities")
        tbl_metric = self._get_table("metrics")
        count = 0
        with self._engine.begin() as conn:
            for u in updates:
                value = u.get("value")
                if value is None:
                    continue
                sec_id = u.get("security_id")
                metric_id = u.get("metric_id")
                if sec_id is None and u.get("ticker"):
                    row = conn.execute(
                        select(tbl_sec.c.id).where(tbl_sec.c.ticker == u["ticker"])
                    ).first()
                    if not row:
                        continue
                    sec_id = row[0]
                if metric_id is None and u.get("metric_name"):
                    row = conn.execute(
                        select(tbl_metric.c.metric_id).where(
                            tbl_metric.c.metric_name == u["metric_name"]
                        )
                    ).first()
                    if not row:
                        continue
                    metric_id = row[0]
                if sec_id is None or metric_id is None:
                    continue
                exists = conn.execute(
                    select(tbl_fund.c.id).where(
                        and_(
                            tbl_fund.c.period == period,
                            tbl_fund.c.security_id == sec_id,
                            tbl_fund.c.metric_id == metric_id,
                        )
                    )
                ).first()
                if exists:
                    conn.execute(
                        update(tbl_fund)
                        .where(tbl_fund.c.id == exists[0])
                        .values(value=float(value))
                    )
                else:
                    conn.execute(
                        tbl_fund.insert().values(
                            security_id=sec_id,
                            metric_id=metric_id,
                            period=period,
                            value=float(value),
                        )
                    )
                count += 1
        return count

    def list_sectors(self) -> list[str]:
        """Return all sector names in the database."""
        tbl = self._get_table("sectors")
        with self._engine.connect() as conn:
            rows = conn.execute(select(tbl.c.sector_name).order_by(tbl.c.sector_name)).fetchall()
        return [r[0] for r in rows if r[0]]

    def list_industries(self) -> list[str]:
        """Return all industry names in the database."""
        tbl = self._get_table("industries")
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(tbl.c.industry_name).order_by(tbl.c.industry_name)
            ).fetchall()
        return [r[0] for r in rows if r[0]]

    def list_indices(self) -> list[str]:
        """Return all index names in the database."""
        tbl = self._get_table("indices")
        with self._engine.connect() as conn:
            rows = conn.execute(select(tbl.c.name).order_by(tbl.c.name)).fetchall()
        return [r[0] for r in rows if r[0]]

    def _na_treatment_col(self, tbl):
        return tbl.c["n/a treatment"]

    def list_metrics(self) -> list[dict]:
        """
        Return all metrics in the database.
        Each dict has: metric_id, metric_name, higher_is_better, na_handling.
        """
        tbl = self._get_table("metrics")
        na_col = self._na_treatment_col(tbl)
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(
                    tbl.c.metric_id,
                    tbl.c.metric_name,
                    tbl.c.higher_is_better,
                    na_col,
                )
            ).fetchall()

        return [
            {
                "metric_id": r[0],
                "metric_name": r[1],
                "higher_is_better": bool(r[2]) if r[2] is not None else None,
                "na_handling": r[3] if r[3] else None,
            }
            for r in rows
        ]

    def create_metric(
        self,
        metric_name: str,
        higher_is_better: Optional[bool] = None,
        na_handling: Optional[str] = None,
    ) -> dict:
        """Create a new metric. Raises ValueError if name already exists."""
        name = metric_name.strip()
        if not name:
            raise ValueError("metric_name cannot be empty.")
        tbl = self._get_table("metrics")
        na_col = self._na_treatment_col(tbl)
        col_values: dict = {tbl.c.metric_name: name}
        if higher_is_better is not None:
            col_values[tbl.c.higher_is_better] = higher_is_better
        if na_handling is not None:
            col_values[na_col] = na_handling
        with self._engine.begin() as conn:
            existing = conn.execute(
                select(tbl.c.metric_id).where(tbl.c.metric_name == name)
            ).first()
            if existing:
                raise ValueError(f"Metric '{name}' already exists.")
            ins = conn.execute(tbl.insert().values(col_values))
            metric_id = ins.inserted_primary_key[0]
        return {"metric_id": int(metric_id), "metric_name": name}


    def delete_metric(self, metric_id: int) -> None:
        """Delete a metric and its fundamental values. Raises ValueError if not found."""
        tbl_metric = self._get_table("metrics")
        tbl_fund = self._get_table("fundamental_values")
        with self._engine.begin() as conn:
            exists = conn.execute(
                select(tbl_metric.c.metric_id).where(tbl_metric.c.metric_id == metric_id)
            ).first()
            if not exists:
                raise ValueError(f"Metric with id {metric_id} not found.")
            conn.execute(delete(tbl_fund).where(tbl_fund.c.metric_id == metric_id))
            conn.execute(delete(tbl_metric).where(tbl_metric.c.metric_id == metric_id))

    def update_metric(
        self,
        metric_id: int,
        higher_is_better: Optional[bool] = None,
        na_handling: Optional[str] = None,
    ) -> None:
        """
        Update metric fields. At least one of higher_is_better or na_handling must be set.
        Raises ValueError if metric_id does not exist.
        """
        tbl = self._get_table("metrics")
        na_col = self._na_treatment_col(tbl)
        updates: dict = {}
        if higher_is_better is not None:
            updates["higher_is_better"] = higher_is_better
        if na_handling is not None:
            updates[na_col] = na_handling
        if not updates:
            return
        with self._engine.begin() as conn:
            exists = conn.execute(
                select(tbl.c.metric_id).where(tbl.c.metric_id == metric_id)
            ).first()
            if not exists:
                raise ValueError(f"Metric with id {metric_id} not found.")
            conn.execute(
                update(tbl)
                .where(tbl.c.metric_id == metric_id)
                .values(updates)
            )

    def create_derived_metric(
        self,
        *,
        metric_names: list[str],
        operations: list[str],
        new_metric_name: str,
        higher_is_better: Optional[bool] = None,
        na_handling: Optional[str] = None,
    ) -> dict:
        """
        Create/update a derived metric across all available periods/securities.

        Operations are applied left-to-right: (m0 op0 m1) op1 m2 ...
        For two metrics, use metric_names=[A, B], operations=[op].
        """
        if len(metric_names) < 2:
            raise ValueError("metric_names must have at least 2 metrics.")
        if len(operations) != len(metric_names) - 1:
            raise ValueError(
                f"operations must have {len(metric_names) - 1} items for {len(metric_names)} metrics."
            )
        for op in operations:
            if op.strip() not in {"+", "-", "*", "/"}:
                raise ValueError("operation must be one of: +, -, *, /")

        if not new_metric_name.strip():
            raise ValueError("new_metric_name cannot be empty.")

        metric_ids = self.get_metric_ids_by_names(metric_names)
        tbl_fund = self._get_table("fundamental_values")
        tbl_metric = self._get_table("metrics")
        na_col = self._na_treatment_col(tbl_metric)

        with self._engine.connect() as conn:
            dfs = []
            for i, metric_name in enumerate(metric_names):
                mid = metric_ids[metric_name]
                rows = conn.execute(
                    select(
                        tbl_fund.c.security_id,
                        tbl_fund.c.period,
                        tbl_fund.c.value,
                    ).where(tbl_fund.c.metric_id == mid)
                ).fetchall()
                df = pd.DataFrame(rows, columns=["security_id", "period", "value"])
                df = df.rename(columns={"value": f"v{i}"})
                dfs.append(df)

        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on=["security_id", "period"], how="inner")

        if merged.empty:
            raise ValueError("No overlapping rows between selected metrics.")

        result_series = pd.to_numeric(merged["v0"], errors="coerce")
        for i, op in enumerate(operations):
            right = pd.to_numeric(merged[f"v{i + 1}"], errors="coerce")
            op = op.strip()
            if op == "+":
                result_series = result_series + right
            elif op == "-":
                result_series = result_series - right
            elif op == "*":
                result_series = result_series * right
            else:
                result_series = result_series / right.replace(0, pd.NA)

        merged["value"] = result_series

        with self._engine.begin() as conn:
            existing_metric = conn.execute(
                select(tbl_metric.c.metric_id).where(tbl_metric.c.metric_name == new_metric_name)
            ).first()

            if existing_metric:
                new_metric_id = existing_metric[0]
            else:
                ins = conn.execute(tbl_metric.insert().values(metric_name=new_metric_name))
                new_metric_id = ins.inserted_primary_key[0]

            updates = {}
            if higher_is_better is not None:
                updates[tbl_metric.c.higher_is_better] = higher_is_better
            if na_handling is not None:
                updates[na_col] = na_handling
            if updates:
                conn.execute(
                    update(tbl_metric)
                    .where(tbl_metric.c.metric_id == new_metric_id)
                    .values(updates)
                )

            conn.execute(delete(tbl_fund).where(tbl_fund.c.metric_id == new_metric_id))

            to_insert = merged[["security_id", "period", "value"]].copy()
            to_insert["metric_id"] = new_metric_id
            to_insert = to_insert[["security_id", "metric_id", "value", "period"]]
            to_insert.to_sql("fundamental_values", con=conn, if_exists="append", index=False)

        return {
            "metric_id": int(new_metric_id),
            "metric_name": new_metric_name,
            "metric_names": metric_names,
            "operations": operations,
            "records_written": int(len(merged)),
            "periods_covered": int(merged["period"].nunique()),
        }

    def save_fundamentals(
        self,
        df: pd.DataFrame,
        period: str,
        index_code: Optional[str] = None,
        mode: str = "replace",
    ) -> ImportResult:
        """
        Save a DataFrame of fundamental data to the database.
        Normalizes sectors, industries, metrics; upserts securities;
        optionally registers index membership.
        mode: 'replace' (default) overwrites period; 'append' merges new metrics/securities.
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
        tbl_classification = self._get_table("security_classification")

        companies = df[FIXED_COLUMNS].copy().drop_duplicates(subset=["Ticker"])
        ticker_map: dict[str, int] = {}
        classification_map: dict[str, tuple[int | None, int | None]] = {}
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
                classification_map[ticker_val] = (sector_id, industry_id)

                sec_id = self._upsert_security(
                    conn,
                    tbl_sec,
                    ticker_val,
                    long_name_val,
                    sector_id,
                    industry_id,
                )
                ticker_map[ticker_val] = sec_id

            sec_ids = list(ticker_map.values())
            if mode == "replace":
                conn.execute(
                    delete(tbl_classification).where(tbl_classification.c.period == period)
                )
            else:
                conn.execute(
                    delete(tbl_classification).where(
                        and_(
                            tbl_classification.c.security_id.in_(sec_ids),
                            tbl_classification.c.period == period,
                        )
                    )
                )
            for ticker in ticker_map:
                sec_id_val, ind_id_val = classification_map[ticker]
                security_id = ticker_map[ticker]
                conn.execute(
                    tbl_classification.insert().values(
                        security_id=security_id,
                        period=period,
                        sector_id=sec_id_val,
                        industry_id=ind_id_val,
                    )
                )

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
            for m_name in df_long["metric_name"].dropna().astype(str).unique().tolist():
                if m_name not in metric_cache:
                    metric_cache[m_name] = self._resolve_metric(
                        conn, tbl_metric, m_name
                    )

        df_long["metric_id"] = df_long["metric_name"].map(metric_cache)
        # Keep NA fundamental values; only IDs are required for persistence.
        # NaN in `value` is inserted as SQL NULL.
        df_long = df_long.dropna(subset=["security_id", "metric_id"])
        df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

        with self._engine.begin() as conn:
            if mode == "append":
                # Load existing for period, merge (new overwrites overlap)
                existing = pd.read_sql_query(
                    select(
                        tbl_fund.c.security_id,
                        tbl_fund.c.metric_id,
                        tbl_fund.c.value,
                    ).where(tbl_fund.c.period == period),
                    con=conn,
                )
                if not existing.empty:
                    existing["period"] = period
                    merge_keys = ["security_id", "metric_id", "period"]
                    existing = existing[merge_keys + ["value"]]
                    new_data = df_long[["security_id", "metric_id", "value", "period"]]
                    combined = pd.concat([existing, new_data], ignore_index=True)
                    combined = combined.drop_duplicates(
                        subset=["security_id", "metric_id", "period"],
                        keep="last",
                    )
                    conn.execute(delete(tbl_fund).where(tbl_fund.c.period == period))
                    combined.to_sql(
                        "fundamental_values",
                        con=conn,
                        if_exists="append",
                        index=False,
                    )
                else:
                    conn.execute(delete(tbl_fund).where(tbl_fund.c.period == period))
                    data = df_long[["security_id", "metric_id", "value", "period"]]
                    data.to_sql(
                        "fundamental_values",
                        con=conn,
                        if_exists="append",
                        index=False,
                    )
            else:
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
        na_col = self._na_treatment_col(tbl)
        vals = {
            "metric_name": str(name),
            "higher_is_better": True,
            na_col.key: "replace_with_zero",
        }
        ins = conn.execute(tbl.insert().values(**vals))
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
