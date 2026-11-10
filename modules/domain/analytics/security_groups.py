"""Sector/industry lookup per (period, security), shared by IC and diagnostics."""

from __future__ import annotations

from sqlalchemy import MetaData, and_, func, select
from sqlalchemy.sql import Select
from sqlalchemy.sql.selectable import FromClause

GROUP_BY_OPTIONS = frozenset({"sector", "industry"})


def select_security_groups(
    metadata: MetaData, universe: FromClause, *, group_by: str
) -> Select:
    """
    Select (period, security_id, ticker, group) for every row of ``universe``.

    ``universe`` must expose ``period`` and ``security_id`` columns. The group is
    the period's classification when there is one, otherwise the security's own
    sector/industry; unclassified securities get a NULL group.
    """
    if group_by not in GROUP_BY_OPTIONS:
        raise ValueError("group_by must be 'sector' or 'industry'.")

    tbl_sec = metadata.tables["securities"]
    tbl_class = metadata.tables["security_classification"]
    tbl_label = metadata.tables["sectors" if group_by == "sector" else "industries"]
    id_column = f"{group_by}_id"
    label_column = tbl_label.c[f"{group_by}_name"]

    return (
        select(
            universe.c.period.label("period"),
            universe.c.security_id.label("security_id"),
            tbl_sec.c.ticker.label("ticker"),
            label_column.label("group"),
        )
        .select_from(universe)
        .join(tbl_sec, tbl_sec.c.id == universe.c.security_id)
        .outerjoin(
            tbl_class,
            and_(
                tbl_class.c.security_id == universe.c.security_id,
                tbl_class.c.period == universe.c.period,
            ),
        )
        .outerjoin(
            tbl_label,
            tbl_label.c[id_column]
            == func.coalesce(tbl_class.c[id_column], tbl_sec.c[id_column]),
        )
    )
