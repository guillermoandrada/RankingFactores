"""Sector/industry lookup shared by the IC analyzer and Metric Diagnostics."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import MetaData, create_engine, literal, select, text

from modules.domain.analytics.security_groups import select_security_groups
from modules.infrastructure.db.schema import create_tables


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    create_tables(engine)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO sectors VALUES (1, 'Tech'), (2, 'Banks')"))
        conn.execute(text("INSERT INTO industries VALUES (1, 'Software'), (2, 'Lenders')"))
        conn.execute(text(
            "INSERT INTO securities (id, ticker, sector_id, industry_id) "
            "VALUES (1, 'AAA', 1, 1), (2, 'BBB', 1, 1), (3, 'CCC', NULL, NULL)"
        ))
        # BBB is reclassified as a bank in this period only.
        conn.execute(text(
            "INSERT INTO security_classification VALUES (2, '2024/03/31', 2, 2)"
        ))
    return engine


def _groups(engine, *, group_by: str) -> dict[str, str | None]:
    metadata = MetaData()
    metadata.reflect(bind=engine)
    securities = metadata.tables["securities"]
    universe = select(
        literal("2024/03/31").label("period"),
        securities.c.id.label("security_id"),
    ).subquery()
    query = select_security_groups(metadata, universe, group_by=group_by)
    with engine.connect() as conn:
        frame = pd.read_sql_query(query, con=conn)
    return {
        row.ticker: None if pd.isna(row.group) else row.group for row in frame.itertuples()
    }


@pytest.mark.parametrize(
    ("group_by", "expected"),
    [
        ("sector", {"AAA": "Tech", "BBB": "Banks", "CCC": None}),
        ("industry", {"AAA": "Software", "BBB": "Lenders", "CCC": None}),
    ],
)
def test_period_classification_wins_over_the_security_default(engine, group_by, expected) -> None:
    assert _groups(engine, group_by=group_by) == expected


def test_unknown_group_by_is_rejected(engine) -> None:
    with pytest.raises(ValueError, match="group_by"):
        _groups(engine, group_by="country")
