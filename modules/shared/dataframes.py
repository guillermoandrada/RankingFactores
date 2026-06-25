"""Build JSON-serializable record lists from DataFrames (avoid to_json/json.loads)."""

from __future__ import annotations

from typing import Any

import pandas as pd


def dataframe_to_jsonable_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Row dicts suitable for FastAPI/JSON: NaN/NA become None."""
    records = df.to_dict(orient="records")
    for row in records:
        for key, val in list(row.items()):
            if pd.isna(val):
                row[key] = None
    return records
