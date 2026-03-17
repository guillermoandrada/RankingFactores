from __future__ import annotations

from typing import Optional

import pandas as pd

from modules.ingestion.readers.base import BaseFileReader


_SOURCE_COLUMNS = {
    "ric": "Identifier (RIC)",
    "company_name": "Company Name",
    "sector": "GICS Sector Name",
    "industry": "GICS Industry Group Name",
    "market_cap_millions": "Company Market Cap (Millions, USD)",
    "score": "Earnings Quality Country Rank, Current",
}


def _normalize_column_name(value: object) -> str:
    text = str(value or "").strip()
    return " ".join(text.split()).lower()


class ReutersMetricsFileReader(BaseFileReader):
    """Reader for Reuters metrics Excel files."""

    def can_read(self, filepath: str) -> bool:
        if not filepath.lower().endswith((".xlsx", ".xls")):
            return False
        return self._detect_header_row(filepath) is not None

    def read(self, filepath: str) -> pd.DataFrame:
        header_row = self._detect_header_row(filepath)
        if header_row is None:
            raise ValueError(
                "Could not detect Reuters metrics header row. "
                f"Expected columns: {list(_SOURCE_COLUMNS.values())}"
            )

        df = pd.read_excel(filepath, header=header_row)
        df = df.rename(columns=lambda value: str(value).strip())
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
        column_map = self._build_column_map(df.columns)

        missing = [
            display_name
            for key, display_name in _SOURCE_COLUMNS.items()
            if key not in column_map
        ]
        if missing:
            raise ValueError(
                "Missing required Reuters columns: "
                f"{missing}. Found: {list(df.columns)}"
            )

        normalized = pd.DataFrame(
            {
                "Ticker": df[column_map["ric"]].map(self._extract_ticker),
                "Long Name": df[column_map["company_name"]],
                "GICS Sector Name": df[column_map["sector"]],
                "GICS Industry Group Name": df[column_map["industry"]],
                "Market Cap (USD)": pd.to_numeric(
                    df[column_map["market_cap_millions"]],
                    errors="coerce",
                )
                * 1_000_000,
                "Reuters Score": pd.to_numeric(
                    df[column_map["score"]],
                    errors="coerce",
                ),
            }
        )
        normalized = normalized.dropna(how="all")
        normalized["Ticker"] = normalized["Ticker"].replace("", pd.NA)
        return normalized

    def extract_period(self, filepath: str) -> str:
        _ = filepath
        return "UNKNOWN"

    def extract_index_code(self, filepath: str) -> Optional[str]:
        _ = filepath
        return None

    def _detect_header_row(self, filepath: str) -> int | None:
        preview = pd.read_excel(filepath, header=None, nrows=10)
        required_headers = {
            _normalize_column_name(name) for name in _SOURCE_COLUMNS.values()
        }
        for row_idx in range(preview.shape[0]):
            row_headers = {
                _normalize_column_name(value)
                for value in preview.iloc[row_idx].tolist()
                if str(value).strip()
            }
            if required_headers.issubset(row_headers):
                return row_idx
        return None

    def _build_column_map(self, columns: pd.Index) -> dict[str, str]:
        normalized_columns = {
            _normalize_column_name(column): str(column).strip()
            for column in columns
        }
        mapping: dict[str, str] = {}
        for key, display_name in _SOURCE_COLUMNS.items():
            normalized = _normalize_column_name(display_name)
            if normalized in normalized_columns:
                mapping[key] = normalized_columns[normalized]
        return mapping

    def _extract_ticker(self, value: object) -> str | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        return raw.split(".", 1)[0].strip() or None
