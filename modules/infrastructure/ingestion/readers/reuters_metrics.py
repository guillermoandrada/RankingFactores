from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from modules.infrastructure.ingestion.readers.base import BaseFileReader

# Minimum file columns: any RIC alias + score. Others are optional (filled with empty/NaN).
_RIC_HEADER_ALIASES = (
    "Identifier (RIC)",
    "Identifier",
)
_SCORE_HEADER = "Earnings Quality Country Rank, Current"

_OPTIONAL_SOURCE_COLUMNS = {
    "company_name": "Company Name",
    "sector": "GICS Sector Name",
    "industry": "GICS Industry Group Name",
    "market_cap_millions": "Company Market Cap (Millions, USD)",
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
                "Required columns (any RIC header): "
                f"{list(_RIC_HEADER_ALIASES)}; and '{_SCORE_HEADER}'. "
                f"Optional: {list(_OPTIONAL_SOURCE_COLUMNS.values())}."
            )

        df = pd.read_excel(filepath, header=header_row)
        df = df.rename(columns=lambda value: str(value).strip())
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
        column_map = self._build_column_map(df.columns)

        if "ric" not in column_map or "score" not in column_map:
            raise ValueError(
                "Missing required Reuters columns: need an Identifier column "
                f"({list(_RIC_HEADER_ALIASES)}) and '{_SCORE_HEADER}'. "
                f"Found: {list(df.columns)}"
            )

        idx = df.index
        if "company_name" in column_map:
            long_name = df[column_map["company_name"]].astype(str)
        else:
            long_name = pd.Series("", index=idx, dtype=object)
        if "sector" in column_map:
            sector = df[column_map["sector"]].astype(str)
        else:
            sector = pd.Series("", index=idx, dtype=object)
        if "industry" in column_map:
            industry = df[column_map["industry"]].astype(str)
        else:
            industry = pd.Series("", index=idx, dtype=object)
        if "market_cap_millions" in column_map:
            market_cap = (
                pd.to_numeric(df[column_map["market_cap_millions"]], errors="coerce")
                * 1_000_000
            )
        else:
            market_cap = pd.Series(np.nan, index=idx, dtype=float)

        normalized = pd.DataFrame(
            {
                "Ticker": df[column_map["ric"]].map(self._extract_ticker),
                "Long Name": long_name,
                "GICS Sector Name": sector,
                "GICS Industry Group Name": industry,
                "Market Cap (USD)": market_cap,
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
        score_norm = _normalize_column_name(_SCORE_HEADER)
        ric_norms = {_normalize_column_name(a) for a in _RIC_HEADER_ALIASES}
        for row_idx in range(preview.shape[0]):
            row_headers = {
                _normalize_column_name(value)
                for value in preview.iloc[row_idx].tolist()
                if str(value).strip()
            }
            if score_norm in row_headers and row_headers.intersection(ric_norms):
                return row_idx
        return None

    def _build_column_map(self, columns: pd.Index) -> dict[str, str]:
        normalized_columns = {
            _normalize_column_name(column): str(column).strip()
            for column in columns
        }
        mapping: dict[str, str] = {}
        for alias in _RIC_HEADER_ALIASES:
            key = _normalize_column_name(alias)
            if key in normalized_columns:
                mapping["ric"] = normalized_columns[key]
                break
        score_norm = _normalize_column_name(_SCORE_HEADER)
        if score_norm in normalized_columns:
            mapping["score"] = normalized_columns[score_norm]
        for key, display_name in _OPTIONAL_SOURCE_COLUMNS.items():
            normalized = _normalize_column_name(display_name)
            if normalized in normalized_columns:
                mapping[key] = normalized_columns[normalized]
        return mapping

    def _extract_ticker(self, value: object) -> str | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        return raw.split(".", 1)[0].strip() or None
