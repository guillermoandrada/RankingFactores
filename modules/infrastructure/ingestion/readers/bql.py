from __future__ import annotations

from typing import Optional
from zipfile import BadZipFile

import pandas as pd

from modules.infrastructure.ingestion.readers.base import BaseFileReader
from modules.infrastructure.ingestion.readers.bloomberg import _format_date_as_period
from modules.shared.tickers import ticker_from_bloomberg_id

_DATA_SHEETS = ("Current", "Past", "Estimated")
_NAME_SHEET = "Name"
_CLASSIFICATION_SHEET = "Classification"
_CONFIG_SHEET = "Config"
_BLOOMBERG_CODE_ALIASES = ("Bloomberg Code", "Bloomberg code", "Ticker", "Identifier")
_NAME_ALIASES = ("Security Name", "Name", "Long Name")
_SECTOR_ALIASES = ("GICS Sector Name", "Sector")
_INDUSTRY_ALIASES = ("GICS Industry Group Name", "Industry")


class BqlFileReader(BaseFileReader):
    """Reader for multi-sheet BQL Excel workbooks."""

    def can_read(self, filepath: str) -> bool:
        if not filepath.lower().endswith((".xlsx", ".xls")):
            return False
        try:
            workbook = pd.ExcelFile(filepath)
        except (BadZipFile, OSError, ValueError):
            return False
        sheet_names = {str(name).strip() for name in workbook.sheet_names}
        required_sheets = set(_DATA_SHEETS) | {
            _NAME_SHEET,
            _CLASSIFICATION_SHEET,
            _CONFIG_SHEET,
        }
        return required_sheets.issubset(sheet_names)

    def read(self, filepath: str) -> pd.DataFrame:
        if not self.can_read(filepath):
            raise ValueError(
                "BQL workbook must contain sheets 'Name', 'Classification', "
                "'Current', 'Past', 'Estimated', and 'Config'."
            )

        sheet_frames: list[pd.DataFrame] = []
        seen_factors: set[str] = set()
        for sheet_name in _DATA_SHEETS:
            frame = self._read_data_sheet(filepath, sheet_name)
            factor_names = {str(column) for column in frame.columns if column != "Ticker"}
            overlapping = sorted(seen_factors.intersection(factor_names))
            if overlapping:
                raise ValueError(
                    "BQL workbook contains duplicated factor names across sheets: "
                    f"{overlapping}"
                )
            seen_factors.update(factor_names)
            sheet_frames.append(frame)

        combined = sheet_frames[0]
        for frame in sheet_frames[1:]:
            combined = combined.merge(frame, on="Ticker", how="outer")

        names = self._read_name_sheet(filepath)
        classifications = self._read_classification_sheet(filepath)
        combined = names.merge(classifications, on="Ticker", how="outer").merge(
            combined,
            on="Ticker",
            how="right",
        )
        # Every sheet reader already returns NA for a missing ticker.
        return combined.dropna(how="all")

    def extract_period(self, filepath: str) -> str:
        try:
            config_df = pd.read_excel(filepath, sheet_name=_CONFIG_SHEET, header=None)
        except (BadZipFile, OSError, ValueError):
            return "UNKNOWN"

        if config_df.shape[0] < 1 or config_df.shape[1] < 2:
            return "UNKNOWN"

        period = _format_date_as_period(config_df.iat[0, 1])
        return period if period else "UNKNOWN"

    def extract_index_code(self, filepath: str) -> Optional[str]:
        try:
            config_df = pd.read_excel(filepath, sheet_name=_CONFIG_SHEET, header=None)
        except (BadZipFile, OSError, ValueError):
            return None

        if config_df.shape[0] < 2 or config_df.shape[1] < 2:
            return None

        raw_value = str(config_df.iat[1, 1] or "").strip()
        return raw_value or None

    def _read_data_sheet(self, filepath: str, sheet_name: str) -> pd.DataFrame:
        raw = pd.read_excel(filepath, sheet_name=sheet_name, header=0)
        raw = raw.rename(columns=lambda value: str(value).strip())
        raw = raw.iloc[1:].reset_index(drop=True)
        raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all")

        if raw.empty:
            raise ValueError(f"BQL sheet '{sheet_name}' is empty.")

        identifier_column = str(raw.columns[0]).strip()
        factor_columns = [column for column in raw.columns[1:] if str(column).strip()]
        if not factor_columns:
            raise ValueError(f"BQL sheet '{sheet_name}' does not contain factor columns.")

        renamed = raw.rename(columns={identifier_column: "Ticker"})
        renamed["Ticker"] = self._read_tickers(renamed["Ticker"])

        numeric_values = renamed[factor_columns].apply(pd.to_numeric, errors="coerce")
        normalized = pd.concat([renamed[["Ticker"]], numeric_values], axis=1)

        duplicate_factors = normalized.columns[normalized.columns.duplicated()].tolist()
        if duplicate_factors:
            raise ValueError(
                f"BQL sheet '{sheet_name}' contains duplicated factor names: {duplicate_factors}"
            )
        duplicate_tickers = (
            normalized["Ticker"].dropna().astype(str).value_counts().loc[lambda values: values > 1]
        )
        if not duplicate_tickers.empty:
            raise ValueError(
                f"BQL sheet '{sheet_name}' contains duplicated tickers: {duplicate_tickers.index.tolist()}"
            )

        return normalized

    def _read_name_sheet(self, filepath: str) -> pd.DataFrame:
        raw = pd.read_excel(filepath, sheet_name=_NAME_SHEET, header=0)
        raw = raw.rename(columns=lambda value: str(value).strip())
        raw = raw.iloc[1:].reset_index(drop=True)
        raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all")

        if raw.empty:
            raise ValueError("BQL sheet 'Name' is empty.")
        ticker_column = self._resolve_column(raw.columns, _BLOOMBERG_CODE_ALIASES, default_index=0)
        name_column = self._resolve_column(raw.columns, _NAME_ALIASES, default_index=1)
        normalized = raw.rename(
            columns={
                ticker_column: "Ticker",
                name_column: "Long Name",
            }
        )[["Ticker", "Long Name"]].copy()
        normalized["Ticker"] = self._read_tickers(normalized["Ticker"])
        duplicate_tickers = (
            normalized["Ticker"].dropna().astype(str).value_counts().loc[lambda values: values > 1]
        )
        if not duplicate_tickers.empty:
            raise ValueError(
                "BQL sheet 'Name' contains duplicated tickers: "
                f"{duplicate_tickers.index.tolist()}"
            )

        return normalized

    def _read_classification_sheet(self, filepath: str) -> pd.DataFrame:
        raw = pd.read_excel(filepath, sheet_name=_CLASSIFICATION_SHEET, header=0)
        raw = raw.rename(columns=lambda value: str(value).strip())
        raw = raw.iloc[1:].reset_index(drop=True)
        raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all")

        if raw.empty:
            raise ValueError("BQL sheet 'Classification' is empty.")
        ticker_column = self._resolve_column(raw.columns, _BLOOMBERG_CODE_ALIASES, default_index=0)
        sector_column = self._resolve_column(raw.columns, _SECTOR_ALIASES, default_index=1)
        industry_column = self._resolve_column(raw.columns, _INDUSTRY_ALIASES, default_index=2)
        normalized = raw.rename(
            columns={
                ticker_column: "Ticker",
                sector_column: "GICS Sector Name",
                industry_column: "GICS Industry Group Name",
            }
        )[
            [
                "Ticker",
                "GICS Sector Name",
                "GICS Industry Group Name",
            ]
        ].copy()
        normalized["Ticker"] = self._read_tickers(normalized["Ticker"])
        duplicate_tickers = (
            normalized["Ticker"].dropna().astype(str).value_counts().loc[lambda values: values > 1]
        )
        if not duplicate_tickers.empty:
            raise ValueError(
                "BQL sheet 'Classification' contains duplicated tickers: "
                f"{duplicate_tickers.index.tolist()}"
            )

        return normalized

    def _read_tickers(self, identifiers: pd.Series) -> pd.Series:
        """Reduce Bloomberg identifiers to tickers, leaving empty cells as NA."""
        return identifiers.map(ticker_from_bloomberg_id).replace("", pd.NA)

    def _resolve_column(
        self,
        columns: pd.Index,
        aliases: tuple[str, ...],
        *,
        default_index: int,
    ) -> str:
        normalized_columns = {str(column).strip().lower(): str(column).strip() for column in columns}
        for alias in aliases:
            match = normalized_columns.get(alias.strip().lower())
            if match:
                return match

        available_columns = [str(column).strip() for column in columns if str(column).strip()]
        if default_index < len(available_columns):
            return available_columns[default_index]

        raise ValueError(
            f"Could not resolve a required BQL column from aliases {list(aliases)}. "
            f"Found columns: {available_columns}"
        )
