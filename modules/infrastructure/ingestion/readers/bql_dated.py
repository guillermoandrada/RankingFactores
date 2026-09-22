from __future__ import annotations

from zipfile import BadZipFile

import pandas as pd

from modules.infrastructure.ingestion.readers.bql import BqlFileReader

_DATED_SHEET = "Estimated"
_DATES_HEADER = "DATES"
_LABEL_ROW = 0
_IDENTIFIER_ROW = 1
_FORMULA_ROW = 2
_FIRST_VALUE_ROW = 3


class BqlDatedFileReader(BqlFileReader):
    """BQL workbook whose 'Estimated' sheet is a dated matrix instead of a plain table.

    Bloomberg answers a ``fill=prev`` request with the last value reported before the
    requested date and lays the result out as one column block per ticker sharing a
    single DATES column, so every value sits on the row of the date it was reported.
    This reader keeps only the values reported on the workbook's own as-of date and
    reads everything carried over from an earlier date as NA. Sheets other than
    'Estimated' follow the plain BQL layout.
    """

    def can_read(self, filepath: str) -> bool:
        return super().can_read(filepath) and self._has_dated_sheet(filepath)

    def _unreadable_message(self) -> str:
        return (
            f"{super()._unreadable_message()} Its '{_DATED_SHEET}' sheet must be in dated "
            f"layout, with '{_DATES_HEADER}' in cell A{_FORMULA_ROW + 1}. "
            "Use the 'bql' reader for a plain workbook."
        )

    def _read_data_sheet(self, filepath: str, sheet_name: str) -> pd.DataFrame:
        if sheet_name != _DATED_SHEET:
            return super()._read_data_sheet(filepath, sheet_name)
        return self._read_dated_sheet(filepath, sheet_name)

    def _read_dated_sheet(self, filepath: str, sheet_name: str) -> pd.DataFrame:
        raw = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
        if not self._is_dated_layout(raw):
            raise ValueError(
                f"BQL sheet '{sheet_name}' is not in dated layout "
                f"(cell A{_FORMULA_ROW + 1} is not '{_DATES_HEADER}'). Use the 'bql' reader."
            )

        factor_labels = self._factor_labels_by_field(raw, sheet_name)
        reported_values = self._values_reported_on(raw, self._read_as_of_date(filepath), sheet_name)
        identifiers = raw.iloc[_IDENTIFIER_ROW]
        formulas = raw.iloc[_FORMULA_ROW]

        values_by_identifier: dict[str, dict[str, object]] = {}
        unknown_fields: set[str] = set()
        for column in raw.columns[1:]:
            identifier = self._cell_text(identifiers[column])
            if not identifier:
                continue
            field = self._bql_field(formulas[column])
            label = factor_labels.get(field)
            if label is None:
                unknown_fields.add(field)
                continue
            values_by_identifier.setdefault(identifier, {})[label] = reported_values[column]

        if unknown_fields:
            raise ValueError(
                f"BQL sheet '{sheet_name}' contains columns whose BQL field has no factor "
                f"name in row {_LABEL_ROW + 1}: {sorted(unknown_fields)}"
            )
        if not values_by_identifier:
            raise ValueError(f"BQL sheet '{sheet_name}' does not contain ticker columns.")

        return self._to_factor_frame(values_by_identifier, list(factor_labels.values()), sheet_name)

    def _to_factor_frame(
        self,
        values_by_identifier: dict[str, dict[str, object]],
        factor_columns: list[str],
        sheet_name: str,
    ) -> pd.DataFrame:
        frame = pd.DataFrame.from_dict(values_by_identifier, orient="index")
        frame = frame.reindex(columns=factor_columns)
        frame.index.name = "Ticker"
        frame = frame.reset_index()
        frame["Ticker"] = self._read_tickers(frame["Ticker"])

        numeric_values = frame[factor_columns].apply(pd.to_numeric, errors="coerce")
        normalized = pd.concat([frame[["Ticker"]], numeric_values], axis=1)
        self._ensure_unique_tickers(normalized["Ticker"], sheet_name)
        return normalized

    def _factor_labels_by_field(self, raw: pd.DataFrame, sheet_name: str) -> dict[str, str]:
        """Map each BQL field to its factor name, taken from the labelled first block."""
        labels = raw.iloc[_LABEL_ROW]
        formulas = raw.iloc[_FORMULA_ROW]

        labels_by_field: dict[str, str] = {}
        for column in raw.columns[1:]:
            label = self._cell_text(labels[column])
            field = self._bql_field(formulas[column])
            if not label or not field:
                continue
            known_label = labels_by_field.get(field)
            if known_label == label:
                continue
            if known_label is not None:
                raise ValueError(
                    f"BQL sheet '{sheet_name}' maps the BQL field '{field}' to more than one "
                    f"factor name: {[known_label, label]}"
                )
            if label in labels_by_field.values():
                raise ValueError(
                    f"BQL sheet '{sheet_name}' contains duplicated factor names: {[label]}"
                )
            labels_by_field[field] = label

        if not labels_by_field:
            raise ValueError(f"BQL sheet '{sheet_name}' does not contain factor columns.")
        return labels_by_field

    def _values_reported_on(
        self,
        raw: pd.DataFrame,
        as_of_date: pd.Timestamp,
        sheet_name: str,
    ) -> pd.Series:
        """Return the single row reported on the as-of date, so earlier ones stay out."""
        body = raw.iloc[_FIRST_VALUE_ROW:]
        reported_dates = pd.to_datetime(body.iloc[:, 0], errors="coerce")
        matching_rows = body.loc[reported_dates == as_of_date]

        formatted_date = as_of_date.date().isoformat()
        if matching_rows.empty:
            raise ValueError(
                f"BQL sheet '{sheet_name}' has no row reported on {formatted_date}, "
                "the date declared in 'Config'."
            )
        if len(matching_rows) > 1:
            raise ValueError(
                f"BQL sheet '{sheet_name}' has more than one row reported on {formatted_date}."
            )
        return matching_rows.iloc[0]

    def _read_as_of_date(self, filepath: str) -> pd.Timestamp:
        as_of_date = pd.to_datetime(self.extract_period(filepath), format="%Y/%m/%d", errors="coerce")
        if pd.isna(as_of_date):
            raise ValueError(
                "BQL sheet 'Config' does not declare a readable date, so the dated "
                f"'{_DATED_SHEET}' sheet cannot be matched to a period."
            )
        return as_of_date

    def _has_dated_sheet(self, filepath: str) -> bool:
        try:
            header = pd.read_excel(
                filepath, sheet_name=_DATED_SHEET, header=None, nrows=_FIRST_VALUE_ROW
            )
        except (BadZipFile, OSError, ValueError, KeyError):
            return False
        return self._is_dated_layout(header)

    @staticmethod
    def _is_dated_layout(raw: pd.DataFrame) -> bool:
        if raw.shape[0] <= _FORMULA_ROW or raw.shape[1] < 2:
            return False
        return str(raw.iat[_FORMULA_ROW, 0] or "").strip().upper() == _DATES_HEADER

    @staticmethod
    def _bql_field(formula: object) -> str:
        """Reduce a BQL formula to its field name: 'ebit_margin(...)' -> 'ebit_margin'."""
        text = "" if formula is None or pd.isna(formula) else str(formula).strip()
        return text.split("(", 1)[0].strip().lower()

    @staticmethod
    def _cell_text(value: object) -> str:
        return "" if value is None or pd.isna(value) else str(value).strip()
