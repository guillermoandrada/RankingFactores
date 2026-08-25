"""Ticker normalization shared by ingestion, pricing and portfolio code.

Two levels of normalization live here, and the difference matters:

**Vendor identifier -> ticker.** :func:`ticker_from_bloomberg_id` and
:func:`ticker_from_ric` strip a vendor identifier down to the ticker *as the vendor
spells it*. The ingestion readers use these, because the `securities` table holds
the vendor's spelling: `BFb` and `BRKa` are stored that way, and `GOOG` and `GOOGL`
are two separate rows.

    "AAPL US Equity"    -> "AAPL"     (Bloomberg)
    "AAPL.O"            -> "AAPL"     (Reuters RIC)

**Ticker -> canonical form.** :func:`canonical_ticker` goes one step further:
upper-case, whitespace-collapsed, without Bloomberg market/yellow-key suffixes, with
known duplicate listings aliased to a single symbol. This is the key the price cache,
backtests and portfolio code join on.

    "aapl"              -> "AAPL"
    "AAPL US Equity"    -> "AAPL"
    " brk/b UN Equity"  -> "BRK/B"
    "GOOG"              -> "GOOGL"

Do not reach for :func:`canonical_ticker` when writing fundamentals: aliasing would
fold `GOOG` values onto the `GOOGL` security, and upper-casing would create a second
`BFB` security next to the stored `BFb`.

Provider-specific syntax stays in the provider: Yahoo Finance's
``BRK/B`` -> ``BRK-B`` mapping lives in
:meth:`YFinancePriceProvider.normalize_identifier`, not here. Canonical form keeps
the slash because that is what the database holds.

Dot-separated share classes (``BF.B``) are intentionally left untouched by
:func:`canonical_ticker`: a trailing ``.X`` is ambiguous between a US share class and
a non-US exchange suffix (``VOD.L``). Use :data:`_KNOWN_ALIASES` for the specific
symbols that need it. Reuters RICs are unambiguous on this point, so
:func:`ticker_from_ric` does cut at the dot.
"""

from __future__ import annotations

import re

# Text that pandas produces for an empty cell; never a real identifier.
_MISSING_TEXT = frozenset({"NAN", "NONE", "NAT"})

# Bloomberg yellow keys that may terminate a security description.
_YELLOW_KEYS = frozenset(
    {
        "EQUITY",
        "INDEX",
        "CURNCY",
        "COMDTY",
        "CORP",
        "GOVT",
        "MTGE",
        "PFD",
    }
)

# Distinct listings of the same issuer that the database stores under one symbol.
_KNOWN_ALIASES = {
    "GOOG": "GOOGL",
    "NXP": "NXPI",
}

_EXCHANGE_CODE = re.compile(r"^[A-Z]{2}$")
_WHITESPACE = re.compile(r"\s+")


def _identifier_text(value: object) -> str:
    """
    Return a vendor identifier as trimmed text, or "" when there is no identifier.

    Absorbs the non-breaking spaces of vendor exports and the literal strings
    "nan"/"none"/"NaT" that pandas produces when stringifying missing cells — without
    which a blank cell would be read as a security named "nan".
    """
    text = str(value if value is not None else "").replace("\xa0", " ").strip()
    return "" if text.upper() in _MISSING_TEXT else text


def ticker_from_bloomberg_id(value: object) -> str:
    """
    Return the ticker of a Bloomberg identifier, or "" if there is none.

        "AAPL US Equity"  -> "AAPL"
        "BFb US Equity"   -> "BFb"

    Keeps the vendor's spelling, which is what the `securities` table holds. Use
    :func:`canonical_ticker` instead when the result has to match the price cache.
    """
    tokens = _identifier_text(value).split()
    return tokens[0] if tokens else ""


def ticker_from_ric(value: object) -> str:
    """
    Return the ticker of a Reuters Instrument Code, or "" if there is none.

        "AAPL.O"  -> "AAPL"
        "VOD.L"   -> "VOD"

    Keeps the vendor's spelling, as :func:`ticker_from_bloomberg_id` does.
    """
    return _identifier_text(value).split(".", 1)[0].strip()


def canonical_ticker(value: object) -> str:
    """
    Return the canonical repository form of a ticker, or "" if there is none.

    Handles the non-breaking spaces and trailing market descriptors that appear in
    vendor exports. Returns "" for None, NaN, empty strings and the literal
    strings "nan"/"none" that pandas produces when stringifying missing cells.
    """
    text = _identifier_text(value).upper()
    if not text:
        return ""

    tokens = _WHITESPACE.sub(" ", text).split(" ")
    if len(tokens) > 1 and tokens[-1] in _YELLOW_KEYS:
        tokens.pop()
    if len(tokens) > 1 and _EXCHANGE_CODE.match(tokens[-1]):
        tokens.pop()

    symbol = " ".join(tokens)
    return _KNOWN_ALIASES.get(symbol, symbol)


def canonical_ticker_map(values: list[object]) -> dict[str, str]:
    """
    Map each original identifier to its canonical form, preserving input order and
    dropping values that canonicalize to "".

    Callers that must return results keyed by the identifier the user supplied
    (price providers) use this to look data up canonically without renaming their
    output columns.
    """
    mapping: dict[str, str] = {}
    for value in values:
        original = str(value if value is not None else "").strip()
        canonical = canonical_ticker(value)
        if original and canonical and original not in mapping:
            mapping[original] = canonical
    return mapping
