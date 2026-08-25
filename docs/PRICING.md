# Securities Pricing Layer — Deep Dive and Audit

Everything that turns a portfolio into a return number passes through this layer. Read this in full
before changing a line of it.

Companion documents: [ARCHITECTURE.md](ARCHITECTURE.md) for the overall system,
[CODE_QUALITY_REVIEW.md](CODE_QUALITY_REVIEW.md) for repo-wide debt.

**Status.** Findings F1–F5, F13 and the price-layer test gap (F12) are **fixed** — see §6.
F6–F11, F14 and F15 remain **open** and are described in §7.

---

## 1. Purpose

Yahoo Finance cannot price everything a research universe contains — delisted names, non-US
listings, share-class oddities, private placements. The price layer therefore has two sources with an
explicit priority rule:

> **Manually uploaded Bloomberg closes take priority over Yahoo Finance**, so an analyst can supply
> correct prices for securities Yahoo does not cover.

Priority applies **per observation**, not per ticker: a security with a partial upload keeps its
cached values on the dates it has, and the remaining dates are filled from Yahoo.

---

## 2. Components

| Component | File | Responsibility |
|---|---|---|
| `canonical_ticker` | [modules/shared/tickers.py](../modules/shared/tickers.py) | The single definition of a cache-joinable ticker. Used by pricing, portfolio and backtesting code. Fundamentals ingestion instead uses `ticker_from_bloomberg_id` / `ticker_from_ric` from the same module, which keep the vendor's spelling — see ARCHITECTURE.md §4 |
| `BasePriceProvider` | [providers/base.py](../modules/infrastructure/market_data/providers/base.py) | ABC: `provider_name`, `fetch_price_matrix(...) -> PriceMatrixResult`; plus `to_period_end` |
| `PriceMatrixResult` | same file | `prices`, `resolved_identifiers`, `missing_identifiers`, `partial_coverage` |
| `YFinancePriceProvider` | [providers/yfinance_provider.py](../modules/infrastructure/market_data/providers/yfinance_provider.py) | Bulk `yf.download` in batches of 100, per-ticker fallback, `auto_adjust=True` |
| `HybridPriceProvider` | [providers/hybrid_provider.py](../modules/infrastructure/market_data/providers/hybrid_provider.py) | DB-first composition with range-aware coverage and per-observation merge |
| `fetch_latest_adjusted_closes` | [latest_closes.py](../modules/infrastructure/market_data/latest_closes.py) | Last non-null close per ticker over a 45-day lookback, via any `BasePriceProvider` |
| `BloombergPriceFileReader` | [readers/bloomberg_prices.py](../modules/infrastructure/ingestion/readers/bloomberg_prices.py) | Wide Excel → `PriceFileParseResult` (long frame + what was skipped) |
| `PriceService` | [api/services/price_service.py](../api/services/price_service.py) | Upload, list, delete, latest closes |
| `price_data` persistence | [repository.py:880-973](../modules/infrastructure/db/repository.py#L880-L973) | `upsert_price_data`, `query_price_matrix`, `list_cached_tickers`, `delete_price_data_for_tickers` |
| Router | [api/routers/prices.py](../api/routers/prices.py) | `POST /prices/upload`, `GET /prices/latest`, `GET /prices/tickers`, `DELETE /prices/tickers` |
| UI | [pages/9_Price_Data.py](../streamlit_app/pages/9_Price_Data.py) | Upload tab + Manage tab |

### Canonical ticker form

`canonical_ticker` is the contract that makes the cache readable. It upper-cases, collapses
whitespace, strips Bloomberg market and yellow-key suffixes, and applies known duplicate-listing
aliases:

```
"aapl"             -> "AAPL"       "AAPL US Equity"   -> "AAPL"
"brk/b UN Equity"  -> "BRK/B"      "GOOG"             -> "GOOGL"
```

It deliberately keeps `/` — the Yahoo `BRK/B` → `BRK-B` mapping is provider syntax and stays in
`YFinancePriceProvider.normalize_identifier`. Dot-separated share classes (`BF.B`) are left alone
because a trailing `.X` is ambiguous with a non-US exchange suffix (`VOD.L`); use the alias map for
specific symbols that need it.

Prices are stored under the canonical form; providers look up canonically but **return columns keyed
by the identifier the caller supplied**.

### Storage contract

```
price_data(id, ticker, price_date, close_price, source)
UNIQUE(ticker, price_date)   -- name: uq_price_ticker_date
```

- `price_date` is a **string** in `YYYY-MM-DD`. Range filters are string comparisons
  ([repository.py:929-932](../modules/infrastructure/db/repository.py#L929-L932)), correct only
  because zero-padded ISO dates sort lexicographically. Any writer that deviates breaks range
  queries silently.
- `close_price` is assumed **already adjusted**. Yahoo data is adjusted (`auto_adjust=True`);
  whether an uploaded Bloomberg column is `PX_LAST` or a total-return index is not validated or
  recorded. Mixing raw and adjusted closes across sources will bias returns.
- `source` is written but never read (open finding **F8**).

---

## 3. End-to-end flows

### 3.1 Upload

```
UI file_uploader (.xlsx/.xls)
  → api_client.upload_price_file             (multipart POST /prices/upload)
  → PriceService.ingest_from_file
      → BloombergPriceFileReader.read        row 1 = headers → canonical_ticker
                                             row 2+ = date | closes
                                             returns frame + skipped_tickers + skipped_rows
      → empty? raise ValueError              → 422 with the expected-layout message
      → FinancialDatabase.upsert_price_data
  → 201 {tickers_imported, tickers_skipped, rows_written, rows_skipped, date_range}
```

### 3.2 Backtest read path

```
BacktestService._run_single_backtest
  → HybridPriceProvider.fetch_price_matrix(tickers, start, end, frequency)
      → canonical_ticker_map(identifiers)                original -> canonical
      → query_price_matrix(canonical tickers)
      → for each identifier: does the cached series span the window (± tolerance)?
      → fallback = yfinance for everything that does not
      → merge: cached.combine_first(fetched)             cached wins per observation
      → classify: populated column | partial_coverage | missing_identifiers
  → warn on missing_identifiers AND partial_coverage
  → drop positions with no price on the first available date
  → reindex to union index, ffill
  → build_portfolio_time_series → compute_summary_metrics
```

### 3.3 IC read path

[ICAnalyzer._compute_forward_returns](../modules/domain/analytics/ic_analyzer.py#L438) requests a
daily matrix over `[period_end + 45d, +forward_months]` and computes
`series.iloc[-1] / series.iloc[0] - 1` per ticker, requiring at least two observations. The injected
provider is the hybrid one ([dependencies.py:94](../api/dependencies.py#L94)); the constructor
default is Yahoo-only, which is what you get if you instantiate `ICAnalyzer` yourself.

### 3.4 Latest closes (portfolio rebalancing)

`GET /prices/latest?tickers=A,B,C` → `PriceService.get_latest_closes` →
`fetch_latest_adjusted_closes` with the **hybrid** provider, over a 45-calendar-day window ending
today. The Streamlit page calls it through `RankingApiClient.get_latest_prices`.

---

## 4. Invariants to preserve

1. `PriceMatrixResult.prices` columns are the **caller's original identifiers**, never the
   canonical or provider-normalised ones. `resolved_identifiers` records the mapping.
2. Every requested identifier ends up in exactly one of: a populated column, `partial_coverage`, or
   `missing_identifiers`. Silence is a bug — downstream warnings are generated from those lists.
3. The index is timezone-naive, midnight-normalised, ascending, with no duplicate columns. Monthly
   data is month-end labelled on **every** path (`to_period_end`).
4. Cached observations beat fetched observations, date by date.
5. `CASH_USD` is never sent to a provider.
6. Providers are read-only. Only `PriceService` / `FinancialDatabase` write `price_data`.
7. No provider call happens from `streamlit_app`.

---

## 5. Test coverage

| File | Covers |
|---|---|
| [test_ticker_canonicalization.py](../tests/test_ticker_canonicalization.py) | vendor header forms, aliases, slash preservation, ordering |
| [test_bloomberg_prices_reader.py](../tests/test_bloomberg_prices_reader.py) | wide→long parsing, canonical headers, skip counting, serial dates, duplicate headers |
| [test_price_repository.py](../tests/test_price_repository.py) | upsert round-trip, conflict replace, inclusive range boundaries, list/delete |
| [test_hybrid_provider.py](../tests/test_hybrid_provider.py) | priority rule, fallback, canonical cache hits, column keying, partial merge, tolerance, monthly alignment |
| [test_price_service.py](../tests/test_price_service.py) | import summary, empty-file rejection, canonical delete, latest closes |
| [test_prices_api.py](../tests/test_prices_api.py) | status codes, payloads, route ordering of `/latest` vs `/tickers` |
| [test_backtest_service.py](../tests/test_backtest_service.py) | partial-coverage warning surfaces to the caller |

All offline. Full suite: 114 tests.

---

## 6. Resolved findings

**F1 · Uploaded tickers were never normalised, so the cache silently missed.** The reader stored the
raw header (`BRK/B US Equity`) while every consumer asked for `BRK/B`, and the SQL lookup is exact
equality — so an upload could show up in the Manage tab with a healthy date range while being
completely unreadable. Fixed by `modules/shared/tickers.py`, applied on the write side in the reader
and on the read side in `HybridPriceProvider` via `canonical_ticker_map`. `normalize_ticker` and the
inline `.strip().upper()` calls in the portfolio and backtest paths now delegate to it, so there is
one definition instead of three. The upload response echoes the canonical names.
*Verified end-to-end:* a workbook with `BRK/B US Equity` / `AAPL US Equity` headers imports 130 rows
and a backtest for `BRK/B` + `AAPL` resolves entirely from cache with zero Yahoo calls.

**F2 · Any single cached day suppressed the Yahoo fallback for the whole window.** Coverage was
`notna().any()`, so one uploaded day inside a five-year backtest meant Yahoo was never consulted and
the position was forward-filled flat for the remainder — with no warning, because the ticker was
neither missing nor droppable. Fixed with a range-aware coverage test plus per-observation merge:
cached values win on their own dates, Yahoo fills the rest, and anything still short of the window is
reported in `partial_coverage` and surfaced as a backtest warning.
*Note on the tolerance:* it is clamped to half the requested window
([`_effective_tolerance`](../modules/infrastructure/market_data/providers/hybrid_provider.py)). A
flat 7-day tolerance would let one observation satisfy any window shorter than a week — exactly the
case where falling back matters most. This was caught by the tests, not by inspection.

**F3 · `fetch_latest_adjusted_closes` bypassed the cache.** It was typed to the concrete Yahoo
provider, so the rebalance "Fetch latest prices" button could not price the delisted names the upload
feature exists for — and since `total_capital` is derived from holding values, an unpriced holding
distorted every target weight. Now accepts any `BasePriceProvider`; `PriceService.get_latest_closes`
injects the hybrid one.

**F4 · The UI imported infrastructure directly.** [pages/6](../streamlit_app/pages/6_Portfolio_Construction.py)
now calls `client.get_latest_prices()` against the new `GET /prices/latest` endpoint. The Streamlit
process no longer needs DB or network access, and `ApiError` replaces a bare `except Exception`.

**F5 · Monthly frequency mixed two index conventions.** DB series resampled to month end while Yahoo
labels `1mo` bars at month start, so a mixed matrix interleaved the two and time-shifted half the
book by up to a month. A shared `to_period_end` helper in `providers/base.py` is now applied by both
providers *and* defensively at the composition point, so one index is guaranteed regardless of what a
provider returns. The Yahoo month-start convention is asserted by
`test_monthly_frequency_puts_both_sources_on_one_month_end_index` against a fake; **confirm it
against a live Yahoo response when network access is available** — this machine has none
(`yf.download` fails with `CertificateVerifyError`).

**F13 · Silent row loss in the reader.** `PriceFileParseResult` now carries `skipped_tickers` and
`skipped_rows`, returned to the caller as `tickers_skipped` / `rows_skipped`.

**F12 · No tests.** Six new test modules, listed in §5.

**Y1 · `_empty_price_result` shadow class.** Replaced with a real `PriceMatrixResult`.

---

## 7. Open findings

### F6 · Medium · A failed bulk download degrades into a serial retry storm

`_download_candidate_matrix` catches any exception per batch, logs a warning and `continue`s
([yfinance_provider.py](../modules/infrastructure/market_data/providers/yfinance_provider.py#L138-L142)).
Because the batch produced nothing, **every** identifier in it falls through to
`_fetch_single_identifier`, which loops candidates sequentially with `timeout=60` each and logs only
at `DEBUG`. With a 739-security universe and a network fault, one request can attempt ~1,400
sequential 60-second calls; the 600 s client timeout fires while the server keeps working. Not
hypothetical here — the certificate failure noted above means every Yahoo call currently fails.

*Fix:* distinguish "batch returned empty" (retry individually) from "batch raised" (do not retry 100
names one at a time); add a failure budget that marks the remainder missing after K consecutive
failures; parallelise the fallback; raise the log level and include the exception type.

### F7 · Low · `auto_cache_yfinance` — removed, but write-through caching is still worth having

The dead flag is gone. The underlying opportunity stands: caching fetched Yahoo series into
`price_data` is the natural counterpart of `upsert_price_data` and the real fix for F6's cost
profile. It needs `source`-aware priority (F8) so cached-Yahoo never outranks uploaded Bloomberg.

### F8 · Low · `source` is written but never used

Hard-coded to `"bloomberg"`, never filtered or ordered by, and not returned by
`list_cached_tickers`, so the Manage tab cannot show provenance. Becomes load-bearing the moment F7
lands.

### F9 · Low-Medium · `upsert_price_data` is an N-statement delete with an unbounded `IN` list

One `DELETE` per ticker with every date bound individually — ≈5,200 parameters per ticker for twenty
years of dailies. Safe under SQLite's modern 32,766 limit, over the historical 999 default, and the
statement count scales with ticker count. A `BETWEEN` range delete or a real
`INSERT ... ON CONFLICT DO UPDATE` is simpler and bounded. Note also the undocumented consequence: an
upload overwrites any other source for the dates it contains.

### F10 · Medium · The upload endpoint has no extension or size guard

[prices.py](../api/routers/prices.py) checks only that a filename exists, then reads the whole body
into memory and maps `ValueError` → 422. A `.csv`, PDF or truncated `.xlsx` raises
`zipfile.BadZipFile` / an openpyxl error — neither is a `ValueError`, so both escape as an unhandled
500 with a traceback. Mirror [periods.py:50-89](../api/routers/periods.py#L50-L89): extension
whitelist → 400, size cap → 413, `ValueError` → 422, `except Exception: logger.exception(...)` → 500.

### F11 · Medium · No `api/schemas/prices.py`, and the route shapes are one-offs

No declared response models; `POST /prices/upload` is a verb route, which `CLAUDE.md` forbids;
`DELETE /prices/tickers` carries a bare JSON array body; there is no `GET /prices/{ticker}`.
Target shape (rename pre-approved, deferred):

```
POST   /prices              201   upload
GET    /prices                    list cached tickers
GET    /prices/latest             latest closes      ← must stay registered BEFORE /{ticker}
GET    /prices/{ticker}           one series
DELETE /prices/{ticker}     204   delete one         (+ ?tickers=A,B for bulk)
```

### F15 · Medium · The documented upload layout is off by one, and the misparse is silent

The reader's docstring uses zero-based row numbers (`Row 0: metadata, Row 1: tickers, Row 2+: data`),
and both the Streamlit caption
([9_Price_Data.py:24-27](../streamlit_app/pages/9_Price_Data.py#L24-L27)) and the `ValueError` message in
[price_service.py](../api/services/price_service.py) restate them verbatim as *"Row 1 = ticker headers
… row 2+ = date"*. Users read those as Excel rows, which is off by one from what the parser does.

Following the caption literally does not fail — it succeeds with garbage. A file with tickers on Excel
row 1 and data from row 2 has its first data row read as the header, so `canonical_ticker("100.0")`
returns `"100"` and the import creates securities named after prices:

```
UI-caption layout -> tickers parsed as: ['100', '200']   rows: 2   skipped: 0
correct layout     -> tickers parsed as: ['AAPL', 'MSFT'] rows: 4
```

*Fix:* restate the layout in 1-based Excel terms in the caption, the docstring and the error message, and
reject headers that parse as numbers (a numeric ticker is never valid) so the misparse fails loudly.
Documented for users in the meantime at [USER_GUIDE.md §13](USER_GUIDE.md#13-supply-your-own-prices--price-data).

### F14 · Low · The IC cache is not invalidated by price changes

`ICService` memoises results in a module-level `lru_cache`; uploading or deleting prices changes
their inputs, but `invalidate_cache()` is never called anywhere. `PriceService.ingest_from_file` and
`delete_tickers` should invalidate it, as should period import.

---

## 8. Verification checklist

1. **Round-trip an upload.** Upload a two-ticker × ten-day file; `rows_written` + `rows_skipped`
   must account for every data cell, and `GET /prices/tickers` must show both tickers.
2. **Confirm the ticker strings match** what the rest of the app asks for:
   ```bash
   python -c "
   import sqlite3; c=sqlite3.connect('financial_data.db')
   p={r[0] for r in c.execute('select distinct ticker from price_data')}
   s={r[0] for r in c.execute('select ticker from securities')}
   print('cached but unmatched:', sorted(p-s)); print('matched:', len(p&s))"
   ```
   With canonicalisation in place this should be empty for any US-listed upload; anything listed is
   a symbol the alias map does not yet know about.
3. **Prove the priority rule.** Upload a deliberately wrong price (e.g. `999.0`) for a liquid ticker
   over a known window, backtest it, and check `components[].start_price` is `999.0`.
4. **Prove partial coverage is handled.** Upload one week for a ticker, run a one-year backtest, and
   check the response `warnings` contains "does not span the full window".
5. **Prove the fallback still fires.** Backtest a ticker with no cached rows and confirm it is priced
   by Yahoo or listed in `warnings` as missing.
6. **Check monthly alignment.** Backtest one cached ticker plus one Yahoo ticker at
   `frequency="monthly"` and confirm `series[].date` holds month-end dates only.
7. **Offline behaviour.** With egress blocked, a backtest of uncached tickers should fail fast with a
   clear warning rather than hang — still open (F6).

---

## 9. Remaining work order

| # | Change | Fixes | Risk |
|---|---|---|---|
| 1 | Harden the upload router (extension, size, exception mapping) to match `periods` | F10 | Low |
| 2 | `api/schemas/prices.py` + noun routes + `response_model` | F11 | Low; client-visible |
| 3 | Batch-failure handling, failure budget, parallel fallback | F6 | Low |
| 4 | Write-through caching with `source`-aware priority | F7, F8 | Medium |
| 5 | `ON CONFLICT` upsert | F9 | Low |
| 6 | Invalidate the IC cache on price writes | F14 | Low |
