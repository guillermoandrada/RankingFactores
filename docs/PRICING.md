# Securities Pricing Layer — Deep Dive and Audit

Everything that turns a portfolio into a return number passes through this layer. Read this in full
before changing a line of it.

Companion documents: [ARCHITECTURE.md](ARCHITECTURE.md) for the overall system,
[CODE_QUALITY_REVIEW.md](CODE_QUALITY_REVIEW.md) for repo-wide debt.

**Status.** Findings F1–F5, F7–F9, F12–F14 are **fixed** — see §6. F6, F10, F11 and F15 remain
**open** and are described in §7.

---

## 1. Purpose

Yahoo Finance cannot price everything a research universe contains — delisted names, non-US
listings, share-class oddities, private placements. The price layer therefore has two sources with an
explicit priority rule:

> **Manually uploaded Bloomberg closes take priority over Yahoo Finance**, so an analyst can supply
> correct prices for securities Yahoo does not cover.

Priority applies **per observation**, not per ticker: a security with a partial upload keeps its
cached values on the dates it has, and the remaining dates are filled from Yahoo.

`price_data` is also the **primary read path**, not just an override store. Every series the hybrid
provider downloads is written back to it, so a window is fetched from Yahoo once and served from
SQLite afterwards. Three consequences follow, and each is load-bearing:

1. **Everything is cached daily**, including the series behind a `frequency="monthly"` request —
   which is why a monthly request downloads *daily* closes and resamples. Month-end resampling
   stamps a bar on the calendar month end rather than the last trading day, so storing monthly bars
   would put dates in the table that never traded, and would let a sparse twelve-point series pass
   the coverage test for a daily window.
2. **A fresh download outranks the rows it replaces.** Splits and dividends rescale the whole
   adjusted history retroactively, so a refetch replaces the provider's entire window rather than
   only the new dates — otherwise one series carries two adjustment bases and fabricates a jump in
   returns. Only *uploaded* observations still outrank a download.
3. **The cache does go stale** for a ticker that splits and is never refetched (its window is still
   "covered", so nothing triggers a download). `DELETE /prices/tickers?source=yfinance` invalidates
   the downloaded tier without touching uploads; the Manage tab exposes it as
   *Invalidate downloaded prices*.

---

## 2. Components

| Component | File | Responsibility |
|---|---|---|
| `canonical_ticker` | [modules/shared/tickers.py](../modules/shared/tickers.py) | The single definition of a cache-joinable ticker. Used by pricing, portfolio and backtesting code. Fundamentals ingestion instead uses `ticker_from_bloomberg_id` / `ticker_from_ric` from the same module, which keep the vendor's spelling — see ARCHITECTURE.md §4 |
| `BasePriceProvider` | [providers/base.py](../modules/infrastructure/market_data/providers/base.py) | ABC: `provider_name`, `fetch_price_matrix(...) -> PriceMatrixResult`; plus `to_period_end` |
| `PriceMatrixResult` | same file | `prices`, `resolved_identifiers`, `missing_identifiers`, `partial_coverage` |
| `YFinancePriceProvider` | [providers/yfinance_provider.py](../modules/infrastructure/market_data/providers/yfinance_provider.py) | Bulk `yf.download` in batches of 100, per-ticker fallback, `auto_adjust=True` |
| `HybridPriceProvider` | [providers/hybrid_provider.py](../modules/infrastructure/market_data/providers/hybrid_provider.py) | DB-first composition with range-aware coverage and per-observation merge; write-through caching of every download |
| `fetch_latest_adjusted_closes` | [latest_closes.py](../modules/infrastructure/market_data/latest_closes.py) | Last non-null close per ticker over a 45-day lookback, via any `BasePriceProvider` |
| `BloombergPriceFileReader` | [readers/bloomberg_prices.py](../modules/infrastructure/ingestion/readers/bloomberg_prices.py) | Wide Excel → `PriceFileParseResult` (long frame + what was skipped) |
| `PriceService` | [api/services/price_service.py](../api/services/price_service.py) | Upload, list, delete, latest closes |
| `price_data` persistence | [repository.py](../modules/infrastructure/db/repository.py) | `upsert_price_data` (uploads, authoritative), `replace_price_data_for_source` (downloads, window-scoped), `query_price_matrix` (`exclude_source=` isolates the authoritative tier), `list_cached_tickers`, `delete_price_data_for_tickers(source=)` |
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
- Every row is **daily**. There is no frequency column and none is wanted: see §1, consequence 1.
- `source` is load-bearing. `"bloomberg"` marks an upload — authoritative, replaced only by another
  upload. `"yfinance"` marks a download — re-fetchable, and replaced wholesale per window on the
  next fetch. Reads that must respect the priority rule use `query_price_matrix(exclude_source=…)`;
  the label written by the hybrid provider is its `download_source`, taken from the wrapped
  provider's `provider_name` rather than hard-coded.

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
      → query_price_matrix(canonical tickers)            full cache      -> coverage test
      → query_price_matrix(..., exclude_source=yfinance) uploaded tier   -> priority
      → for each identifier: does the cached series span the window (± tolerance,
        bounds snapped to the frequency's index convention)?
      → download = yfinance DAILY for everything that does not
      → replace_price_data_for_source(download, source=yfinance, window)   write-through
      → merge: uploaded.combine_first(download)          uploads win per observation
               else full cache                           when nothing was downloaded
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
provider is the hybrid one; so is the constructor default whenever a `db` is available, so
instantiating `ICAnalyzer` yourself no longer silently bypasses the cache. Passing only an `engine`
leaves no repository to cache through and still falls back to Yahoo directly.

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
   data is month-end labelled on **every** path (`to_period_end`), and window bounds are judged in
   the same terms (`to_period_end_bound`).
4. **Uploaded** observations beat downloaded observations, date by date. A *fresh* download beats the
   stale download it replaced — the cache never outranks a refetch of the same window.
5. `CASH_USD` is never sent to a provider.
6. `price_data` holds daily observations only, and every row carries a truthful `source`.
7. A cache write must never break the read it rode in on: `HybridPriceProvider._persist` logs and
   continues, and returns the resolved prices regardless.
8. No provider call happens from `streamlit_app`.

---

## 5. Test coverage

| File | Covers |
|---|---|
| [test_ticker_canonicalization.py](../tests/test_ticker_canonicalization.py) | vendor header forms, aliases, slash preservation, ordering |
| [test_bloomberg_prices_reader.py](../tests/test_bloomberg_prices_reader.py) | wide→long parsing, canonical headers, skip counting, serial dates, duplicate headers |
| [test_price_repository.py](../tests/test_price_repository.py) | upsert round-trip, conflict replace, inclusive range boundaries, list/delete, per-source listing and delete, duplicate collapse, multi-batch histories, `replace_price_data_for_source` window semantics |
| [test_hybrid_provider.py](../tests/test_hybrid_provider.py) | priority rule, fallback, canonical cache hits, column keying, partial merge, tolerance, monthly alignment, write-through persistence, canonical write keys, upload protection, daily-only cache, monthly served from a daily cache, refetch replacement, opt-out, write-failure tolerance |
| [test_price_service.py](../tests/test_price_service.py) | import summary, empty-file rejection, canonical delete, source-scoped delete, latest closes |
| [test_prices_api.py](../tests/test_prices_api.py) | status codes, payloads, route ordering of `/latest` vs `/tickers`, `source` filter pass-through |
| [test_ic_cache_invalidation.py](../tests/test_ic_cache_invalidation.py) | price writes clear memoised IC results; every mutating router imports a hook |
| [test_backtest_service.py](../tests/test_backtest_service.py) | partial-coverage warning surfaces to the caller |

All offline. Full suite: 326 tests.

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

**F7 · Yahoo downloads were thrown away.** `price_data` was an override store that only the upload
endpoint ever wrote, so it sat empty (`select count(*) from price_data` → `0`) and *every* backtest,
IC run and latest-close lookup went to Yahoo from scratch — the DB-first provider had nothing to be
first about. `HybridPriceProvider` is now a write-through cache: whatever it downloads is persisted
under the canonical ticker before the result is composed, so the second request for a window is
served from SQLite. Two design decisions came out of getting it right, both documented in §1 and
both enforced by tests:

- Monthly requests **download daily and resample**, so the cache has one granularity. Storing
  month-end-stamped bars would have written dates that never traded and let twelve points satisfy a
  daily window.
- A refetch **replaces the provider's whole window**, not just the new dates. Adjusted closes are
  rescaled retroactively by splits, so appending to a stale segment puts two adjustment bases in one
  series. The first version of this got it wrong in the read path too — `cached.combine_first(fetched)`
  let a pre-split cached segment outrank the rescaled download — and
  `test_refetching_replaces_the_whole_downloaded_window` is what caught it.

*Residual risk:* a ticker that splits and is never refetched keeps a stale adjusted history, because
its window still passes the coverage test. `DELETE /prices/tickers?source=yfinance` is the escape
hatch. Making this automatic (store the fetch date, expire downloaded rows after N days) is the
obvious next step and is **not** implemented.

**F8 · `source` was written but never read.** It is now the axis the priority rule turns on:
`query_price_matrix(exclude_source=…)` isolates the authoritative tier,
`replace_price_data_for_source` scopes a rewrite to one provider,
`delete_price_data_for_tickers(source=…)` scopes an invalidation, and `list_cached_tickers` returns
`sources` so the Manage tab shows provenance. The label the hybrid provider writes comes from the
wrapped provider's `provider_name`, so it cannot drift from reality.

**F9 · `upsert_price_data` built an unbounded `IN` list.** The delete is now batched at 500 dates per
statement (`_delete_price_dates`), which matters far more now that a writer routinely hands it
thousands of daily rows per ticker; `test_upsert_writes_histories_longer_than_one_sql_batch` covers
3,000 dates twice over. Frame normalisation moved to a shared `_prepare_price_frame`, which also
collapses duplicate `(ticker, price_date)` pairs — two caller identifiers can canonicalize to one
ticker (`GOOG`/`GOOGL`) and the unique constraint would otherwise reject the whole batch. A real
`ON CONFLICT DO UPDATE` is still the tidier form, but it is no longer a correctness concern.

**F13 · Silent row loss in the reader.** `PriceFileParseResult` now carries `skipped_tickers` and
`skipped_rows`, returned to the caller as `tickers_skipped` / `rows_skipped`.

**F14 · The IC cache ignored price changes.** `invalidate_price_caches()` sits beside
`invalidate_fundamentals_caches()` in `api/dependencies.py` and is called by both mutating price
routes, matching the pattern the periods and db_metrics routers already follow. IC forward returns
are computed from the price matrix, so an upload or an invalidation changes IC inputs exactly as a
fundamentals import does.

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

*Partly mitigated by F7:* write-through caching means a universe is only exposed to this once per
window rather than on every run. The storm itself is unchanged.

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
DELETE /prices/{ticker}     204   delete one         (+ ?tickers=A,B for bulk, ?source= to scope)
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
8. **Prove the cache fills itself.** With an empty `price_data`, run a backtest of two liquid
   tickers, then check the table:
   ```bash
   python -c "
   import sqlite3; c=sqlite3.connect('financial_data.db')
   print(c.execute('select source, count(*), count(distinct ticker), min(price_date), max(price_date) from price_data group by source').fetchall())"
   ```
   A `('yfinance', …)` row must appear. Re-run the same backtest with egress blocked: it must
   succeed with no `missing_identifiers`.
9. **Prove monthly reuses the daily cache.** Run the daily backtest above, then the same window at
   `frequency="monthly"` with egress blocked. It must succeed, and `row_count` must not change —
   monthly bars are never written.
10. **Prove the invalidation is scoped.** Upload one price for a ticker that also has downloaded
    rows, then `DELETE /prices/tickers?source=yfinance`. `GET /prices/tickers` must still list the
    ticker with `sources: ["bloomberg"]`.

---

## 9. Remaining work order

| # | Change | Fixes | Risk |
|---|---|---|---|
| 1 | Restate the upload layout in 1-based Excel rows; reject numeric headers | F15 | Low |
| 2 | Harden the upload router (extension, size, exception mapping) to match `periods` | F10 | Low |
| 3 | `api/schemas/prices.py` + noun routes + `response_model` | F11 | Low; client-visible |
| 4 | Batch-failure handling, failure budget, parallel fallback | F6 | Low |
| 5 | Age-based expiry of downloaded rows, so a split does not need a manual invalidation | F7 residual | Medium |
| 6 | `ON CONFLICT` upsert in place of delete-then-append | tidiness | Low |
