# Code Quality Review

Assessment of the repository against the principles in [CLAUDE.md](../CLAUDE.md) (identical to
`AGENTS.md`). Reviewed at commit `2d39f9d` on branch `junerefactor`.

Price-layer findings are numbered `F1…F14` and live in [docs/PRICING.md](PRICING.md); this document
covers everything else and references them where relevant.

**Status.** The price-correctness phase of the backlog (items 1–5 in §5) has been implemented: the
suite now stands at **114 passing tests**, up from 56. Resolved here: **D2** (ticker normalisation),
**Y1** (`_empty_price_result`), **Y2** (test symmetry for the price resource), and the `PriceService`
DIP leak. Everything else below is open.

---

## 1. Verdict

The June refactor produced a genuinely well-structured codebase. Layering is clean and consistently
applied, routers are thin, dependency injection is centralised, the domain is framework-free, and
the naming is explicit throughout. Measured against the stated principles this is solidly above
average — the architecture is one an unfamiliar developer can navigate from the folder names alone.

The debt is concentrated in three places:

1. **One class doing too much** — `FinancialDatabase` is a 1,009-line repository spanning six
   unrelated concerns. This is the clearest SRP violation in the repo and the main obstacle to
   further growth.
2. **The `prices` slice never received the symmetry treatment the other resources got** — no schema
   module, verb route, body-on-`DELETE`, no tests, unguarded upload. It reads as the newest code,
   because it is.
3. **Correctness risk in the price layer** — F1 and F2 can produce silently wrong backtest results.

Nothing here is structural rot; all of it is reachable with incremental, well-scoped changes.

---

## 2. Scorecard

| Principle (`CLAUDE.md`) | Rating | Note |
|---|---|---|
| Python / PEP 8 | **Strong** | Consistent naming, imports, spacing. No lint config committed — worth adding `ruff`. |
| SOLID — SRP | **Weak** | `FinancialDatabase` (1,009 lines) mixes fundamentals, metrics, derived metrics, reference lookups, index membership and prices. `PortfolioService` (526) and `ICAnalyzer` (548) are also large but internally cohesive. |
| SOLID — OCP / LSP | **Strong** | `BasePriceProvider` / `BaseFileReader` are extended, not modified. New readers and providers slot in cleanly. |
| SOLID — ISP | **Good** | Provider and reader interfaces are minimal and focused. |
| SOLID — DIP | **Good** | `BacktestService` and `ICService` depend on the `BasePriceProvider` abstraction; concretes are chosen in `dependencies.py`. Two leaks: `PriceService` constructs `BloombergPriceFileReader` internally, and `fetch_latest_adjusted_closes` types its parameter as the concrete Yahoo provider (**F3**). |
| Clean Code | **Good** | Small functions, honest names, docstrings that state contracts. Some over-long UI pages (730 lines). |
| Simplicity | **Good** | Little speculative abstraction. |
| DRY | **Mixed** | Seven copies of the same HTTP error-handling block in `api_client.py`; near-identical `_resolve_sector` / `_resolve_industry`; ticker normalisation reimplemented in three places (**F1**). |
| Explicit names | **Strong** | `fetch_latest_adjusted_closes`, `preserve_existing_classification`, `score_quantile_cutoff` — no cryptic abbreviations. |
| Small focused functions | **Good** | A handful of exceptions: `save_fundamentals` (180 lines, five phases), `_build_trade_payload` (104). |
| Self-explanatory code | **Strong** | Comments explain *why* (e.g. the phase markers in `save_fundamentals`). |
| Layering / modules | **Strong** | Clear `api` / `modules.domain` / `modules.infrastructure` / `streamlit_app` split, one live violation (**F4**). |
| **Symmetry (first-class constraint)** | **Mixed** | Excellent within routers/schemas/services/client for periods, metrics, scorings, portfolios, backtests. The `prices` resource breaks it on every axis (**F10–F12**). |
| API design | **Good** | Thin routers, resource-oriented, consistent envelopes, correct status codes — except `prices`. |
| Type hints | **Strong** | `from __future__ import annotations` everywhere, modern `X | None` syntax. |
| Explicit errors over silent fallbacks | **Mixed** | Good in services; violated by silent `except Exception` in schema migrations, reader metadata extraction, and per-batch provider failures (**F6**). |
| Tests | **Mixed** | 56 tests, correctly isolated (temp SQLite per test, no network). Coverage tracks age: mature areas are covered, the price layer is not (**F12**). |
| Repository conventions (`src/`, `tests/`, `docs/`) | **Partial** | `tests/` yes; `src/` deliberately replaced by `api/` + `modules/` (coherent, keep it); `docs/` did not exist before this review. |

---

## 3. What is done well

- **Layer discipline.** `modules/domain` imports no web framework. Routers contain no business
  logic — [portfolios.py](../api/routers/portfolios.py) is 24 lines and does exactly the four things
  `CLAUDE.md` prescribes. This is the strongest aspect of the codebase.
- **Dependency injection.** [api/dependencies.py](../api/dependencies.py) is a single, readable
  composition root; `@lru_cache(maxsize=1)` gives process-wide singletons without a framework.
  Deferred imports inside the factories cleanly break circular dependencies.
- **Transactional integrity.** [`save_fundamentals`](../modules/infrastructure/db/repository.py#L595)
  runs all five write phases inside one `engine.begin()`, with the guarantee documented in the
  docstring. `delete_period` likewise. Correct and deliberate.
- **Test isolation.** Every DB test builds its own `sqlite:///{tmp_path}` database; provider tests
  monkeypatch `yf.download`/`yf.Ticker` or inject fakes. No test touches `financial_data.db` or the
  network — a discipline many repos lose early.
- **Async hygiene.** Long CPU/IO work (`/ic`, `/backtests/*`) is dispatched with
  `loop.run_in_executor`, and the client pairs it with a separate long-timeout HTTP client. The
  batch scoring endpoint uses a bounded `ThreadPoolExecutor(max_workers=min(8, …))`.
- **Stable response envelopes.** The `portfolio` row shape produced by `POST /portfolios/{period}`
  is consumed unchanged by `POST /backtests/portfolio`. Contracts between endpoints are respected.
- **Domain modelling.** Period-scoped `security_classification` with a `COALESCE` fallback to the
  base classification is a thoughtful solution to GICS reclassification — a real quant problem
  handled properly.

---

## 4. Findings

### 4.1 Structure

**S1 · High · `FinancialDatabase` is a god object.**
[repository.py](../modules/infrastructure/db/repository.py) is 1,009 lines covering fundamentals,
metric CRUD, derived-metric materialisation, reference lookups, index membership and price data.
Every service depends on the whole surface, `_metadata.reflect()` runs on construction, and the
class has no natural place left to grow.

*Fix:* split along the existing seams into `FundamentalsRepository`, `MetricsRepository`,
`ReferenceRepository` and `PriceRepository` (the last is already a clean, self-contained group at
lines 880–973 and can move first with almost no risk). Keep `FinancialDatabase` as a thin facade so
`dependencies.py` and existing call sites are unaffected, then migrate services one at a time.
Doing `PriceRepository` first also unblocks several price-layer fixes.

**S2 · Medium · Two competing implementations of derived metrics.**
[`FinancialDatabase.create_derived_metric`](../modules/infrastructure/db/repository.py#L486) (108
lines) materialises derived values into `fundamental_values`, while
[`DerivedMetricStore`](../modules/config/derived_metrics.py) stores formulas as JSON and
[metric_loader.py](../modules/domain/analytics/metric_loader.py) computes them on the fly. The API
uses only the JSON path — the repository method is **called from nowhere**, and so are
`create_metric` and the module-level `compute_ranking_for_profile`
([ranking_service.py:107](../api/services/ranking_service.py#L107), a pure pass-through alias).

*Fix:* delete the unused code. Two implementations of one concept is the most expensive kind of
duplication, because the wrong one will eventually be extended. If materialisation is wanted later,
reintroduce it deliberately with tests.

**S3 · Medium · No lint/format/type gate.**
No `ruff`/`black`/`mypy` config and no CI. Style is currently maintained by hand, which will not
survive multiple contributors — and an AI tool has nothing to check its output against.

*Fix:* add `ruff` (lint + format) and a `mypy` run over `modules/` and `api/` to `pyproject.toml`,
plus a minimal GitHub Actions job running `ruff check`, `mypy`, `pytest`.

**S4 · Low · `__pycache__` is not git-ignored.**
[.gitignore](../.gitignore) covers `*.db` only; `git status` currently shows untracked `.pyc` files.
Add `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.idea/`, `.history/`.

**S5 · Low · Duplicate Streamlit page prefix.**
`6_Metric_Selection.py` and `6_Portfolio_Construction.py` both start with `6_`, making navigation
order dependent on filesystem sort. Renumber.

### 4.2 API design

**A1 · Medium · The `prices` router deviates on every documented convention.**
Verb route (`POST /prices/upload`), bare-array body on `DELETE`, no `api/schemas/prices.py`, no
`response_model`, no per-item `GET`, no extension/size validation, and an exception mapping that
lets non-`ValueError` parse failures escape as unhandled 500s. Details and fixes in **F10** and
**F11**. The [periods router](../api/routers/periods.py) is the correct template — it validates the
extension, whitelists options, maps `ValueError` → 400, and wraps the unexpected in a logged,
non-leaking 500.

**A2 · Low · `db-metrics` exposes only `POST` and `PUT`.**
[db_metrics.py](../api/routers/db_metrics.py) has two endpoints while its sibling `metrics`
router has full CRUD. `POST /db-metrics` creates a metric from an individual-variable file; reads
are served by `/reference/db/metrics`, deletes by the periods router (`delete_metrics` in the `PUT`
body). Defensible, but the split means "delete a metric" lives in an unrelated resource's update
payload. Worth documenting explicitly or consolidating.

**A3 · Low · Inconsistent success envelopes.**
Some endpoints return `{"success": True, ...}` ([metrics.py:53](../api/routers/metrics.py#L53)),
others return the payload directly, others `204`. Pick one convention: HTTP status carries success,
the body carries data.

### 4.3 Error handling

**E1 · Medium · Silent `except Exception` in schema migration.**
[schema.py:110-126](../modules/infrastructure/db/schema.py#L110-L126) attempts three
`ALTER TABLE`s and swallows every failure into a rollback. A genuine failure (locked DB, permissions,
corrupt file) is indistinguishable from the expected "column already exists". There is no schema
version and no migration tool.

*Fix:* short term, inspect the exception and re-raise anything that is not a duplicate-column error;
medium term, adopt Alembic — the DDL is already SQLAlchemy Core, so the migration path is short.

**E2 · Medium · The `"n/a treatment"` column name.**
[schema.py:34](../modules/infrastructure/db/schema.py#L34) declares a column containing a space and
a slash, forcing a `_na_treatment_col()` accessor at every use site and quoting in raw SQL. Rename
to `na_treatment` under a migration (E1) and drop the accessor.

**E3 · Low · Provider failures are logged below the level anyone reads.**
`_download_candidate_matrix` logs at `WARNING` but then discards the batch;
`_fetch_single_identifier` logs at `DEBUG`. In a default configuration a total data-source outage is
invisible except as empty results. See **F6**.

### 4.4 DRY

**D1 · Medium · Seven copies of the HTTP error block in `api_client.py`.**
`_request` already implements status handling, but `create_period`, `update_period_with_file`,
`run_portfolio_backtest`, `run_strategy_backtest`, `run_ic_analysis`, `upload_price_file` and
`delete_cached_price_tickers` each re-implement the same six lines — because they need either the
long-timeout client or multipart. *Fix:* give `_request` a `long: bool = False` parameter and route
everything through it.

**D2 · Medium · Ticker normalisation exists in three incompatible forms.**
`normalize_ticker` (holdings, with alias map),
`YFinancePriceProvider.normalize_identifier` (Yahoo syntax), and inline `.strip().upper()` in the
portfolio and backtest paths — while the price reader does none of it. This is the root cause of
**F1**. *Fix:* one `modules/shared/tickers.py` with `canonical_ticker()` (repo form) and keep the
provider-specific mapping inside the provider.

**D3 · Low · `_resolve_sector` and `_resolve_industry` are the same function twice.**
[repository.py:776-806](../modules/infrastructure/db/repository.py#L776-L806) differ only in table
and column. Collapse into `_resolve_lookup(conn, table, column, name, cache)`.

### 4.5 Symmetry

**Y1 · Medium · `BacktestService._empty_price_result` is a one-off shadow of `PriceMatrixResult`.**
[backtest_service.py:292-299](../api/services/backtest_service.py#L292-L299) defines a local class
with class-level attributes that duck-types the real dataclass. `CLAUDE.md` forbids exactly this
("Do not introduce one-off structures … without necessity"). Replace with
`PriceMatrixResult(prices=pd.DataFrame())`.

**Y2 · Medium · Test structure does not mirror source structure.**
`test_backtests_api.py` + `test_backtest_service.py` model the right pattern; there is no
`test_prices_api.py`, `test_price_service.py`, `test_hybrid_provider.py` or
`test_bloomberg_prices_reader.py`. See **F12**.

**Y3 · Low · Streamlit pages vary in how they load and cache data.**
Compare [9_Price_Data.py](../streamlit_app/pages/9_Price_Data.py) (manual `session_state` keys with
a Refresh button that pops them) against other pages. A shared
`ui/cache.py::cached_fetch(key, loader)` helper would make the pattern uniform and remove per-page
key bookkeeping.

### 4.6 Configuration and operations

**C1 · Medium · No environment override for configuration.**
[settings.py](../modules/config/settings.py) reads `config.json` at import time with no env-var
fallback, so `DB_URL` cannot be changed per environment without editing a tracked file. *Fix:*
`DB_URL = os.getenv("RANKINGFACTORES_DB_URL", _config["db_url"])`, same for the API base URL used by
the UI (currently a hard-coded default inside [ui/api.py](../streamlit_app/ui/api.py#L20)).

**C2 · Medium · No authentication, CORS policy, or upload limits.**
[api/main.py](../api/main.py) mounts routers and nothing else. Two endpoints accept arbitrary file
uploads that are read fully into memory and parsed by openpyxl. Acceptable for a localhost research
tool, but it must be a documented, deliberate decision — record it in the README and do not bind the
app to `0.0.0.0` without adding at least an API key and a body-size limit.

**C3 · Low · `lru_cache` singletons make state hard to reset.**
Every provider in `dependencies.py` is a process-wide singleton, so tests or scripts that want a
different DB must clear caches manually. Acceptable, but pair it with a
`reset_dependencies()` helper for testability if you start writing API-level tests with a temp DB.

**C4 · Low · `.history/` is tracked in the working tree.**
Two stale editor snapshots of `api/routers/scorings.py` sit in `.history/`. Ignore the directory.

---

## 5. Prioritised backlog

Ordered by (risk of wrong output) × (cost to fix). Items 1–3 are what I would do first.

| # | Item | Refs | Effort | Status |
|---|---|---|---|---|
| 1 | Shared `canonical_ticker()` applied on price write **and** read; echo normalised tickers in the upload response | F1, D2 | S | **done** |
| 2 | Tests for the price layer: hybrid provider, Bloomberg price reader, `price_data` round-trip, `/prices` router | F12, Y2 | M | **done** |
| 3 | Range-aware DB coverage + per-observation merge + `partial_coverage` warning | F2 | M | **done** |
| 4 | `GET /prices/latest`; widen `fetch_latest_adjusted_closes` to `BasePriceProvider`; move the UI onto the API client | F3, F4 | S | **done** |
| 5 | Single monthly index convention across providers | F5 | S | **done** |
| 6 | Harden `POST /prices/upload` to match the periods router; add `api/schemas/prices.py` and noun routes | F10, F11, A1 | S–M | open |
| 7 | Extract `PriceRepository` from `FinancialDatabase`, then the remaining three repositories behind a facade | S1 | M–L | open |
| 8 | Delete dead code: `create_derived_metric`, `create_metric`, `compute_ranking_for_profile` | S2 | S | open |
| 9 | Add `ruff` + `mypy` + a CI job | S3 | S | open |
| 10 | Provider batch-failure budget, parallel fallback, honest log levels | F6, E3 | M | open |
| 11 | Alembic migrations; rename `"n/a treatment"` → `na_treatment` | E1, E2 | M | open |
| 12 | Collapse `api_client` error handling into `_request(long=...)` | D1 | S | open |
| 13 | Env-var config overrides; document the no-auth posture | C1, C2 | S | open |
| 14 | Housekeeping: `.gitignore`, page renumbering, `_resolve_lookup` | S4, S5, D3 | S | open |

Item 6 is the natural next step: the `prices` router is now the only part of the slice that has not
been brought in line, and the rename has been approved in principle (Streamlit is the sole consumer).
