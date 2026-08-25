# RankingFactores — Architecture Guide

Onboarding document for developers and AI coding tools. Read this first, then
[docs/PRICING.md](PRICING.md) if you touch anything that reads or writes prices, and
[docs/CODE_QUALITY_REVIEW.md](CODE_QUALITY_REVIEW.md) for the known-debt backlog.

For how the application is *used* — building factors, ranking, constructing portfolios and
backtesting them through the Streamlit UI — see [docs/USER_GUIDE.md](USER_GUIDE.md). Read it before
changing UI behaviour or the semantics of a scoring/portfolio parameter; it documents the contract
users rely on.

The rules in [CLAUDE.md](../CLAUDE.md) (identical to `AGENTS.md`) override anything here.
This document describes *what exists*; `CLAUDE.md` describes *how to extend it*.

---

## 1. What the system does

RankingFactores is a quantitative equity research tool. It:

1. **Ingests** fundamental data per period from Excel exports (Bloomberg DAPI, Bloomberg BQL, Reuters).
2. **Scores and ranks** securities by combining metrics into factors using configurable
   *scoring profiles* (z-score normalisation + weighted combination).
3. **Constructs portfolios** from a ranked universe (three strategies, with sector/industry
   constraints, ethical exclusions and rebalancing against existing holdings).
4. **Backtests** those portfolios against price history and a benchmark.
5. **Diagnoses factors** with Information Coefficient (Rank IC) and inter-factor correlation.

Everything downstream of step 3 needs **security prices**, which come from a two-source
price layer (uploaded Bloomberg closes + Yahoo Finance). That layer is documented separately in
[docs/PRICING.md](PRICING.md).

---

## 2. Runtime topology

Three independent processes:

| Process | Entry point | Role |
|---|---|---|
| API | [api/main.py](../api/main.py) — FastAPI on `:8000` | All business logic. The only writer to the DB. |
| UI | [streamlit_app/Home.py](../streamlit_app/Home.py) — Streamlit on `:8501` | Presentation only. Talks HTTP to the API. |
| Storage | `financial_data.db` — SQLite | Fundamentals, reference data, cached prices. Git-ignored. |

Scoring profiles and derived-metric formulas live in **JSON files** next to
[modules/config/](../modules/config/), not in SQLite — see
[ranking_profiles.py](../modules/config/ranking_profiles.py) and
[derived_metrics.py](../modules/config/derived_metrics.py).

### Running it

```bash
pip install -r requirements.txt

# terminal 1 — API (docs at http://127.0.0.1:8000/docs)
uvicorn api.main:app --reload

# terminal 2 — UI
streamlit run streamlit_app/Home.py

# tests (no network, no shared DB — every test builds its own temp SQLite file)
python -m pytest tests -q
```

The UI reads its API base URL from a sidebar text input rendered by
[get_api_client](../streamlit_app/ui/api.py), defaulting to `http://127.0.0.1:8000`.
There is no authentication, CORS policy, or rate limiting — this is an internal
single-user research tool. Do not expose the API to an untrusted network as-is.

---

## 3. Layer map and dependency rules

```
streamlit_app/            presentation — no business logic, no SQL, no network besides the API
├── Home.py               entry page + KPI dashboard
├── pages/                one file per screen, numeric prefix drives nav order
├── components/           reusable stateful widgets (profile editor, wizard)
├── client/api_client.py  the ONLY place that knows HTTP + endpoint paths
└── ui/                   shared page furniture (header, CSS, API client factory)

api/                      HTTP boundary + orchestration
├── main.py               router registration only
├── dependencies.py       dependency injection; every provider is an @lru_cache singleton
├── routers/              thin: parse → validate shape → delegate → map errors to HTTP
├── schemas/              Pydantic request models (one module per resource)
└── services/             orchestration; owns transactions across domain + infrastructure

modules/                  reusable core, framework-agnostic (no FastAPI, no Streamlit)
├── domain/               pure business logic
│   ├── analytics/        z-scores, factor combination, ranking, IC
│   ├── portfolio/        strategies, constraints, holdings parsing
│   ├── backtesting/      series construction, performance metrics
│   └── models/           shared entities
├── infrastructure/       I/O
│   ├── db/               SQLAlchemy Core schema + FinancialDatabase repository
│   ├── ingestion/        Excel readers per vendor + DataImporter
│   └── market_data/      price providers (see PRICING.md)
├── config/               settings + JSON-backed stores
└── shared/               cross-cutting helpers (DataFrame → JSON, canonical tickers)
```

`modules/shared/tickers.py` holds the **single** definition of every ticker string rule, at two
levels. Never re-implement `.strip().upper()` or `.split()[0]` on a ticker in new code.

| Function | Rule | Used by |
|---|---|---|
| `ticker_from_bloomberg_id` | `AAPL US Equity` → `AAPL`, vendor spelling kept | Bloomberg/BQL readers |
| `ticker_from_ric` | `AAPL.O` → `AAPL`, vendor spelling kept | Reuters reader |
| `canonical_ticker` | upper-case, suffixes stripped, duplicate listings aliased | pricing, portfolio, backtesting |

The split is load-bearing, not cosmetic. The `securities` table stores the vendor's spelling, so
fundamentals ingestion must use the vendor-level functions: `canonical_ticker` would fold `GOOG`
values onto the separate `GOOGL` security and create a second `BFB` next to the stored `BFb`.
Everything that joins on the price cache uses `canonical_ticker`.

**Allowed dependency direction:** `streamlit_app → (HTTP) → api → modules.domain / modules.infrastructure`.
`modules.domain` must not import from `api` or `streamlit_app`. `modules.infrastructure` must not
import from `modules.domain` except for entity/DTO types.

Streamlit pages import `modules.domain` for pure parsing helpers only (holdings and filter
workbooks). Anything requiring the database or the network goes through `RankingApiClient`.

Note the repo deliberately does **not** use the `src/` layout suggested in `CLAUDE.md`;
`api/` + `modules/` + `streamlit_app/` is the established structure. Extend it, don't relocate it.

---

## 4. Request lifecycle — a worked example

`POST /portfolios/2024%2F12%2F31` (build a portfolio for period `2024/12/31`):

1. **Router** [api/routers/portfolios.py](../api/routers/portfolios.py) validates the period exists
   (404 if not), delegates, maps `ValueError` → 400.
2. **Service** [PortfolioService.construct_portfolio](../api/services/portfolio_service.py#L51)
   calls `compute_ranking(...)` to get the scored universe, converts rows into
   `SecurityCandidate` objects enriched with period-scoped sector/industry from the DB, then
   dispatches to the requested strategy.
3. **Ranking** [api/services/ranking_service.py](../api/services/ranking_service.py) resolves the
   scoring profile, loads the metric matrix, computes z-scores
   ([ZScoreCalculator](../modules/domain/analytics/zscore.py)), combines factors
   ([combiners.py](../modules/domain/analytics/combiners.py)) and ranks
   ([ranking.py](../modules/domain/analytics/ranking.py)).
4. **Strategy** one of [legacy_rebalance.py](../modules/domain/portfolio/legacy_rebalance.py),
   [smart_beta.py](../modules/domain/portfolio/smart_beta.py),
   [long_short.py](../modules/domain/portfolio/long_short.py) returns
   `(list[TargetPosition], PortfolioDiagnostics)`.
5. **Service** assembles the stable response envelope: `portfolio`, `current_portfolio`, `trades`,
   `excluded`, `constraint_diagnostics`, `notes`, `summary`, `source_count`.

The same `portfolio` list feeds `POST /backtests/portfolio` unchanged — that shape is a contract
between the two endpoints. Keep it stable.

---

## 5. Data model

Defined imperatively in [modules/infrastructure/db/schema.py](../modules/infrastructure/db/schema.py)
and created on first `FinancialDatabase()` construction.

| Table | Grain | Notes |
|---|---|---|
| `securities` | one row per ticker | base sector/industry + `market_cap_usd` |
| `sectors`, `industries` | lookup | names normalised (NBSP → space, whitespace collapsed) |
| `security_classification` | (security, period) | period-scoped GICS; lets a stock change sector over time. Queries `COALESCE` this over `securities` |
| `metrics` | one row per metric | `higher_is_better`, and a column literally named `"n/a treatment"` |
| `fundamental_values` | (security, metric, period) | `value` may be SQL NULL; NA policy applied at read time |
| `indices`, `index_membership` | (index, security, period) | universe filtering |
| `price_data` | (ticker, price_date) unique | close prices + `source`. See PRICING.md |

**Periods are strings, not dates.** Bloomberg imports store `YYYY/MM/DD`
([bloomberg.py](../modules/infrastructure/ingestion/readers/bloomberg.py#L11-L28));
quarter labels like `2024 Q4` are also accepted by the IC analyser
([_period_to_quarter_end_date](../modules/domain/analytics/ic_analyzer.py#L495)). Because periods
can contain `/`, every period path parameter uses `{period:path}` and the client URL-encodes with
`quote(period, safe='')`.

Two schema warts to be aware of before you touch DDL:

- The column `"n/a treatment"` contains a space and a slash, so it can only be reached through
  [`_na_treatment_col()`](../modules/infrastructure/db/repository.py#L353). It is exposed to
  the API as `na_handling`.
- Migrations are `try: ALTER TABLE ... except Exception: rollback` blocks at
  [schema.py:110-126](../modules/infrastructure/db/schema.py#L110-L126). There is no migration
  tool and no schema version. Adding a column means adding another such block.

---

## 6. The two data planes

Keep these mentally separate — they use different identifiers and different ingestion paths.

**Fundamentals plane.** Excel → a vendor reader in
[ingestion/readers/](../modules/infrastructure/ingestion/readers/) → a wide DataFrame with the
`FIXED_COLUMNS` from [config.json](../modules/config/config.json) → `DataImporter` →
[`save_fundamentals`](../modules/infrastructure/db/repository.py#L595), which runs all five write
phases inside a single transaction (sectors/industries/securities → classifications → index
membership → metric resolution → melt and insert). `mode="replace"` overwrites the period;
`mode="append"` merges.

The same plane has a second entry point for files holding **one variable across many periods**:
Excel → [BloombergIndividualVariableReader](../modules/infrastructure/ingestion/readers/bloomberg_individual_variable.py)
→ `DbMetricService` → `save_fundamentals(mode="append")`, once per period. The reader emits one
`FIXED_COLUMNS`-shaped frame per period (the metric column is named after the sheet), so the write
goes through the same five phases as any other import — no second persistence path. Two differences
from `DataImporter`: the file spans several periods, so each is committed on its own; and no
`index_code` is passed, because a variable file may cover part of a universe and
`_write_index_membership` replaces a period's membership wholesale.

**The file decides how far the write may go.** Blocks are five columns (ticker, long name, both GICS
levels, value) or two (ticker, value); the reader measures the distance between period blocks to tell
which, and reports it as `creates_securities`. The narrow form cannot populate a security, so
`DbMetricService._restrict_to_existing_securities` filters the frames against
`get_existing_tickers` before writing and passes `preserve_existing_classification=True`. That filter
is the whole guarantee — `save_fundamentals` upserts every ticker it is handed, so "never create a
security" has to be enforced by *what reaches it*, not by a flag inside it. With no new security in
the frame, phase 1 finds every row and updates nothing (it only fills blank fields) and phase 2
skips every classification write, which is why no second write path is needed for the narrow form
either.

**Intentional asymmetry in the UI.** Both entry points sit in the same **Periods → Create** tab, so
the `Reader` dropdown there dispatches to two endpoints: `POST /periods` for the three period readers
and `POST /db-metrics` for `bloomberg_individual_variable`
([1_Periods.py](../streamlit_app/pages/1_Periods.py)). The endpoints stay separate because their
responses are different shapes — one period envelope versus a per-period list — and overloading
`POST /periods` with a multi-period response would break its contract. The dropdown is a UI
affordance, not a claim that one endpoint serves all four readers.

**Price plane.** Excel → [BloombergPriceFileReader](../modules/infrastructure/ingestion/readers/bloomberg_prices.py)
→ `price_data`; reads go through [HybridPriceProvider](../modules/infrastructure/market_data/providers/hybrid_provider.py),
which prefers `price_data` and falls back to Yahoo Finance. **Fully documented in
[docs/PRICING.md](PRICING.md) — read it before changing anything here.**

---

## 7. Core domain concepts

- **Scoring profile** — a JSON tree of *factors*; each factor holds `{metric_name: weight}` and a
  combination `method`. Resolved per (sector, industry) scope by
  [RankingProfileResolver](../modules/config/ranking_profiles.py). Edited in the UI by
  [components/profile_editor/](../streamlit_app/components/profile_editor/) and the wizard.
- **Derived metric** — a formula (`metric_names` + left-to-right `operations`) stored as JSON and
  computed on the fly at load time by [metric_loader.py](../modules/domain/analytics/metric_loader.py).
  (Note: [`create_derived_metric`](../modules/infrastructure/db/repository.py#L486) also *materialises*
  derived values into `fundamental_values`; the JSON store is the path the API actually uses.)
- **NA handling** — per metric, one of `replace_with_zero`, `replace_with_high`, `replace_with_low`,
  `eliminate`. Applied when the metric matrix is loaded, before z-scoring.
- **Strategies** — `legacy_rebalance` (score-quantile driven, mirrors the legacy Excel process,
  specified in [REBALANCING_SPEC.md](../REBALANCING_SPEC.md)), `smart_beta` (top-N capped weights),
  `long_short` (bucketed long/short with gross/net exposure targets).
- **Backtest** — `fixed_weights` (constant target weights, daily rebalance) or `drifting_weights`
  (buy-and-hold from rebased prices). Built in
  [series_builder.py](../modules/domain/backtesting/series_builder.py); summary statistics in
  [metrics.py](../modules/domain/backtesting/metrics.py). `CASH_USD` is a reserved pseudo-ticker
  that is never priced.
- **IC (Information Coefficient)** — cross-sectional Spearman rank correlation between a metric and
  forward return. Forward window starts at period end **+ 45 days publication lag**
  ([ic_analyzer.py:44](../modules/domain/analytics/ic_analyzer.py#L44)) and runs `forward_months`.

---

## 8. API surface

| Router | Endpoints |
|---|---|
| [periods](../api/routers/periods.py) | `GET/POST/PUT/DELETE` — canonical CRUD; copy this router's shape |
| [metrics](../api/routers/metrics.py) | `GET/POST/PUT/DELETE` derived-metric formulas |
| [db_metrics](../api/routers/db_metrics.py) | `POST /db-metrics` (individual-variable file upload), `PUT /db-metrics/{id}` |
| [reference](../api/routers/reference.py) | read-only lookups: stats, periods, sectors, industries, indices, metrics |
| [scoring_profiles](../api/routers/scoring_profiles.py) | `GET/PUT/DELETE` |
| [scorings](../api/routers/scorings.py) | `POST /scorings/{period}` and `/batch` (computation) |
| [portfolios](../api/routers/portfolios.py) | `POST /portfolios/{period}` (computation) |
| [backtests](../api/routers/backtests.py) | `POST /backtests/portfolio`, `POST /backtests/strategy` |
| [ic](../api/routers/ic.py) | `POST /ic` |
| [prices](../api/routers/prices.py) | `POST /prices/upload`, `GET /prices/latest`, `GET /prices/tickers`, `DELETE /prices/tickers` |

Conventions in force:

- Computation endpoints are `POST` with a body even though they are read-only — accepted deviation
  from the CRUD default, because the request payloads are too large for query strings.
- Long-running work (`/ic`, `/backtests/*`) is pushed to a thread via
  `loop.run_in_executor` so the event loop stays responsive; the client uses a separate
  600 s-timeout HTTP client for those paths.
- Error mapping: `404` for unknown period/metric, `400` for malformed input, `422` for
  well-formed-but-invalid payloads, `204` for deletes with no body, `500` only for genuinely
  unexpected failures (and always logged with `logger.exception`, never leaking internals —
  see [periods.py:84-89](../api/routers/periods.py#L84-L89) for the reference implementation).
- The `prices` router is the least conformant one (verb route, body on `DELETE`, no schema module).
  Do not treat it as the pattern; see **F10–F11** in [docs/PRICING.md](PRICING.md).

---

## 9. Adding code — the checklist

`CLAUDE.md` makes symmetry a hard constraint. Before writing anything, find the sibling and copy it.

**New endpoint on an existing resource**
1. Request model in `api/schemas/<resource>.py`.
2. Route in `api/routers/<resource>.py` — parse, validate, delegate, map errors. No logic.
3. Logic in `api/services/<resource>_service.py`.
4. Persistence in `FinancialDatabase` (or a new repository — see the review doc).
5. Client method in [api_client.py](../streamlit_app/client/api_client.py), named after the endpoint.
6. Test in `tests/test_<resource>_api.py`, modelled on
   [test_backtests_api.py](../tests/test_backtests_api.py).

**New vendor file reader**
Subclass [BaseFileReader](../modules/infrastructure/ingestion/readers/base.py), register it in
[readers/__init__.py](../modules/infrastructure/ingestion/readers/__init__.py) and in the reader
whitelist at [periods.py:55](../api/routers/periods.py#L55), add a period-import test mirroring
[test_bql_period_import.py](../tests/test_bql_period_import.py).

A file that is *not* one period of fundamentals does not belong to that interface. Copy
[bloomberg_prices.py](../modules/infrastructure/ingestion/readers/bloomberg_prices.py) or
[bloomberg_individual_variable.py](../modules/infrastructure/ingestion/readers/bloomberg_individual_variable.py)
instead: parse `bytes` into a frozen result carrying the long frame *and* what was skipped, let the
service turn an empty result into a `ValueError`, and test the reader on its own.

**New price source**
Implement [BasePriceProvider](../modules/infrastructure/market_data/providers/base.py) and return a
`PriceMatrixResult` keyed by the **original** identifier. Wire it in
[dependencies.py](../api/dependencies.py). Never call a provider from `streamlit_app`.

**New portfolio strategy**
Add a module under [modules/domain/portfolio/](../modules/domain/portfolio/) returning
`(list[TargetPosition], PortfolioDiagnostics)`, extend the `strategy` literal in
[schemas/portfolios.py](../api/schemas/portfolios.py), and add a branch in
[`PortfolioService._run_strategy`](../api/services/portfolio_service.py#L219).

**Style**
`from __future__ import annotations` at the top; type hints on public functions; keyword-only
arguments for anything with more than two parameters; docstrings that state the contract, not the
implementation; explicit `raise ValueError` over silent fallbacks; no bare `except Exception` unless
you log it and explain why.

---

## 10. Testing

114 tests in [tests/](../tests/), all passing, all offline. Two patterns to imitate:

- **DB tests** construct their own database: `FinancialDatabase(db_url=f"sqlite:///{tmp_path}/x.db")`.
  Never touch `financial_data.db` from a test.
- **Provider tests** monkeypatch `yf.download` / `yf.Ticker` at the module path
  ([test_yfinance_provider.py](../tests/test_yfinance_provider.py)) or inject a fake provider class
  ([test_backtest_service.py](../tests/test_backtest_service.py)). No test may hit the network.
- **Router tests** override dependencies rather than patching internals:
  `app.dependency_overrides[get_price_service] = lambda: FakeService()`
  ([test_prices_api.py](../tests/test_prices_api.py)).

The price layer is covered by six dedicated modules — see §5 of [docs/PRICING.md](PRICING.md).

---

## 11. Where the debt is

- [docs/PRICING.md](PRICING.md) — the price layer's audit. The two findings that could produce
  silently wrong backtest numbers are fixed; six operational and API-shape findings remain open.
- [docs/CODE_QUALITY_REVIEW.md](CODE_QUALITY_REVIEW.md) — structural review against `CLAUDE.md`,
  with a prioritised backlog.
