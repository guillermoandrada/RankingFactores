# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI backend
uvicorn api.main:app --reload

# Run Streamlit frontend (requires backend running separately)
streamlit run streamlit_app/Home.py

# Import data from Excel/CSV
python run_import.py
python run_import.py "path/to/file.xlsx"

# Run tests
pytest

# Run a single test file
pytest tests/test_portfolio.py
```

## Architecture

**RankingFactores** is a financial security ranking and portfolio construction platform. It ingests fundamental financial metrics (from Bloomberg/Reuters Excel exports), computes factor-based security rankings, analyzes factor predictiveness (Information Coefficient), constructs portfolios, and backtests strategies.

### Three-tier structure

1. **FastAPI backend** (`api/`) — REST API with a service layer; Streamlit calls this at runtime.
2. **Streamlit frontend** (`streamlit_app/`) — Multi-page UI; pages map to the workflow: Periods → Metrics → IC Analysis → Scoring Profiles → Ranking → Portfolio → Backtest.
3. **Core modules** (`modules/`) — All business logic; consumed by the API services.

### Data flow

```
Excel/CSV → FileReader (Bloomberg/Reuters) → DataImporter (validate, normalize)
  → FinancialDatabase (SQLite via SQLAlchemy)
  → ZScoreCalculator → RankingEngine (weighted scores)
  → PortfolioService (smart-beta or long/short)
  → BacktestService (rolling performance)
```

### Key modules

| Module | Responsibility |
|---|---|
| `modules/db/` | SQLite repository — securities, metrics, fundamentals, period-scoped classifications |
| `modules/ingestion/` | File readers (Bloomberg, Reuters) + DataImporter orchestrator |
| `modules/analytics/zscore.py` | Winsorized z-score normalization, filterable by index/industry |
| `modules/analytics/ranking.py` | Combines z-scores via linear or softplus into a final score |
| `modules/analytics/ic_analyzer.py` | Spearman rank correlation of factors vs forward returns |
| `modules/config/ranking_profiles.py` | JSON-persisted scoring profiles (weights, metrics, method) |
| `modules/config/derived_metrics.py` | Computed metrics defined as formulas (e.g. Debt/Assets) |
| `modules/portfolio/` | Smart-beta and long/short portfolio builders + rebalancing |
| `modules/backtesting/` | Time-series backtesting across rolling periods |
| `api/dependencies.py` | Singleton factory — all services and repositories are injected here |
| `api/routers/` | Thin HTTP routers (periods, metrics, scorings, scoring-profiles, portfolios, backtests, reference, db_metrics) |
| `api/services/` | Business logic layer called by routers |
| `streamlit_app/api_client.py` | HTTP client wrapping every FastAPI endpoint for the UI |

### Configuration

`config.json` at the project root controls the SQLite database URL (`financial_data.db`), fixed required columns, and the default input file. The committed `financial_data.db` contains default data and is intentionally tracked in git.

### Market data

`modules/analytics/base_price_provider.py` defines the pluggable `BasePriceProvider` interface. The only concrete implementation is `YFinanceService` (wrapping `yfinance`), used for ticker validation and historical return retrieval.

---

## Coding principles (from AGENTS.md)

**Symmetry is a first-class constraint.** Before implementing anything, check whether analogous modules (sibling routers, schemas, services, tests, UI pages) should follow the same pattern. Introduce asymmetry only when necessary, and explain it in the final summary.

**Architecture expectations:**
- Identify and extend the existing local pattern rather than inventing a new one.
- Keep routers thin: parse input → delegate to service → translate errors to HTTP responses.
- Business logic lives in services, not routers. Persistence lives in the repository, not in services.
- Depend on abstractions (e.g. `BasePriceProvider`) over concrete implementations.

**API defaults** (4–5 endpoints per router, CRUD-aligned):
- `GET /resources` → list
- `GET /resources/{id}` → retrieve one
- `POST /resources` → create (returns `201`)
- `PUT /resources/{id}` → update (returns `200` or `204`)
- `DELETE /resources/{id}` → delete (returns `204`)

**Implementation steps for every non-trivial task:**
1. Inspect nearby files and identify the dominant repository pattern.
2. Reuse existing architecture and naming conventions.
3. Check sibling modules for symmetry opportunities.
4. Implement the smallest clean solution that fits the architecture.
5. Update or add tests in the same style as existing tests.
6. Summarize what changed, which pattern was followed, which symmetric counterparts were checked, and any intentional asymmetry.

**Code style:** type hints where the codebase already uses them, explicit errors over silent fallbacks, no premature abstraction, no hidden side effects.
