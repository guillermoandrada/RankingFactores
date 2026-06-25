# Portfolio Rebalancing System — Specification

## 1. System Overview

This is a **rules-based, top-down / bottom-up equity portfolio rebalancer** for a subset of the S&P 500 universe.

Given:
- A current portfolio of equity positions (tickers + share counts + cash)
- A scored investment universe (S&P 500 companies ranked by a composite quality score, organised by industry)
- A set of qualitative filters (ESG, earnings quality)
- A target industry-level weight allocation

The system produces a list of **sell trades** and **buy trades** that bring the portfolio in line with the target, subject to position-level and industry-level constraints. All operations run sequentially on a single portfolio object, mutating it in place.

---

## 2. Data Model

### 2.1 Security

Represents one investable company.

| Field | Type | Default | Description |
|---|---|---|---|
| `ticker` | `str` | required | Uppercase ticker symbol, max 5 alphanumeric characters |
| `name` | `str \| None` | `None` | Company display name |
| `industry` | `str \| None` | `None` | One of the 25 recognised industries (see Section 5) |
| `score` | `float \| None` | `None` | Composite quality score on a 0–10 scale |
| `ethics_filter` | `bool` | `True` | `False` if the security fails the ESG screen |
| `reuters_filter` | `bool` | `True` | `False` if Earnings Quality Country Rank ≤ 50 |

A security with `industry = None` is considered **outside the investment universe**.

### 2.2 Position

Represents a holding of one security in the portfolio.

| Field | Type | Default | Description |
|---|---|---|---|
| `security` | `Security` | required | The security being held |
| `quantity` | `float` | required | Number of shares held |
| `price` | `float \| None` | `None` | Current market price per share |

Key computed values:
- `amount` = `quantity × price`
- `weight(portfolio_value)` = `amount / portfolio_value`

Weight modification: given a `weight_change` (signed, as a fraction of total portfolio value) and the current `portfolio_total_value`, the quantity delta is:

```
delta_quantity = round(weight_change × portfolio_total_value / price, 0)
quantity += delta_quantity
```

The method returns a new Position object representing only the **delta** (quantity = `|delta_quantity|`, same price), which is used as the trade record.

### 2.3 Portfolio

Represents the entire managed account.

| Field | Type | Default | Description |
|---|---|---|---|
| `positions` | `List[Position]` | required | All current equity positions |
| `cash` | `float` | `0.0` | Uninvested cash (in the same currency as prices) |

Key computed values:
- `total_value` = sum of all position amounts + cash
- `get_position_weights()` → `dict[ticker → weight]`
- `get_industry_weights()` → `dict[industry → aggregate weight]`
- `get_industry_positions(industry)` → filtered list of positions
- `get_lowest_scoring_position(industry)` → Position with minimum `security.score` in that industry

When a position is **sold** (full close): it is removed from `positions` and its `amount` is added to `cash`.  
When a position is **partially sold**: its quantity is reduced in place and the sold `amount` is added to `cash`.  
When a position is **bought**: `cash` is reduced by `amount_to_buy`; if the ticker is already in `positions` the quantity is incremented, otherwise a new Position is appended.

---

## 3. Input File Specifications

All input files are Excel (`.xlsx`). File paths shown are the defaults used by the reference implementation.

### 3.1 Holdings Statement — `inputs/holdings_statement.xlsx`

Current portfolio snapshot.

| Column index | Name | Type | Description |
|---|---|---|---|
| 0 | Ticker | str | Uppercase ticker. Special value `CASH_USD` sets the portfolio cash balance. |
| 1 | Quantity | float | Number of shares (or cash amount for `CASH_USD`). |

No header skipping. One row per position. The last row is **not** a summary row.

Known ticker aliases resolved at load time:

| Raw ticker | Resolved ticker |
|---|---|
| `GOOG` | `GOOGL` |
| `NXP` | `NXPI` |

Validation rule: every non-cash ticker must be alphanumeric and at most 5 characters; otherwise raise an error.

### 3.2 Scoring File — `inputs/scoring.xlsx`

The investment universe with quality scores.

- **25 sheets**, one per industry. Sheet names may be truncated (see Section 5 for the full name mapping).
- **3 header rows** at the top of each sheet (skip with `skiprows=3`).
- Only the first 4 columns of each sheet are used:

| Column index | Name | Type | Description |
|---|---|---|---|
| 0 | Ticker | str | Uppercase ticker |
| 1 | Company Name | str | Display name |
| 2 | Market Cap (USD) | float | Not used by the rebalancing logic |
| 3 | Total Score | float | Composite quality score (0–10 scale). This is the primary ranking signal. |

All 25 sheets are loaded into a flat list of Security objects, each tagged with the industry derived from their sheet name.

### 3.3 Industry Allocation — `inputs/allocation.xlsx`

Target portfolio weights by industry.

| Column | Name | Type | Description |
|---|---|---|---|
| 0 | Industry | str | Full industry name (must match the 25 recognised industries) |
| 1 | Weight | float | Target weight as a decimal (e.g. `0.0453` = 4.53%) |

The **last row** is a summary/total row and must be dropped before use (`iloc[:-1]`). The weights across all 25 industries should sum to approximately 1.0.

### 3.4 Ethical Filter — `inputs/ethics_filter.xlsx`

Binary ESG evaluation per security.

- Sheet name: `S&P500`
- Skip first 3 rows (`skiprows=3`)

| Column | Name | Type | Description |
|---|---|---|---|
| — | Ticker | str | Ticker symbol. Any `.` in the ticker is replaced with `/` before matching. |
| — | Ethical Evaluation | str | `"Yes"` (passes) or `"No"` (fails) |

A security is marked `ethics_filter = False` if its ticker maps to `"No"` in this file.

### 3.5 Reuters Quality Filter — `inputs/reuters.xlsx`

Earnings quality ranking per security.

- Single sheet (first sheet).

| Column | Name | Type | Description |
|---|---|---|---|
| — | Identifier (RIC) | str | Reuters RIC code, e.g. `AAPL.O`. The ticker is extracted by splitting on `.` and taking the first part. |
| — | Earnings Quality Country Rank, Current | float | Numeric rank. |

A security is marked `reuters_filter = False` if its rank is **≤ 50**.  
The filter can be disabled globally (all securities pass by default when disabled).

---

## 4. Rebalancing Parameters

| Parameter | Value | Description |
|---|---|---|
| `MAXIMUM_POSITION` | `0.05` (5%) | Hard ceiling for any single position's portfolio weight |
| `NEUTRAL_POSITION` | `0.03` (3%) | Target weight when buying a new position or trimming an oversized one |
| `LIMIT_QUANTILE` | `0.50` (50th percentile) | Score threshold below which a position is sold (per industry) |
| `MAX_INDUSTRY_DIFFERENCE` | `0.003` (0.3%) | Minimum deviation from target that triggers a rebalancing trade |
| Reuters `MIN_SCORE` | `50` | Earnings Quality rank threshold (strict: rank must be > 50 to pass) |

---

## 5. Industry Universe (25 Industries)

The full set of recognised industries. Sheet names in `scoring.xlsx` may be truncated; the mapping from truncated sheet name to full name is:

| Sheet name (truncated) | Full industry name |
|---|---|
| `Household & Personal Products` | Household & Personal Products |
| `Food, Beverage & Tobacco` | Food, Beverage & Tobacco |
| `Consumer Staples Distribution &` | Consumer Staples Distribution & Retail |
| `Automobiles & Components` | Automobiles & Components |
| `Consumer Discretionary Distribu` | Consumer Discretionary Distribution & Retail |
| `Consumer Durables & Apparel` | Consumer Durables & Apparel |
| `Consumer Services` | Consumer Services |
| `Energy` | Energy |
| `Health Care Equipment & Service` | Health Care Equipment & Services |
| `Pharmaceuticals, Biotechnology ` | Pharmaceuticals, Biotechnology & Life Sciences |
| `Capital Goods` | Capital Goods |
| `Transportation` | Transportation |
| `Commercial & Professional Servi` | Commercial & Professional Services |
| `Telecommunication Services` | Telecommunication Services |
| `Media & Entertainment` | Media & Entertainment |
| `Financial Services` | Financial Services |
| `Insurance` | Insurance |
| `Banks` | Banks |
| `Technology Hardware & Equipment` | Technology Hardware & Equipment |
| `Software & Services` | Software & Services |
| `Semiconductors & Semiconductor ` | Semiconductors & Semiconductor Equipment |
| `Utilities` | Utilities |
| `Materials` | Materials |
| `Equity Real Estate Investment T` | Equity Real Estate Investment Trusts (REITs) |
| `Real Estate Management & Develo` | Real Estate Management & Development |

---

## 6. Rebalancing Algorithm

Operations run **sequentially** in the order below. Each step mutates the portfolio in place and returns a list of trade records (Positions) for reporting. Portfolio `total_value` is recalculated at each step that needs it.

### Step 1 — Ethical Filter (full closes)

Sell every position where `security.ethics_filter == False`.

- Full position close: remove from `positions`, add `amount` to `cash`.
- Rationale label: `"Trades por filtro etico"`

### Step 2 — Reuters Quality Filter (full closes)

Sell every position where `security.reuters_filter == False`.

- Full position close.
- Rationale label: `"Trades por no superar 50 en el ranking Reuters"`

### Step 3 — Universe Filter (full closes)

Sell every position where `security.industry is None`.

- Full position close.
- Rationale label: `"Trades por universo de inversion"`

### Step 4 — Concentration Cap (partial sells)

For every position where `position.weight(portfolio_value) > MAXIMUM_POSITION`:

```
excess_weight = current_weight - NEUTRAL_POSITION   # always positive
delta_quantity = round(excess_weight × portfolio_value / price, 0)
position.quantity -= delta_quantity
portfolio.cash += delta_quantity × price
```

The trade record is a Position with `quantity = delta_quantity` and the same price.  
Rationale label: `"Trades por exceso de concentracion"`

### Step 5 — Low-Score Filter (full closes)

For every remaining position:

1. Compute the 50th-percentile score across **all securities in the same industry** from the scoring universe (not just those currently held).
2. If `position.security.score < industry_50th_percentile`: full close.

Rationale label: `"Trades por bajo scoring"`

### Step 6 — Top-Down Sales (industry overweight correction)

For each industry where:

```
current_industry_weight - target_industry_weight > MAX_INDUSTRY_DIFFERENCE
```

Execute the following loop (using the portfolio value snapshot taken before this step):

```
remaining_overweight = current_weight - target_weight

while remaining_overweight > MAX_INDUSTRY_DIFFERENCE + 0.0001:
    position = lowest_scoring_position_in(industry)
    w = position.weight(snapshot_portfolio_value)

    if w <= round(remaining_overweight, 4):
        # sell entire position
        full_close(position)
        remaining_overweight -= w
    else:
        # partial sell of exactly the overweight
        partial_sell(position, weight_to_sell=remaining_overweight)
        remaining_overweight = 0
```

Rationale label: `"Trades por overweight en industria"`

### Step 7 — Bottom-Up Purchases (industry underweight correction)

For each industry where:

```
target_industry_weight - current_industry_weight >= MAX_INDUSTRY_DIFFERENCE
```

Execute the following loop (using the portfolio value snapshot taken before this step):

```
remaining_underweight = target_weight - current_weight
rank = 1

while remaining_underweight >= MAX_INDUSTRY_DIFFERENCE:
    security = nth_ranked_security_in(industry, rank)   # ranked by score descending

    if security is None:
        # no more candidates in this industry
        break

    if not security.ethics_filter or not security.reuters_filter:
        rank += 1
        continue

    weight_to_buy = min(NEUTRAL_POSITION, remaining_underweight)

    existing_position = portfolio.get_position(security.ticker)

    if existing_position is None:
        # new position
        buy at weight_to_buy
        remaining_underweight -= bought_weight
    else:
        # top up existing position
        current_weight = existing_position.weight(portfolio.total_value())
        weight_to_add = weight_to_buy - current_weight
        if weight_to_add > 0:
            buy at weight_to_add
            remaining_underweight -= weight_to_add

    rank += 1
```

Price for each new buy is fetched live from Yahoo Finance at the moment of purchase.  
Quantity purchased = `round(amount_to_buy / price, 0)` (whole shares only).  
`portfolio.cash -= amount_to_buy`

Rationale label: `"Compras por underweight en industria"`

---

## 7. Pricing

Prices are fetched using **Yahoo Finance** (`yfinance` library):

- Initial pricing: all portfolio positions are priced at once before any operations run.
- Buy pricing: each new security being purchased is priced individually at the moment of the buy call.
- If a price cannot be fetched automatically, the system may prompt the user to enter it manually.
- A date can be specified to fetch historical prices (used by the backtest module).

---

## 8. Output Specification

The output is a single Excel file with 6 sheets.

### Sheet 1 — Portfolio Inicial

Portfolio snapshot **before** any trades.

| Column | Description |
|---|---|
| Ticker | Security ticker |
| Name | Company name |
| Industry | Assigned industry |
| Weight | Position weight in portfolio (decimal) |
| Price | Market price per share |
| Quantity | Number of shares |
| Amount | Market value (quantity × price) |

One row per position. An additional `CASH` row represents uninvested cash.

### Sheet 2 — Ventas (Sales)

All sell trades, in execution order, with a `Rationale` column identifying the rule that triggered the sale.

| Column | Description |
|---|---|
| Ticker | — |
| Name | — |
| Industry | — |
| Weight | Weight **at the time of sale** |
| Price | Price at sale |
| Quantity | Shares sold (always positive) |
| Amount | Proceeds |
| Rationale | One of the 6 sell labels (see Section 6) |

### Sheet 3 — Compras (Purchases)

All buy trades.

| Column | Description |
|---|---|
| Ticker | — |
| Name | — |
| Industry | — |
| Weight | Weight of the purchased tranche |
| Price | Price at purchase |
| Quantity | Shares bought |
| Amount | Cost |
| Rationale | `"Compras por underweight en industria"` |

### Sheet 4 — Portfolio Final

Portfolio snapshot **after** all trades. Same columns as Sheet 1.

### Sheet 5 — Diferencias Industrias (Industry Differences)

| Column | Description |
|---|---|
| Portfolio | Current weight of each industry in the final portfolio (%) |
| Objetivo | Target weight from `allocation.xlsx` (%) |
| Diferencia | Portfolio − Objetivo (%) |

Row index is the industry name. One row per industry in the target allocation.

### Sheet 6 — Universo de Inversion (Investment Universe)

Full security universe from `scoring.xlsx`.

| Column | Description |
|---|---|
| Ticker | — |
| Name | — |
| Industry | — |
| Score | Composite quality score |
| Ethics | `True` / `False` (ethics_filter value) |
| Reuters | `True` / `False` (reuters_filter value) |
| In Portfolio | `True` if the ticker appears in the final portfolio |

---

## 9. Backtest Variant

A separate backtest module replays the rebalancing algorithm quarterly using historical inputs.

- Historical scoring files are stored in timestamped subdirectories under `inputs/historico/`.
- Historical industry allocations are stored in `inputs/inputs backtest/ind_hist.xlsx` (index = industry, columns = quarter-end dates).
- Historical prices are stored in `inputs/inputs backtest/prices_db.xlsx` and supplemented by Yahoo Finance for missing data.
- For each quarter: load the scoring file for that period → reclassify the portfolio → price at quarter-end → run all 7 operations → record portfolio value.
- Output is `backtest_results.xlsx` with one sheet per quarter (named `YYYY_MM_DD`) showing portfolio composition, plus a summary sheet with quarter return, cumulative return, total return, annualised return, trade count, and turnover.
