# RankingFactores — User Guide

How to build your own factors and backtest them, using the Streamlit app.

No programming required. This guide follows the app's pages in the order you will actually use them.
If you maintain the code, see [ARCHITECTURE.md](ARCHITECTURE.md) instead.

---

## Contents

1. [The workflow at a glance](#1-the-workflow-at-a-glance)
2. [Starting the app](#2-starting-the-app)
3. [Load your data — *Periods*](#3-load-your-data--periods)
4. [Inspect your metrics — *Metric Diagnostics*](#4-inspect-your-metrics--metric-diagnostics)
5. [Set metric behaviour — *Periods → View & Edit*](#5-set-metric-behaviour--periods--view--edit)
6. [Add and build metrics — *Metrics*](#6-add-and-build-metrics--metrics)
7. [Pick which metrics are worth using — *Metric Selection (IC)*](#7-pick-which-metrics-are-worth-using--metric-selection-ic)
8. [Build a scoring profile — *Scoring Profile Wizard*](#8-build-a-scoring-profile--scoring-profile-wizard)
9. [Refine a profile — *Scoring Profiles*](#9-refine-a-profile--scoring-profiles)
10. [Rank the universe — *Ranking*](#10-rank-the-universe--ranking)
11. [Build a portfolio — *Portfolio Construction*](#11-build-a-portfolio--portfolio-construction)
12. [Backtest — *Portfolio backtest* and *Strategy Backtest*](#12-backtest)
13. [Supply your own prices — *Price Data*](#13-supply-your-own-prices--price-data)
14. [Worked example, start to finish](#14-worked-example-start-to-finish)
15. [Troubleshooting](#15-troubleshooting)
16. [Parameter reference](#16-parameter-reference)

---

## 1. The workflow at a glance

```
    Periods            upload fundamentals for one date  ──┐
      │                                                    │
    Metric Diagnostics  is this metric clean enough?        │  data preparation
      │                                                    │
    Periods → Edit      higher-is-better? missing values?   │
      │                                                    │
    Metrics             combine metrics into new ones     ──┘
      │
    Metric Selection    which metrics actually predict?   ──┐
      │                                                    │  factor design
    Scoring Wizard      group metrics into factors          │
      │                                                    │
    Scoring Profiles    adjust weights, validate, preview ──┘
      │
    Ranking             score and rank the universe       ──┐
      │                                                    │  portfolio & test
    Portfolio Constr.   turn ranks into weights             │
      │                                                    │
    Backtest            simulate against a benchmark      ──┘
```

Vocabulary used throughout:

| Term | Meaning |
|---|---|
| **Period** | One snapshot of fundamental data, labelled by date (`2024/12/31`) or quarter (`2024 Q4`). Everything is scoped to a period. |
| **Metric** | One raw fundamental column (`Current ROA`). Either uploaded or *derived*. |
| **Factor** | A named group of metrics with weights (`Quality = 0.5·ROA + 0.5·ROIC`). |
| **Scoring profile** | The full recipe: the factor tree, plus how metrics are normalised and combined. Reusable across periods. |
| **Score** | The single number each security ends up with. Its **scale depends on the normalisation you choose** — see §8.3, this matters more than it sounds. |
| **Strategy** | How scores become portfolio weights (`legacy_rebalance`, `smart_beta`, `long_short`). |

---

## 2. Starting the app

Two processes must be running. In two terminals, from the project folder:

```bash
uvicorn api.main:app --reload            # the engine, on port 8000
streamlit run streamlit_app/Home.py      # the interface, on port 8501
```

Open <http://localhost:8501>.

**Every page has an `API Base URL` box in the left sidebar**, defaulting to `http://127.0.0.1:8000`. Leave
it alone unless the engine runs elsewhere. Most pages also have a **Test API connection** button in the
sidebar — click it first. If it fails, the engine is not running and nothing else on the page will work.

The **Home** page shows how much data you have: periods, securities, metrics, derived metrics, saved
profiles, plus per-period coverage. A fresh install shows zeros — start at *Periods*.

> **Note on state.** Results live in your browser session. Switching pages keeps them; refreshing the
> browser clears them. Nothing is lost permanently — periods, metrics and profiles are all saved
> server-side — but rankings and backtests must be re-run after a refresh.

---

## 3. Load your data — *Periods*

Three tabs: **Create**, **View & Edit**, **Delete**.

### 3.1 Create

Pick the **Reader** matching your file, upload it, click **Create period**.

| Reader | What the file must contain | Period comes from |
|---|---|---|
| **Bloomberg** | Header row on Excel row 4; a `Ticker` column plus `Long Name`, `GICS Sector Name`, `GICS Industry Group Name`, `Market Cap (USD)`; metric columns after those. Optionally a cell labelled *Universe Name* with the index name below it. | Cell **A2** of the file |
| **BQL** | Sheets `Name`, `Classification`, `Current`, `Past`, `Estimated`, `Config`. Factors are read from the three data sheets. | Cell **B1** of the `Config` sheet (universe from `B2`) |
| **Reuters Metrics** | An `Identifier` or `Identifier (RIC)` column and an `Earnings Quality Country Rank, Current` column. The header row is auto-detected in the first 10 rows. Other columns are ignored. Stored as a single metric named **`Reuters Score`**. | **You type it** — the file does not contain it |
| **Bloomberg Individual Variable** | One variable for **several periods at once**: five columns per period (ticker, name, both GICS levels, value) or two (ticker, value). Fills many periods in one upload rather than creating one — see §3.2. | Row 1 of each block |

**If period exists** (Bloomberg and BQL):
- `replace` — wipe that period and load the file fresh. Use this when re-importing a corrected file.
- `append` — merge into the existing period. New metrics and new securities are added; where the same
  security/metric already exists, the uploaded value wins.

**Reuters uploads** ask instead for an *Import target*: create/replace a named period, or **append to an
existing period**, which merges `Reuters Score` into that period by ticker. Use append when you want the
Reuters score to sit alongside Bloomberg fundamentals in the same period.

On success you get a green banner with companies, metrics and record counts. Check them — a much lower
company count than expected usually means the header row was misread.

### 3.2 Upload a single variable across periods

Reader **Bloomberg Individual Variable**. Use this when you have *one* variable — volatility, beta, a
custom Bloomberg field — exported for several periods at once, instead of a full period file. Upload
it, click **Upload variable**.

Row 1 holds the index name and the period date; row 2 holds vendor field labels and is skipped
entirely, so formula errors such as `#NAME?` there are harmless; data starts on row 3. **The columns
you supply decide whether the import may create securities**, and the reader works out which layout
you used — you do not tell it.

**Wide layout — five columns per period.** Creates securities the database has not seen:

| | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| **1** | `SPX Index` | `6/30/2026` | | | | `SPX Index` | `3/30/2026` |
| **2** | *(labels — ignored)* | | | | | | |
| **3+** | `AAPL US Equity` | `Apple Inc` | `Information Technology` | `Hardware` | `0.25` | `AAPL US Equity` | … |

**Narrow layout — two columns per period.** Appends to existing securities only:

| | A | B | C | D |
|---|---|---|---|---|
| **1** | `SPX Index` | `6/30/2026` | `SPX Index` | `3/30/2026` |
| **2** | `ID` | `volatility(calendar)` | `ID` | `volatility(calendar)` |
| **3+** | `AAPL US Equity` | `0.25` | `AAPL US Equity` | `0.31` |

| | Wide (5 columns) | Narrow (2 columns) |
|---|---|---|
| Creates a security the DB lacks | **yes** | **never** — reported and skipped |
| Period sector/industry | refreshed from the file | untouched |
| Security name | filled if blank | untouched |
| Unknown tickers | imported | listed in the warning |

Use narrow when the universe is already loaded and you only want to add a column. Use wide when the
file introduces securities — a name and both GICS levels are the minimum needed to create one, which
is why the narrow form cannot. If *nothing* in a narrow file matches, the upload is rejected rather
than silently importing zero rows.

- **The sheet name is the metric name** — a sheet called `Volatility 12m` creates the metric
  `Volatility 12m`. Rename the sheet before uploading if you want a different name. Leave the
  **Sheet** box empty to read the first sheet.
- The date in row 1 becomes the period (`2026/06/30`), so one file fills several periods at once.
  **Periods that do not exist yet are created** in both layouts.
- Tickers keep the vendor's spelling with the market suffix removed (`AAPL US Equity` → `AAPL`).
- Each period is imported the same way as an **append** period upload: this variable replaces its
  own previous values and other metrics of the period are untouched. Stored market caps are left
  alone in both layouts — neither carries a market cap column.
- Every period block must use the same layout. A file that mixes them, or that has blank spacer
  columns between periods, is rejected with a message naming the column it tripped on.
- A new metric starts as *higher is better* with N/A treatment *replace with zero* — change it in
  **Periods → View & Edit** (§5).
- The index name in row 1 is reported back for checking only; it never changes index membership,
  because a variable file may cover only part of a universe.
- Periods are written one at a time. If one fails, the periods already written stay written; the
  green banner lists exactly what was committed.

Beware the reverse order: re-importing a period with **replace** (§3.1) wipes the variable values
in that period. Use **append** if you have already uploaded variables into it.

### 3.3 View & Edit

Select a period → **Load content**. You get a spreadsheet of every security × metric, with `ticker`,
`name`, `sector`, `industry` first.

- **Edit any cell**, then **Save changes**. Only changed numeric cells are sent.
- **Remove a security from this period** — drops one company from this period only.
- **Remove a metric from this period** — drops one metric column from this period only.
- **Metric parameters** — see §5. This is the important part of this tab.

### 3.4 Delete

Removes the period entirely: all fundamental values, classifications and index membership. Metric
*definitions* survive, which has a consequence worth knowing — see the warning in §8.2.

---

## 4. Inspect your metrics — *Metric Diagnostics*

Do this **before** designing a factor. Pick a metric, choose a **Breakdown** (Total / By sector / By
industry), click **Compute**.

You get one row per period with colour-coded columns:

| Column | Meaning | How to read the colours |
|---|---|---|
| **Rows** | Securities with a row for this metric in that period | — |
| **% N/A** | Share of those rows with no value | 🟢 ≤5 % · 🟡 ≤25 % · 🔴 >25 % |
| **Min / Max** | Raw extremes | Absurd values here mean unit or sign errors upstream |
| **Skew** | `(Mean − Median) / MAD` | 🔵 left tail · ⚪ symmetric · 🔴 right tail. `\|value\| ≥ 1` = strong |
| **Std** | Standard deviation | — |
| **Std/MAD** | Spread vs robust spread — an outlier detector | 🟢 ≤2 · 🟡 ≤4 · 🔴 >4 |
| **% Outliers** | Share outside Tukey's 1.5·IQR fence | 🟢 ≤5 % · 🟡 ≤15 % · 🔴 >15 % |

Expand **Show mean, median, percentiles & split outliers** for the 1st/5th/10th/90th/95th/99th
percentiles and outliers split above vs below.

**What to do with what you find:**

| Finding | Action |
|---|---|
| Red **% N/A** | Choose the N/A treatment deliberately (§5), or drop the metric. A metric that is 40 % missing and filled with zeros is mostly noise. |
| Red **Std/MAD** or **% Outliers** | Turn winsorisation on in the profile (§8.2) and consider tightening it to 5 %/95 %. |
| Strong **Skew** | Prefer **Percentile rank** normalisation — it is immune to shape. Z-scores on a heavily skewed metric concentrate almost every name on one side. |
| Wild **Min/Max** | Fix the source data. No normalisation setting rescues a unit error. |
| Very different stats **by sector** | Score within sector scopes instead of across the whole universe (§10). |

---

## 5. Set metric behaviour — *Periods → View & Edit*

At the bottom of the View & Edit tab, each metric has two global settings. **These apply across all
periods**, not just the one loaded.

**Higher is better** — the direction of the metric.
- *Higher is better* → high raw values raise the score.
- *Lower is better* → the sign is flipped, so high raw values **lower** the score. Use this for
  debt ratios, P/E, volatility.
- *Unset* → treated as higher-is-better.

Getting this wrong silently inverts a factor: your "quality" score rewards the worst companies and the
ranking looks plausible. Set it explicitly for every metric you use.

**N/A treatment** — what happens to missing values, applied before normalisation:

| Option | Effect | When to use |
|---|---|---|
| `Replace with zero` | Missing → 0 | Default. Careful: 0 is *not* neutral in raw units. |
| `Replace with high` | Missing → a high value | Rarely — only if missing genuinely means good |
| `Replace with low` | Missing → a low value | Penalise non-disclosure |
| `Eliminate rows with N/A` | Drop the security entirely for that period | Cleanest for a must-have metric; shrinks your universe |
| `Unset` | No special handling | — |

Select **Keep current** for anything you do not want to change, then **Save metric parameters**.

> Metrics created automatically by an import default to *higher is better* + *replace with zero*.
> Review them rather than assuming.

---

## 6. Add and build metrics — *Metrics*

Three tabs: **Create**, **Get & Edit** and **Delete**, all for derived metrics. Loading a variable
from Excel happens in **Periods → Create** (§3.2), not here.

### 6.1 Build a derived metric

For ratios and spreads your data provider did not give you.

In **Create**:
1. Pick **Metric 1** and **Metric 2** (both existing metrics — raw or derived).
2. Choose the operator between them: `+ − * /`.
3. **+ Add metric** for a longer chain; **- Remove last** to shorten it.
4. Check the **Formula preview**.
5. Name it. The default name is the metrics joined by `/`; overwrite it with something readable.
6. Set **Higher values are better** and **Handle missing values** — same meaning as §5.
7. **Create metric**.

**Operators apply strictly left to right — there is no precedence and no brackets.**
`A + B / C` is evaluated as `(A + B) / C`, not `A + (B / C)`. To get `A + (B / C)`, first create
`B / C` as its own derived metric, then add `A` to it.

Derived metrics are stored as *formulas* and recomputed whenever they are used, so they automatically
cover every period where their inputs exist. They appear alongside raw metrics everywhere else in the
app. Deleting one removes only the formula.

---

## 7. Pick which metrics are worth using — *Metric Selection (IC)*

This page answers two questions: *does this metric predict returns?* and *am I double-counting?*

Select **at least two** metrics, choose a **Forward horizon** (1, 3 or 6 months), click
**Run IC analysis**. This fetches prices and can take a while.

### How the calculation works

For each period, the app takes the cross-section of securities, ranks them by the metric, ranks them by
their **forward return**, and computes the Spearman correlation between the two rankings — the
**Rank IC**. The forward window starts **45 days after the period end** (a publication lag, so you are
not using data that was not yet public) and runs for the horizon you chose. Repeating this across
periods gives a time series of ICs, summarised as:

| Column | Meaning | Rough reading |
|---|---|---|
| **Mean Rank IC** | Average predictive correlation | 0.02–0.05 is normal for a single fundamental factor; >0.10 is strong |
| **IC Standard Deviation** | How much it varies period to period | Lower is more dependable |
| **Information Ratio** | Mean IC ÷ IC std | The signal-quality number. Higher = more consistent |
| **Periods (T)** | How many periods contributed | **With T = 1 or 2 these numbers mean nothing.** |

Sign matters: a **negative** mean IC means high values of the metric preceded *low* returns. That is
still information — flip *Higher is better* (§5) rather than discarding the metric.

### Section B — inter-factor correlation

A heat map of mean Spearman correlation between the selected metrics, computed on the shared
cross-section (securities that have all of them). **Absolute correlation above ≈0.7 means redundancy** —
keep the one with the better Information Ratio and drop the other, or you will silently double-weight
the same bet.

Expand **Periods used in this IC analysis** to see how many periods each metric actually contributed.
A metric with far fewer used periods than the others is being evaluated on different history, so the
comparison is not like-for-like.

> **Prerequisites.** IC needs price history, so it needs either internet access to Yahoo Finance or
> prices you uploaded yourself (§13). It also needs **several periods** — with one period loaded the page
> runs but tells you nothing. Load history first.

---

## 8. Build a scoring profile — *Scoring Profile Wizard*

Three steps, with **← Back** / **Next →** at the bottom.

### 8.1 Step 1 — Transform chain

How each individual metric is treated before anything is combined.

**Use winsorization** (on by default) clamps extreme values so a handful of outliers cannot dominate:
- **Quantile winsorization** — values below the lower percentile and above the upper are pulled to
  those boundaries. Defaults 0.01 / 0.99. Tighten to 0.05 / 0.95 for metrics flagged red in §4.
- **Semi winsorization** — clamps to `mean ± k·σ`, default `k = 3.0`.

**Terminal transform** — the normalisation that puts every metric on a comparable scale:

| Option | Output scale | Notes |
|---|---|---|
| **Standardize (z-score)** | ≈ −3 … +3, mean 0 | Preserves relative distances. Sensitive to skew. |
| **Normalized z-score** | 0 … 10 | Z-score linearly rescaled so the minimum maps to 0 and the maximum to 10. Because it is anchored on the extremes, a skewed metric squashes most names low. |
| **Percentile rank** | 0 … 100 | Pure ranking. Immune to skew and outliers; discards magnitude. |

**Aggregation method:**
- **Linear** — weighted sum. Predictable, additive; a strong metric can carry a weak one.
- **Softplus** — a smooth, geometric-like combination. Being weak on one input drags the whole score
  down, so it rewards all-round names rather than specialists.

### 8.2 Step 2 — Base structure

Build the factor tree with nested boxes. The root box is always called **Scoring**.

- **Add metric** — attach a metric with a weight to the current box.
- **Add subfactor** — nest a new box inside it. Name it (`Value`, `Quality`, `Momentum`), give it a
  weight in its parent, then add metrics or further subfactors inside. Up to 5 levels deep.
- **Method** per box — each box can use `linear` or `softplus` independently.

A typical two-level tree:

```
Scoring  (linear)
├── Value   weight 0.40   (linear)
│   ├── Current Earnings Yield   0.5
│   └── Current Book to Price    0.5
└── Quality weight 0.60   (linear)
    ├── Current ROA              0.5
    └── Current ROIC             0.5
```

**Weight convention.** Weights are multipliers, not shares — nothing normalises them for you. Making
each box's children sum to `1.0` keeps the final score on the same scale as the individual metrics,
which is what you want. If `Value` and `Quality` are weighted 0.4 and 0.6 they sum to 1.0 and the score
stays on the metric scale; weight them 0.4 and 0.4 and every score shrinks by 20 %.

> **Only pick metrics that exist in the period you will rank.** The dropdown lists every metric
> *definition* in the database, including ones whose data was deleted or that came from a different
> import. A metric with no rows in the period you rank does not raise an error — with the default
> `replace with zero` treatment it contributes **a column of zeros**, quietly diluting your score. In
> this repository's database, for example, 59 of 116 defined metrics have data for period `2024/12/31`.
> Cross-check against *Periods → View & Edit* (which lists only metrics actually present) or the
> *Metric Diagnostics* row count before you commit to a metric.

### 8.3 Step 3 — Review & save

Name the profile and review the generated JSON. Warnings appear for an empty name, empty factors or
empty weights; **Save scoring profile** stays disabled until they are cleared. Saving resets the wizard
for a fresh profile.

### ⚠️ Choose the normalisation with your strategy in mind

This is the single most common way to end up with an empty portfolio, so it is worth being explicit.

The `legacy_rebalance` strategy (§11.1) **refuses to hold anything scoring below 5.0** — a fixed floor.
Whether any security clears it depends entirely on the normalisation you picked in Step 1. Measured on
this repository's data (503 securities, period `2024/12/31`, a two-factor tree over ROA / ROIC / P/S):

| Normalisation | Resulting score range | Median | Securities clearing the 5.0 floor |
|---|---|---|---|
| **Standardize (z-score)** | −2.07 … 3.71 | −0.20 | **0 of 503** |
| **Normalized z-score** | 0.00 … 10.00 | 3.20 | 72 of 503 |
| **Percentile rank** | 0.70 … 99.50 | 48.76 | 497 of 503 |

Practical rules:

- **Using `legacy_rebalance`? Do not use plain z-score.** Every security is excluded with reason
  `score_below_5` and you get an all-cash portfolio. Use **Percentile rank** (recommended — the score
  reads as "percentile", and selection is then controlled by *Score quantile cutoff*) or
  **Normalized z-score** if you want the floor to bind harder.
- **Using `smart_beta` or `long_short`?** Any normalisation works. Both only use the *ordering* of
  scores, never the absolute level. Z-score is a fine default there.
- Changing normalisation changes the score scale, so **any absolute threshold you tuned must be
  revisited**.

---

## 9. Refine a profile — *Scoring Profiles*

The wizard creates; this page maintains. Select a **Profile to edit** (**Reload from API** discards
unsaved edits).

**Layout.** Tree navigator on the left, node editor on the right. Click any node to edit it.

**Node editor** lets you rename the node, set its **Method** (`linear` / `softplus`, overriding the
profile default), **+ Add metric**, **+ Add subfactor**, and edit each input's weight. **Weight tools**
help redistribute weights across a node's inputs. Subfactor children have a `→ name` button to jump
into them.

**Validation** runs continuously at the top of the page. Issues are clickable — they navigate straight
to the offending node. Fix everything before saving; a node with empty inputs will fail at ranking time.

**Normalization, Winsorization & Aggregation** (expander) holds the profile-wide settings from wizard
Step 1. You can change normalisation here at any time — re-read §8.3 before you do.

**Preview** shows the effect of your current settings without saving.

**Save changes** persists to the server. **Delete profile** cannot be undone.

> Profiles are independent of periods. One profile is applied to whatever period you rank, which is
> exactly what makes a multi-period backtest meaningful.

---

## 10. Rank the universe — *Ranking*

Choose **Period**, **Scoring profile**, and optionally an **Index** to restrict the universe to that
index's members for that period.

**Scope** decides how many rankings you get:

- **All** — one ranking over the whole (optionally index-filtered) universe.
- **Sector** — one ranking per sector, each scored *within* that sector. Use this when metrics are not
  comparable across sectors (bank margins vs software margins).
- **Industry** — same, one per industry.

With Sector or Industry scope you can **Add override** rows to apply a *different scoring profile* to a
specific sector or industry — a bank-specific profile for Financials, the default everywhere else. Each
override must target a unique group. **Clear all overrides** resets them.

**Run ranking.** Results appear as one collapsible section per scope, headed by the company count and
the profile used. Warnings (missing metrics, dropped rows) appear inside each section. Read them —
they explain gaps.

Columns: `ticker`, `name`, then one `… Score` column per metric and factor, and `Scoring` — the final
score, sorted descending.

**Export to Excel** concatenates every successful scope into a single worksheet, with a leading
sector/industry column when there is more than one.

Sanity-check the ranking before moving on. Recognise the names at the top? Does the score spread look
reasonable? If everything scores identically, a metric is probably all-zeros (§8.2).

---

## 11. Build a portfolio — *Portfolio Construction*

Turns a ranking into weights. Choose **Period**, **Scoring profile**, **Index**, **Sector**,
**Industry**, then:

**Strategy** — `legacy_rebalance`, `smart_beta` or `long_short` (§11.1–11.3).

**Construction mode:**
- **new_portfolio** — build from scratch. Uses **Capital base**.
- **rebalance_existing** — start from your current holdings and generate trades. Total capital comes
  from the uploaded holdings plus cash, *not* from Capital base.

### Current portfolio input (rebalance mode only)

Upload a holdings workbook. Expected layout, by **column position** (a header row is expected and skipped):

| Column | Content |
|---|---|
| A | Ticker |
| B | Quantity (or market value) |
| C | Price *(optional)* |
| D | Market value *(optional)* |

Use ticker **`CASH_USD`** for the cash line; its value is read from market value, cash or quantity.

If price and market value are both missing, click **Fetch latest prices**. This pulls the latest close
per ticker — from prices you uploaded (§13) first, falling back to Yahoo Finance — and fills price and
market value as `quantity × price`. Anything it could not price is listed in a warning.

**Price every equity line before building.** Total capital is the sum of holding values plus cash, so an
unpriced holding understates the portfolio and distorts *every* target weight and trade.

### External filters

**Ethical filter workbook** — an exclusion list. Requires a sheet named **`STANDARD&POOR'S500`** with
the header row on row 4, containing **`Ticker`** and **`Ethical Evaluation`** columns. Any row whose
evaluation reads **`No`** is excluded, and appears under *Diagnostics → Excluded* with reason
`blocked_by_filter`.

### Constraint targets

**Constraint type**: `None`, `Sector restrictions`, or `Industry restrictions`. You then get one row per
group with an **On** checkbox and a **Weight (%)**:

- **Off** — unrestricted; the strategy allocates freely.
- **On with a weight > 0** — target that share of the portfolio for the group.
- **On with weight 0.0** — explicit **no-buy**: nothing new is allocated to that group.

**Enable/Disable all** toggles every row at once. Only enabled rows are sent. Sector and industry
constraints are mutually exclusive.

### 11.1 `legacy_rebalance`

Reproduces the legacy Excel process: ranked allocation with a hard quality floor.

Eligibility, in order — a name must pass **all** of:
1. Not blocked by the ethical filter.
2. **Score ≥ 5.0** (fixed; see §8.3 — this is where a z-score profile empties your universe).
3. Score at or above its **industry** score quantile.

Eligible names are then allocated by rank within their group, up to `neutral_position` each, capped at
`max_position`, respecting group targets. Unallocated weight becomes cash.

| Parameter | Default | Meaning |
|---|---|---|
| **Max position** | 5.00 % | Hard cap per security. Holdings above it are trimmed. |
| **Neutral position** | 3.00 % | Standard allocation given to each newly selected name. |
| **Score quantile cutoff** | 50.00 % | Keep only names at or above this quantile **within their industry**. 0 % keeps all eligible names; 80 % keeps the top fifth of each industry. |
| **Min trade weight** | 0.00 % | Suppress trades smaller than this, to avoid dust. |

Each trade is tagged with a reason, which is the fastest way to understand the output:

| Reason | Meaning |
|---|---|
| `buy_to_objective` | Buying toward target weight |
| `sell_ethical_filter` | Excluded by the filter workbook |
| `sell_score_below_5` | Score under the 5.0 floor |
| `sell_below_score_quantile` | Below its industry quantile cutoff |
| `sell_trim_max_position` | Position exceeded Max position |
| `sell_rebalance_to_objective` | Ordinary reduction toward target |
| `rebalance_to_target` / `cash_rebalance` | Ordinary adjustment; cash adjustment |

### 11.2 `smart_beta`

Long-only, weights proportional to score. Every allowed name is score-tilted (scores are shifted so the
lowest gets ≈0 weight), group targets are applied, then weights are capped and renormalised.

| Parameter | Default | Meaning |
|---|---|---|
| **Top N** | 0 = all | Keep only the N highest-scoring names |
| **Max weight** | 10.00 % | Cap per security, applied after the score tilt |

No score floor, so any normalisation works.

### 11.3 `long_short`

Splits the score-ranked universe into buckets and goes long the top, short the bottom.

| Parameter | Default | Meaning |
|---|---|---|
| **Bucket count** | 10 | Number of equal buckets (10 = deciles) |
| **Long bucket count** | 1 | How many top buckets to hold long |
| **Short bucket count** | 1 | How many bottom buckets to short |
| **Weighting** | equal | `equal` = same weight each; `score` = proportional to score |
| **Gross exposure** | 100.00 % | Baseline exposure **per leg**. The default is +100 % long / −100 % short. |
| **Net exposure** | 0.00 % | Long/short tilt: long leg gets `gross + net/2`, short leg `gross − net/2`. 0 % = market-neutral. |

### Reading the result

Headline tiles: **Capital**, **Positions**, **Trades**, **Excluded**. Then five tabs:

| Tab | Contents |
|---|---|
| **Portfolio** | Target book: ticker, name, sector, industry, score, current weight, target weight, target amount |
| **Current** | Your uploaded holdings with quantity, price, amount, weight (rebalance mode) |
| **Trades** | Buy/sell per name with weight delta, quantity delta, price and **reason** |
| **Diagnostics** | **Excluded** names with reasons, **constraint diagnostics** (target vs actual vs difference per group), and **Notes** |
| **Raw JSON** | Everything, for auditing |

Check *Diagnostics → Excluded* whenever the portfolio is smaller than expected — the reason column
tells you which rule removed each name. `CASH_USD` appearing with a large target weight means the
strategy could not deploy the capital; the *Notes* say why.

---

## 12. Backtest

Two backtests, for two different questions.

### 12.1 Portfolio backtest — "how would *this* book have done?"

At the bottom of *Portfolio Construction*, after you build a portfolio. Set **start** and **end** dates,
a **methodology**, an optional **benchmark ticker** (e.g. `SPY`), then **Run portfolio backtest**.

It holds the exact weights you just built across the whole window. Useful for evaluating a book, but
note the weights come from one period's data applied over the entire window — for a realistic
simulation use §12.2.

### 12.2 Strategy Backtest — "how would the *method* have done?"

Rebuilds the portfolio from scratch in each window, chaining the results.

1. **Shared portfolio inputs** — profile, index, strategy, sector, industry, capital base. These apply
   to every window.
2. **External filters** and **Constraint targets** — same as §11, shared across all windows.
3. **Strategy parameters** — same as §11.
4. **Backtest setup** — methodology, benchmark, and the **schedule windows**.

**Schedule windows** are the heart of it. Each row pairs a **ranking period** with a **calendar
start/end date**: build the portfolio from that period's fundamentals, then hold it over those dates.
**Add window** for more, **Remove** to delete.

```
Period 2024/03/31  →  hold 2024-05-15 … 2024-08-14
Period 2024/06/30  →  hold 2024-08-15 … 2024-11-14
Period 2024/09/30  →  hold 2024-11-15 … 2025-02-14
```

Rules: **windows must not overlap**, and every period must exist. Leave a realistic gap between the
period end and the window start — fundamentals are not public on the period end date. The IC page uses
45 days; matching that is a sensible convention.

Capital compounds: each window starts at the previous window's ending value.

### 12.3 Methodology

- **Fixed weights** — weights are restored to target continuously (implicit daily rebalancing). Measures
  the signal, ignoring drift and trading costs.
- **Drifting weights** — buy at the start and let positions drift with prices. More realistic; winners
  grow their share.

Run both. A large gap between them tells you the strategy depends on frequent rebalancing.

### 12.4 Reading the results

Four tiles — **Total return**, **Annualized return**, **Volatility**, **Max drawdown** — then a chart of
portfolio vs benchmark value, a full series table (returns, cumulative, excess return, relative
cumulative) and **Component total returns** per holding with start price, end price and status.

The strategy backtest adds an **Intervals** table (one row per window with starting/ending value and
position count) and per-interval component expanders, so you can see which window drove the result.

Where a benchmark is supplied you also get benchmark total and annualised return, **tracking error**
(volatility of excess return), excess total return and relative total return.

**Always read the warnings.** They change how you should interpret the numbers:

| Warning | Meaning |
|---|---|
| `Missing price data for: …` | Those names were never priced and are absent from the result. |
| `Price history does not span the full window for: …` | Priced for only part of the window; **their prices are held flat outside the covered dates**, which biases the return toward zero for those positions. Upload the missing history (§13). |
| `Dropped positions without a starting price …` | No price on the first date, so excluded entirely. Their target weight is not invested. |
| `Benchmark '…' could not be priced and was omitted` | All benchmark-relative figures are absent. |

A component with `status = missing_start_price` was dropped; `cash` is the cash line and never priced.

---

## 13. Supply your own prices — *Price Data*

Yahoo Finance cannot price everything — delisted names, unusual listings, non-US venues. Upload your own
closes and they take priority over Yahoo in **backtests, IC analysis and Fetch latest prices**.

**Upload tab.** A Bloomberg wide-format Excel file:

| | Column A | Column B | Column C | … |
|---|---|---|---|---|
| Row 1 | *(title / metadata — ignored)* | | | |
| Row 2 | *(blank or a label)* | `AAPL US Equity` | `BRK/B US Equity` | … |
| Row 3+ | `2024-01-02` | `185.64` | `401.20` | … |

> ⚠️ **The layout above is correct; the app's own caption is off by one.** The Upload tab (and the error
> message you get on a failed upload) says *"Row 1 = ticker headers; Row 2+ = date + close prices"*. Those
> are internal zero-based row numbers, not Excel rows. **Your ticker headers must be on Excel row 2, with
> data from row 3**, leaving row 1 free for the export's title.
>
> This matters because following the caption literally fails *silently*: the parser reads your first data
> row as the header, so a file with tickers on row 1 imports securities named `100` and `200` — the prices
> from row 2 — with no error and nothing reported as skipped. If **Manage** shows tickers that look like
> numbers, this is what happened: delete them and re-upload with the title row in place.

Ticker headers are canonicalised automatically: `AAPL US Equity` → `AAPL`, `brk/b UN Equity` → `BRK/B`,
so you can paste a Bloomberg export unchanged. The response reports tickers imported, rows written, and
anything skipped — check `rows_skipped` and `tickers_skipped`, which flag unparseable dates, text in
price cells and headers that could not be read.

Prices must be **adjusted closes** if you want returns to be comparable with Yahoo's (which are
adjusted). Mixing raw and adjusted closes across securities biases returns.

**Manage tab.** Lists every cached ticker with its date range and row count. **Refresh** reloads;
select tickers and delete to remove them. Re-uploading overlapping dates overwrites the existing values.

**Coverage matters.** Uploading part of a window is handled honestly — cached dates are used, the rest
comes from Yahoo, and if neither source covers the whole window the backtest warns you about partial
coverage. But the fewer gaps you leave, the less you depend on Yahoo.

---

## 14. Worked example, start to finish

Building a quality-value factor and backtesting it over 2024.

**1. Load data.** *Periods → Create*, reader **Bloomberg**, upload `2024 Q1.xlsx`. Repeat for Q2, Q3,
Q4. Confirm four periods on **Home**. *(One period is enough to rank; several are needed for IC and a
multi-window backtest.)*

**2. Screen candidate metrics.** *Metric Diagnostics* for `Current ROA`, `Current ROIC`,
`Current Earnings Yield`, `Current Book to Price`. Note anything with red **% N/A** or **% Outliers**.
Suppose `Current ROIC` shows Std/MAD 6.2 — plan on winsorisation.

**3. Fix directions.** *Periods → View & Edit → Metric parameters*. All four are higher-is-better;
confirm that explicitly. Set N/A treatment to `Replace with low` for the two profitability metrics so
non-disclosure is penalised rather than rewarded.

**4. Test predictive power.** *Metric Selection (IC)*: select all four, horizon **3 months**, run.
Say `Current ROA` shows IR 0.55 and `Current ROIC` 0.48, but their correlation is **0.82** — redundant.
Keep ROA, drop ROIC. `Current Earnings Yield` shows IR 0.31, correlation with ROA 0.15 — keep it, it
diversifies.

**5. Build the profile.** *Scoring Profile Wizard*.
- Step 1: winsorisation on, quantile 0.05 / 0.95 (the outliers seen in step 2);
  terminal transform **Percentile rank** (we intend to use `legacy_rebalance` — see §8.3);
  aggregation **Linear**.
- Step 2: under **Scoring**, add subfactor `Quality` weight `0.6` containing `Current ROA` at `1.0`;
  add subfactor `Value` weight `0.4` containing `Current Earnings Yield` at `1.0`.
- Step 3: name it `QualityValue`, save.

**6. Rank.** *Ranking*: period `2024/12/31`, profile `QualityValue`, scope **All**. Check the top names
and that `Scoring` spans roughly 0–100 (percentile normalisation). Export if you want a record.

**7. Build a portfolio.** *Portfolio Construction*: same period and profile, strategy
**legacy_rebalance**, mode **new_portfolio**, capital base `1000000`. Max position 5 %, neutral position
3 %, score quantile cutoff 60 %. Build. Check *Diagnostics → Excluded* — with percentile scoring, most
exclusions should read `below_industry_quantile_0.60`, not `score_below_5`. If you see
`score_below_5` for nearly everything, your profile is on a z-score scale (§8.3).

**8. Backtest the method.** *Strategy Backtest*: profile `QualityValue`, strategy **legacy_rebalance**,
the same parameters, benchmark `SPY`, methodology **Drifting weights**. Four windows:

```
2024/03/31 → 2024-05-15 … 2024-08-14
2024/06/30 → 2024-08-15 … 2024-11-14
2024/09/30 → 2024-11-15 … 2025-02-14
2024/12/31 → 2025-02-15 … 2025-05-14
```

Run. Read the warnings first, then compare **Annualized return** and **Max drawdown** against the
benchmark, and use the **Intervals** table to see whether one window drove everything.

**9. Iterate.** Change one thing at a time — the cutoff, the weights, the normalisation, the
methodology — and re-run. Changing several at once tells you nothing about which mattered.

---

## 15. Troubleshooting

| What you see | Cause | Fix |
|---|---|---|
| `Cannot load data` / `Connection refused` on every page | The API is not running | Start `uvicorn api.main:app --reload`; check the sidebar URL |
| "No periods found" | No data loaded | *Periods → Create* |
| "No scoring profiles found" | None saved yet | *Scoring Profile Wizard* |
| Ranking: `Metric 'X' (column 'X_zscore') is not available for the selected period/scope` | The profile references a metric with no data in this period/scope | Remove it from the profile, or rank a period that has it |
| Ranking runs but every score is identical | A metric is all zeros — usually absent from the period and filled by `replace with zero` | Check §8.2; verify in *Periods → View & Edit* |
| Ranking output has an inverted look — worst companies on top | A metric's **Higher is better** is wrong | Fix it in §5 |
| Portfolio is entirely `CASH_USD`, everything excluded with `score_below_5` | Profile uses **z-score** normalisation with `legacy_rebalance`, whose floor is 5.0 | Switch the profile to **Percentile rank** (§8.3), or use `smart_beta` |
| Portfolio much smaller than expected | Score quantile cutoff too high, or a `0.0` constraint blocking groups | *Diagnostics → Excluded* names the rule |
| `Ranking output contains duplicate display columns` | Two profile nodes or metrics render to the same display name | Rename one node in *Scoring Profiles* |
| `current_holdings is required when construction_mode is 'rebalance_existing'` | No holdings workbook uploaded | Upload one, or switch to `new_portfolio` |
| `Existing portfolio value must be positive` | Holdings have no prices or market values | Click **Fetch latest prices**, or add a price column |
| Backtest: "Missing price data for: …" for most names | No price source reachable | Check internet access, or upload prices (§13) |
| Backtest hangs, then times out | Every Yahoo request is failing and being retried per ticker | Verify internet/certificates; upload prices for the universe (§13) |
| Backtest: `schedule rows must not overlap` | Two windows share dates | Adjust the dates |
| Backtest returns look implausibly flat | Partial price coverage — flat-filled outside cached dates | Read the warnings; upload the missing history |
| *Price Data → Manage* lists tickers that look like numbers (`100`, `185.64`) | Price file had ticker headers on Excel row 1 instead of row 2 | Delete them and re-upload with a title row above the headers (§13) |
| IC: "No predictive IC results" | Too few periods, or no price coverage | Load more periods; check prices |
| IC numbers look extreme with **Periods (T)** = 1 | One observation is not a time series | Load more periods |
| Metric Selection: `Select at least two metrics` | It is a multivariate analysis by design | Select two or more |
| Derived metric gives unexpected values | Operators apply strictly left to right | Split the formula into intermediate derived metrics (§6) |
| A page's results vanished | Browser refresh clears session state | Re-run; saved data is intact |

---

## 16. Parameter reference

### Scoring profile

| Setting | Options | Default |
|---|---|---|
| Winsorization | off · quantile (`lower`, `upper`) · semi (`k`) | on, quantile 0.01 / 0.99 |
| Normalization | `zscore` · `normalized_zscore` · `percentile` | `zscore` |
| Aggregation method | `linear` · `softplus` (per profile, overridable per node) | `linear` |
| Node weights | any number; children summing to 1.0 preserves scale | 0.5 per added input |
| Max nesting depth | 5 | — |

### Portfolio construction

| Parameter | Default | Strategies |
|---|---|---|
| Capital base | 1.0 | all (new_portfolio only) |
| Max position | 5.00 % | legacy_rebalance |
| Neutral position | 3.00 % | legacy_rebalance |
| Score quantile cutoff | 50.00 % | legacy_rebalance |
| Min trade weight | 0.00 % | legacy_rebalance |
| Top N | 0 (all) | smart_beta |
| Max weight | 10.00 % | smart_beta |
| Bucket count | 10 | long_short |
| Long / short bucket count | 1 / 1 | long_short |
| Weighting | equal | long_short |
| Gross exposure | 100.00 % per leg | long_short |
| Net exposure | 0.00 % | long_short |
| Constraint type | none | all |
| Hard score floor | 5.0 (not configurable) | legacy_rebalance |

### Backtest

| Parameter | Options | Default |
|---|---|---|
| Methodology | `fixed_weights` · `drifting_weights` | drifting_weights |
| Benchmark ticker | any ticker your price source knows | none |
| Frequency | daily · monthly | daily |
| Reported metrics | total return, annualized return, volatility, max drawdown, benchmark total & annualized return, tracking error, excess total return, relative total return | — |

### Required file layouts

| File | Requirement |
|---|---|
| Bloomberg fundamentals | Header on Excel row 4; `Ticker`, `Long Name`, `GICS Sector Name`, `GICS Industry Group Name`, `Market Cap (USD)`; period in cell A2 |
| BQL fundamentals | Sheets `Name`, `Classification`, `Current`, `Past`, `Estimated`, `Config`; period in `Config!B1`, universe in `Config!B2` |
| Reuters metrics | `Identifier` or `Identifier (RIC)` + `Earnings Quality Country Rank, Current`; period entered manually |
| Bloomberg prices | Row 2 = ticker headers from column B; row 3+ = date in column A, closes alongside |
| Holdings | Column A ticker, B quantity/market value, C price, D market value; `CASH_USD` for cash |
| Ethical filter | Sheet `STANDARD&POOR'S500`, header on row 4, columns `Ticker` and `Ethical Evaluation`; `No` excludes |
