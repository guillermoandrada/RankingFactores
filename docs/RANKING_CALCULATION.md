# RankingFactores — Calculating a Ranking

Agent-facing procedure for producing a ranking from **derived metrics** and **ranking profiles**,
written against the code as it exists. It tells you which objects to create, in which order, what the
engine does with them, and where each rule is enforced.

Companion documents:
[docs/ARCHITECTURE.md](ARCHITECTURE.md) (layer map and dependency rules),
[docs/USER_GUIDE.md](USER_GUIDE.md) (the same workflow through the Streamlit UI),
[docs/PRICING.md](PRICING.md) (prices — not needed for ranking).
The rules in [CLAUDE.md](../CLAUDE.md) override anything here.

---

## 1. What a ranking is

A ranking is a single `pandas.DataFrame`, one row per security, sorted descending by the column
`Scoring`. Producing it requires exactly three inputs:

| Input | Where it lives | Owner |
|---|---|---|
| **Base metrics** — raw values per security, per period | SQLite: `metrics`, `fundamental_values` | [modules/infrastructure/db/repository.py](../modules/infrastructure/db/repository.py) |
| **Derived metrics** — formulas over base metrics | `modules/config/derived_metrics.json` | [DerivedMetricStore](../modules/config/derived_metrics.py) |
| **Ranking profile** — the factor tree plus normalisation settings | `modules/config/ranking_profiles.json` | [RankingProfileStore](../modules/config/ranking_profiles.py) |

Derived metrics and profiles are **JSON, not SQL**. They are period-independent: one profile is
applied to whatever period you rank, which is what makes multi-period comparison meaningful.

Entry point for the whole calculation:
[`compute_ranking()`](../api/services/ranking_service.py#L45) in `api/services/ranking_service.py`.

---

## 2. Prerequisites — verify before building anything

A metric that is *defined* in the database is not the same as a metric that *has data* in the period
you intend to rank. This distinction is the single largest source of silently wrong rankings.

```bash
# 1. Which periods exist?
curl -s http://127.0.0.1:8000/reference/periods

# 2. Which metrics exist at all (DB metrics + derived formulas)?
curl -s http://127.0.0.1:8000/reference/db/metrics/available
```

`GET /reference/db/metrics/available` is the authoritative name list. Every string you put in a
formula or a profile weight must appear there, spelled identically — matching is exact, including
case and spaces (`"Current Book to Price"`, not `"current book to price"`).

To confirm a metric actually has values in your target period, preview it (§3.4) or check
*Periods → View & Edit* in the UI, which lists only metrics present in that period.

---

## 3. Derived metrics

### 3.1 Semantics

A derived metric is a flat, left-to-right chain over other metric names. There is no operator
precedence and no parentheses:

```
metric_names = [A, B, C]
operations   = ["-", "/"]
value        = ((A - B) / C)
```

Implemented in [`_compute_derived()`](../modules/domain/analytics/metric_loader.py#L56). Notes that
follow from the implementation:

- Division replaces `0` in the denominator with `NA` — no `ZeroDivisionError`, the row simply
  becomes missing.
- Every input is coerced with `pd.to_numeric(..., errors="coerce")`; non-numeric values become `NA`.
- Inputs may themselves be derived metrics. Dependencies are resolved depth-first in
  [`_resolve_dependencies()`](../modules/domain/analytics/metric_loader.py#L77) and computed in
  topological order, so a formula can build on another formula.
- Cycles raise `ValueError("Circular dependency in derived metric ...")`.

### 3.2 Stored shape

```json
{
  "formulas": {
    "Past Book to Price": {
      "metric_names": ["Current Book to Price", "5Y Average Book to Price"],
      "operations": ["-"],
      "higher_is_better": true,
      "na_handling": "replace_with_zero"
    }
  }
}
```

| Field | Rule |
|---|---|
| `metric_names` | ≥ 2 entries, ordered. Each must be a base metric or another derived metric. |
| `operations` | Exactly `len(metric_names) - 1` entries, each one of `+ - * /`. |
| `higher_is_better` | `true` / `false`. `null` is treated as `true` at ranking time. |
| `na_handling` | `replace_with_zero`, `replace_with_high`, `replace_with_low`, `eliminate`, or `null`. |

Validation lives in two places and both run:
[`DerivedMetricStore.upsert_formula()`](../modules/config/derived_metrics.py) checks arity and
operators; [`MetricsService._validate_formula()`](../api/services/metrics_service.py) additionally
walks the dependency graph via
[`validate_formula_graph()`](../modules/domain/analytics/metric_loader.py#L100), so an unresolvable
formula cannot be saved and then fail weeks later inside a ranking.

### 3.3 `na_handling` — what each option does

Applied in [`_apply_na_handling()`](../modules/domain/analytics/metric_loader.py#L22), **after** all
derived metrics are computed and **before** any transform:

| Option | Effect |
|---|---|
| `replace_with_zero` | Fill `NA` with `0.0`. |
| `replace_with_high` | Fill `NA` with the column maximum (falls back to `0.0` if the column is empty). |
| `replace_with_low` | Fill `NA` with the column minimum (same fallback). |
| `eliminate` | **Drop the row entirely** from the universe. |
| `null` / unset | Leave `NA`; the terminal transform then fills it (z-score fills with the mean, percentile with the median). |

`eliminate` is applied across all eliminate-flagged columns in one `dropna`, so it shrinks the
universe for every metric in the ranking, not just its own. Use it deliberately.

For base metrics this value comes from the `"n/a treatment"` column of the `metrics` table
([`_get_base_na_handling()`](../modules/domain/analytics/metric_loader.py#L245)); for derived metrics
it comes from the formula.

### 3.4 API

| Action | Call |
|---|---|
| List | `GET /metrics` — add `?metric_name=` for one |
| **Preview without saving** | `POST /metrics/preview` |
| Create | `POST /metrics` → `201`, `409` if the name is taken, `422` if the formula is invalid |
| Update | `PUT /metrics/{metric_name}` |
| Delete | `DELETE /metrics/{metric_name}` → `204` |

**Always preview before creating.** It computes the candidate on one period without writing
anything, and returns count/coverage, the distribution (`min`, `p05`, `median`, `p95`, `max`, `mean`,
`std`) and the 5 highest and 5 lowest securities by ticker. A unit mismatch or an inverted sign shows
up there; it does not show up in a saved formula.

```bash
curl -s -X POST http://127.0.0.1:8000/metrics/preview \
  -H 'Content-Type: application/json' \
  -d '{
        "period": "2024/12/31",
        "metric_name": "Book to Price Momentum",
        "metric_names": ["Estimated Book to Price", "Current Book to Price"],
        "operations": ["-"]
      }'
```

Read `missing_pct` from the response. A formula whose inputs rarely overlap produces a mostly-empty
column; with `replace_with_zero` that column then contributes noise-free zeros and quietly dilutes
the factor it sits in.

Then create it:

```bash
curl -s -X POST http://127.0.0.1:8000/metrics \
  -H 'Content-Type: application/json' \
  -d '{
        "new_metric_name": "Book to Price Momentum",
        "metric_names": ["Estimated Book to Price", "Current Book to Price"],
        "operations": ["-"],
        "higher_is_better": true,
        "na_handling": "replace_with_zero"
      }'
```

Derived names must not collide with a DB metric name — `MetricsService` raises
`DuplicateMetricError` → `409`.

---

## 4. Ranking profiles

### 4.1 Stored shape

A profile is a tree of **nodes** plus profile-wide settings. A node's `inputs` map child names to
weights; a child is either another node or a metric name (base or derived). Leaf-ness is inferred,
never declared: **any input name that is not itself a node key is treated as a metric.**

```json
{
  "profiles": {
    "quality_value": {
      "nodes": {
        "Value":   { "inputs": { "Current Earnings Yield": 0.5, "Current Book to Price": 0.5 }, "method": "linear" },
        "Quality": { "inputs": { "Current ROA": 0.5, "Current ROIC": 0.5 },                     "method": "linear" },
        "Scoring": { "inputs": { "Value": 0.4, "Quality": 0.6 },                                "method": "linear" }
      },
      "normalization": "zscore",
      "winsorization": { "lower": 0.01, "upper": 0.99 },
      "winsor_mode": "quantile",
      "method": "linear"
    }
  }
}
```

| Field | Values | Default applied by [`normalize_profile()`](../modules/config/ranking_profiles.py#L48) |
|---|---|---|
| `nodes` | `{node_name: {inputs: {name: weight}, method?}}` | required, non-empty |
| `normalization` | `zscore` \| `normalized_zscore` \| `percentile` | `"zscore"` |
| `winsorization` | `false` \| `{lower, upper}` \| `{k}` | `false` |
| `winsor_mode` | `quantile` \| `semi` | `"quantile"` |
| `method` | `linear` \| `softplus` | `"linear"` — also inherited by every node lacking its own `method` |

A legacy `overrides` key is stripped on load. Sector-specific behaviour is expressed by *choosing a
different profile per scope* at ranking time (§5.2), not inside the profile.

### 4.2 Weight convention

Weights are **multipliers, not shares** — nothing normalises them. Make each node's enabled children
sum to `1.0` so the final score stays on the same scale as the individual metric scores. Weighting
two children `0.4` and `0.4` shrinks every score by 20 %. The UI validator
([`validate_nodes()`](../streamlit_app/components/profile_editor/validators.py)) flags any node whose
weights deviate from `1.0` by more than `0.001`; the API does **not** enforce this, so a
programmatically written profile can be silently off-scale.

### 4.3 Choosing the normalisation

| Option | Output scale | Behaviour |
|---|---|---|
| `zscore` | ≈ −3 … +3, mean 0 | Preserves relative distances; sensitive to skew. `NA` → column mean. Constant column → all `0.0`. |
| `normalized_zscore` | 0 … 10 | Z-score rescaled so min→0, max→10. Anchored on the extremes, so a skewed metric squashes most names low. Constant column → all `5.0`. |
| `percentile` | 0 … 100 | Rank only. Immune to outliers, discards magnitude. `NA` → column median. |

This choice is not cosmetic: the `legacy_rebalance` portfolio strategy applies a **fixed 5.0 score
floor**, so a plain `zscore` profile feeding it yields an all-cash portfolio. See
[USER_GUIDE §8.3](USER_GUIDE.md) for the measured comparison. `smart_beta` and `long_short` use only
the ordering and are indifferent.

### 4.4 Choosing the aggregation method

Both combiners live in [modules/domain/analytics/combiners.py](../modules/domain/analytics/combiners.py):

- **`linear`** — `Σ sign · weight · z`, with `NA` filled as `0.0`. Additive and predictable; one
  strong input can carry a weak one.
- **`softplus`** — `exp( Σ sign · weight · log(softplus(z)) )`, with `z` clipped to `[-50, 50]`.
  Geometric in character: being weak on any one input drags the whole score down, so it rewards
  all-round names over specialists.

`sign` is `+1` when `higher_is_better`, `-1` otherwise, taken from the direction map. Set per node;
a node without `method` inherits the profile default.

### 4.5 API

| Action | Call |
|---|---|
| List | `GET /scoring-profiles` — add `?profile_name=` for one |
| Create | `POST /scoring-profiles` → `201` (name in body) |
| Upsert | `PUT /scoring-profiles/{profile_name}` |
| Delete | `DELETE /scoring-profiles/{profile_name}` → `204` |

```bash
curl -s -X PUT http://127.0.0.1:8000/scoring-profiles/quality_value \
  -H 'Content-Type: application/json' \
  -d '{"profile": { "nodes": { ... }, "normalization": "percentile", "winsorization": {"lower":0.01,"upper":0.99}, "winsor_mode": "quantile", "method": "linear" }}'
```

Both write endpoints take the profile under a `profile` key ([api/schemas/profiles.py](../api/schemas/profiles.py))
and return `{success, profile_name, profile}`.

---

## 5. Running the ranking

### 5.1 Single scope

```bash
curl -s -X POST "http://127.0.0.1:8000/scorings/2024%2F12%2F31" \
  -H 'Content-Type: application/json' \
  -d '{"scoring_profile": "quality_value", "index": "", "sector": "", "industry": "", "export": false}'
```

The period is a path parameter and must be URL-encoded (periods contain `/`). It is validated
against `list_periods()` → `404` if unknown; a `ValueError` from the engine surfaces as `400`.

Filters are ANDed and all optional (empty string = no filter): `index` restricts to that index's
members *for that period*, `sector` and `industry` restrict by classification. Set `export: true` to
receive an XLSX attachment instead of JSON.

Response:

```json
{
  "period": "2024/12/31",
  "industry": null,
  "sector": null,
  "scoring_profile": "quality_value",
  "count": 503,
  "warnings": ["Metric 'X' is completely missing for this scope; its score was set to 0 for all securities."],
  "ranking": [ { "ticker": "...", "name": "...", "... Score": 1.23, "Scoring": 0.87 } ]
}
```

**Always read `warnings`.** They are the only signal that a metric contributed a column of zeros.

### 5.2 Multiple scopes, optionally with per-scope profiles

`POST /scorings/{period}/batch` runs scopes in parallel (`ThreadPoolExecutor`, ≤ 8 workers) and
returns `{"results": {scope_key: {...}}}`, where `scope_key` is the sector, else the industry, else
`"All"`. A scope that fails yields `{"error": "...", "ranking": []}` instead of failing the batch.

```json
{
  "scoring_profile": "quality_value",
  "index": "S&P 500",
  "scopes": [
    { "sector": "Financials", "scoring_profile": "bank_specific" },
    { "sector": "Information Technology" },
    { "sector": "Health Care" }
  ]
}
```

`scopes[].scoring_profile` overrides the batch-level default for that scope only. This is the
supported way to score a sector differently — do not reintroduce in-profile overrides.

Scoping matters statistically: every transform and every combination is computed **within the
returned universe**. A z-score under sector scope is relative to that sector's peers; under `All` it
is relative to the whole universe.

### 5.3 In-process (tests, scripts)

```python
from api.services.ranking_service import compute_ranking

df = compute_ranking(
    quarter="2024/12/31",
    scoring_profile="quality_value",
    index="",
    sector="",
    industry="",
)
warnings = df.attrs.get("warnings", [])
```

`compute_ranking_for_profile()` is a thin alias with the same signature. Note the parameter is named
`quarter`, not `period`.

---

## 6. What the engine actually does

Follow this order when debugging a number that looks wrong.

**1 — Resolve the profile.**
[`RankingProfileResolver.resolve()`](../modules/config/ranking_profiles.py#L249) applies defaults,
then [`_convert_profile_to_legacy()`](../modules/config/ranking_profiles.py#L142) flattens the node
tree into the execution shape:

- `factors` — nodes whose inputs are *all* metrics (leaves), in topological order.
- `layers` — nodes with at least one node input, in topological order. If the tree has none, a
  synthetic `{"name": "score", "weights": {<last factor>: 1.0}}` layer is appended.
- `metric_transforms` — the chain built by
  [`_profile_to_transform_chain()`](../modules/config/ranking_profiles.py#L15): the winsor step (if
  any) followed by the terminal normalisation.

**2 — Collect metric names.**
[`get_metric_names_from_profile()`](../api/services/ranking_service.py#L11) takes the union of all
`factors[].weights` keys — i.e. **only leaf metrics**. Metrics referenced directly by a non-leaf node
are combined but are *not* added to the load list, so mixing metrics and nodes in one node's inputs
is an asymmetric case to avoid.

**3 — Load and build the metric matrix.**
[`fetch_metric_matrix()`](../modules/domain/analytics/metric_loader.py#L115):
resolve dependencies → fetch base metrics from `fundamental_values` for the period and filters →
pivot long-to-wide on `(security_id, ticker, long_name)` → compute derived metrics in dependency
order → build `direction_map` from `metrics.higher_is_better` and the formulas → apply
`na_handling` → record `attrs["missing_metrics"]`.

An empty result set raises `ValueError("No data found for the given period, metrics, and filters.")`
→ `400`.

**4 — Transform each metric.**
[`ZScoreCalculator.compute()`](../modules/domain/analytics/zscore.py#L41) applies the chain per
column, producing `<metric>_zscore`. When winsorization is on it also keeps `<metric>_winsor`; when
it is off it keeps the raw value column for display. A metric that is entirely missing for the scope
is set to `0.0` and appended to `warnings` rather than raising.

**5 — Combine, bottom-up.**
[`FactorScoringService._run_multistage()`](../modules/domain/analytics/factors.py#L62): each factor
is combined from its metric `_zscore` columns and written back as `<factor>_zscore`, making it
available to layers. Each layer is then combined in order; intermediate layers are written back as
`<layer>_zscore`, and the **last layer always outputs the column `scoring`**.

Composed nodes are treated as higher-is-better (direction `+1`); the `direction_map` sign applies
only to raw metrics.

**6 — Present.**
[`compute_ranking()`](../api/services/ranking_service.py#L45) sorts by `scoring` descending, drops
`security_id`, renames `long_name` → `name`, moves `ticker`/`name` first, then
[`_apply_display_labels()`](../api/services/ranking_service.py#L23) rewrites the `_zscore` suffix to
` Score` and capitalises the first letter — so `scoring` becomes **`Scoring`**.

If two columns collapse to the same display label it raises
`"Ranking output contains duplicate display columns"`. **Therefore a node must not be named after a
metric it contains, and two nodes must not differ only by leading case.**

---

## 7. Failure modes

| Symptom | Cause | Where |
|---|---|---|
| `400 No data found for the given period, metrics, and filters.` | Period/filter combination is empty, or no base metric has rows | [metric_loader.py#L115](../modules/domain/analytics/metric_loader.py#L115) |
| `Metric 'X' not found in database.` | Name not in `metrics` and not a derived formula | [`_get_metric_ids()`](../modules/domain/analytics/metric_loader.py#L276) |
| `Circular dependency in derived metric 'X'.` | Formula graph has a cycle | [`_resolve_dependencies()`](../modules/domain/analytics/metric_loader.py#L77) |
| `No base metrics to fetch. All requested metrics are derived.` | Every leaf resolved to formulas only | [metric_loader.py#L115](../modules/domain/analytics/metric_loader.py#L115) |
| `Profile must have at least one factor (leaf node).` | Every node references another node | [`_convert_profile_to_legacy()`](../modules/config/ranking_profiles.py#L142) |
| `Factor 'X' has empty weights.` / `Layer 'X' has empty weights.` | A node has `inputs: {}` | [factors.py#L62](../modules/domain/analytics/factors.py#L62) |
| `Metric 'X' (column 'X_zscore') is not available…` | A node input was neither loaded nor produced — usually a metric referenced by a non-leaf node (see §6 step 2) | [combiners.py](../modules/domain/analytics/combiners.py) |
| `Ranking output contains duplicate display columns` | Node name collides with a metric name after labelling | [ranking_service.py#L23](../api/services/ranking_service.py#L23) |
| Every security scores identically | A metric is all zeros — check `warnings` | [zscore.py#L41](../modules/domain/analytics/zscore.py#L41) |
| Portfolio comes back all cash | `zscore` profile fed to `legacy_rebalance`'s 5.0 floor | §4.3 |

Caches worth knowing: `_DB_METRIC_CACHE` and `_BASE_NA_CACHE` in `metric_loader.py` are keyed by
`id(engine)` and live for the process, so metric metadata edited outside the API is not picked up
until restart. `RankingProfileStore` caches on file mtime and *does* pick up external edits.

---

## 8. Checklist

1. `GET /reference/periods` — confirm the target period exists, exactly as spelled.
2. `GET /reference/db/metrics/available` — copy metric names verbatim from here.
3. For each new derived metric: `POST /metrics/preview`, read `missing_pct` and the extremes, then
   `POST /metrics`.
4. Set `higher_is_better` and `na_handling` on every metric you use — base metrics via
   `PUT /db-metrics/{metric_id}`, derived via the formula. An unset direction defaults to
   *higher is better*, which is wrong for most valuation ratios.
5. Write the profile: node names distinct from metric names, each node's weights summing to `1.0`,
   leaf nodes containing metrics only.
6. Pick `normalization` for the downstream consumer, not for the ranking table (§4.3).
7. `PUT /scoring-profiles/{name}`.
8. `POST /scorings/{period}` — or `/batch` for per-sector scopes and per-scope profiles.
9. Read `warnings`, check `count` against the expected universe size, and sanity-check the top names
   and the score spread before anything downstream consumes the ranking.
