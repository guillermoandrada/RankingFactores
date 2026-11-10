"""Nested walk-forward for the general Value profile (protocol 2026-11-06).

Each test quarter t uses only quarters < t to (1) pick metrics inside each pillar by pooled IC (top half),
(2) choose pillar weights on a 10-35% grid by mean excess over FF B/M, shrunk to equal pillars unless the
winner beats them by > 0.5 pp/yr, (3) exclude industries whose fit-window excess over FF has t < -1.5.
Banks and Insurance (and any industry where a pillar has > 40% raw NA) move the Caja/Ventas weight to a
book pillar (P/B, Est P/B, P/TBV)."""
import itertools

import numpy as np
import pandas as pd

from vlib import PERIODS, MIN_FIT, long_short, rank_ic, build_cells, PANEL_DIR

PILLARS = ["Beneficios", "Caja", "Valor contable", "Ventas", "Dividendo", "Change/Trend"]
BASE_SETS = {
    "Beneficios": ["Current P/E Ratio", "Estimated P/E Ratio"],
    "Caja": ["Current EV / EBITDA", "Estimated EV / EBITDA", "Current FCF Yield", "Estimated FCF Yield"],
    "Valor contable": ["Current P/B Ratio", "Estimated P/B Ratio"],
    "Ventas": ["Current P/S Ratio", "Estimated P/S Ratio"],
    "Dividendo": ["Current Dividend Yield"],
    "Change/Trend": ["P/E Change", "P/E Trend", "P/B Change", "P/B Trend", "P/S Change", "P/S Trend",
                     "EV/EBITDA Change", "EV/EBITDA Trend", "FCF Yield Change", "FCF Yield Trend"],
}
GP_METRICS = ["Current P/GP", "Estimated P/GP"]
BOOK_SUB = ["Current P/B Ratio", "Estimated P/B Ratio", "Current Price to Tangible Book"]
THEORY_SUB = {"Banks": {"Caja", "Ventas"}, "Insurance": {"Caja", "Ventas"}}
DATA_SUB_PILLARS = ["Beneficios", "Caja", "Ventas"]
SECTOR_EXTRA = ["Current EV / EBITDAR", "OCF Yield", "Current ROE", "P/TBV Change"] if PANEL_DIR != "rpanel" else []
METRICS = sorted({m for ms in BASE_SETS.values() for m in ms} | set(GP_METRICS) | set(BOOK_SUB) | set(SECTOR_EXTRA))
COL = {m: i for i, m in enumerate(METRICS)}
EQ = np.full(6, 1 / 6)
LEVELS_W = np.round(np.arange(0.10, 0.351, 0.05), 2)
GRID = np.array([c for c in itertools.product(*[LEVELS_W] * 6) if abs(sum(c) - 1) < 1e-9])
# Sensitivity (protocol deviation): Dividendo and Change/Trend may be 0%
GRID0 = np.array([c for c in itertools.product(*([LEVELS_W] * 4 + [np.r_[0.0, LEVELS_W]] * 2))
                  if abs(sum(c) - 1) < 1e-9])

CELLS, INDUSTRIES = build_cells(METRICS, min_firms=8)
IC = np.vstack([rank_ic(c["z"], c["fwd"]) for c in CELLS])
for c in CELLS:
    c["live"] = c["z"].std(axis=0) > 0
CP = np.array([c["p"] for c in CELLS])
CI = np.array([c["ind"] for c in CELLS])
FFR = np.array([c["ff"] for c in CELLS])


def mean_live(c, metrics):
    idx = [COL[m] for m in metrics if c["live"][COL[m]]]
    return c["z"][:, idx].mean(axis=1) if idx else np.zeros(len(c["fwd"]))


def pick_metrics(fit_mask, sets):
    pooled = np.nanmean(IC[fit_mask], axis=0)
    return {p: sorted(ms, key=lambda m: -pooled[COL[m]])[:int(np.ceil(len(ms) / 2))] for p, ms in sets.items()}


def substitutions(fit_mask, picks, use_sub, theory=None):
    """Pillars replaced by the book pillar, per industry."""
    out = {}
    for ind in INDUSTRIES:
        subs = set((THEORY_SUB if theory is None else theory).get(ind, set())) if use_sub else set()
        if use_sub:
            rows = [c for c, f in zip(CELLS, fit_mask) if f and c["ind"] == ind]
            for p in DATA_SUB_PILLARS:
                na = np.concatenate([c["na"][:, [COL[m] for m in picks[p]]].all(axis=1) for c in rows])
                if na.mean() > 0.40:
                    subs.add(p)
        out[ind] = subs
    return out


def pillar_matrix(c, picks, subs):
    book = mean_live(c, BOOK_SUB) if subs else None
    cols = []
    for p in PILLARS:
        if p in subs or (subs and p == "Valor contable"):
            cols.append(book)
        else:
            cols.append(mean_live(c, picks[p]))
    return np.column_stack(cols)


def ann_excess(ret, ff, periods, objective="mean"):
    """Mean over periods of the cross-industry mean excess, x4 (or its IR per quarter)."""
    d = pd.DataFrame({"p": periods, "x": ret - ff}).dropna().groupby("p").x.mean()
    return d.mean() / d.std() if objective == "ir" else d.mean() * 4


def run(sets, use_sub=True, select="top5", exclude=True, label="", objective="mean", theory=None, grid=None, eq=None):
    grid = GRID if grid is None else grid
    eq_w = EQ if eq is None else np.asarray(eq, float)
    rows, weights, picks_log, excl_log, subs_log = [], {}, {}, {}, {}
    for k in range(MIN_FIT, len(PERIODS)):
        test, fit_periods = PERIODS[k], PERIODS[:k]
        fit = np.isin(CP, fit_periods)
        picks = pick_metrics(fit, sets)
        subs = substitutions(fit, picks, use_sub, theory)
        idx_fit = np.where(fit)[0]
        P = {i: pillar_matrix(CELLS[i], picks, subs[CELLS[i]["ind"]]) for i in np.where(fit | (CP == test))[0]}
        if select == "eq":
            w = eq_w
        else:
            G = np.vstack([grid, eq_w])
            R = np.vstack([long_short(P[i] @ G.T, CELLS[i]["fwd"]) for i in idx_fit])
            by_p = pd.DataFrame(R - FFR[idx_fit, None]).groupby(CP[idx_fit]).mean()
            ex = (by_p.mean() / by_p.std()).to_numpy() if objective == "ir" else by_p.mean().to_numpy() * 4
            ex_grid, ex_eq = ex[:-1], ex[-1]
            if select == "top5":
                top = np.argsort(-ex_grid)[:max(1, int(len(grid) * 0.05))]
                w_win = grid[top].mean(axis=0)
                r_win = np.array([long_short((P[i] @ w_win)[:, None], CELLS[i]["fwd"])[0] for i in idx_fit])
                ex_win = ann_excess(r_win, FFR[idx_fit], CP[idx_fit], objective)
            else:
                w_win, ex_win = grid[int(np.argmax(ex_grid))], ex_grid.max()
            w = w_win if ex_win - ex_eq > (0.005 if objective == "mean" else 0.05) else eq_w
        excluded = set()
        if exclude:
            r_fit = np.array([long_short((P[i] @ w)[:, None], CELLS[i]["fwd"])[0] for i in idx_fit])
            d = pd.DataFrame({"ind": CI[idx_fit], "x": r_fit - FFR[idx_fit]}).dropna()
            for ind, g in d.groupby("ind"):
                if len(g) > 2 and g.x.std() > 0 and g.x.mean() / g.x.std() * np.sqrt(len(g)) < -1.5:
                    excluded.add(ind)
        weights[test], picks_log[test], excl_log[test] = w, picks, excluded
        subs_log[test] = {i: s for i, s in subs.items() if s}
        for i in np.where(CP == test)[0]:
            c, s = CELLS[i], P[i] @ w
            rows.append(dict(period=test, industry=c["ind"], ret=long_short(s[:, None], c["fwd"])[0],
                             ff=c["ff"], ic=rank_ic(s[:, None], c["fwd"])[0], ff_ic=c["ff_ic"],
                             included=c["ind"] not in excluded))
    return dict(label=label, df=pd.DataFrame(rows), weights=pd.DataFrame(weights, index=PILLARS).T,
                picks=picks_log, excluded=excl_log, subs=subs_log)


def final_fit(sets, use_sub=True):
    """Deployment weights: same rule fitted on all 36 quarters (for annex / ranking_profiles only)."""
    fit = np.ones(len(CELLS), bool)
    picks = pick_metrics(fit, sets)
    subs = substitutions(fit, picks, use_sub)
    P = {i: pillar_matrix(CELLS[i], picks, subs[CELLS[i]["ind"]]) for i in range(len(CELLS))}
    G = np.vstack([GRID, EQ])
    R = np.vstack([long_short(P[i] @ G.T, CELLS[i]["fwd"]) for i in range(len(CELLS))])
    ex = pd.DataFrame(R - FFR[:, None]).groupby(CP).mean().mean().to_numpy() * 4
    top = np.argsort(-ex[:-1])[:int(len(GRID) * 0.05)]
    w_win = GRID[top].mean(axis=0)
    r_win = np.array([long_short((P[i] @ w_win)[:, None], CELLS[i]["fwd"])[0] for i in range(len(CELLS))])
    w = w_win if ann_excess(r_win, FFR, CP) - ex[-1] > 0.005 else EQ
    rows = []
    for i, c in enumerate(CELLS):
        s = P[i] @ w
        rows.append(dict(period=c["p"], industry=c["ind"], ret=long_short(s[:, None], c["fwd"])[0], ff=c["ff"],
                         ic=rank_ic(s[:, None], c["fwd"])[0], ff_ic=c["ff_ic"], included=True))
    return dict(w=w, picks=picks, subs={i: s for i, s in subs.items() if s}, df=pd.DataFrame(rows))
