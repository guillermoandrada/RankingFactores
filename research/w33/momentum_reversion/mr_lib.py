"""Walk-forward engine of the common improvement protocol (mejora/protocolo.md) for Momentum and Reversion.

YJ (lambda per metric/period, S&P body 0.5-99.5%) -> winsor 5/95 per industry -> Z(0,1), NA neutral.
Within-industry long-short halves, BBG 3m TR (missing = 0). Nested walk-forward: metric picks by pooled IC,
pillar weights by excess over the benchmark with shrinkage to equal pillars, pre-registered industry rule.
"""
import os
import pickle
import itertools

import numpy as np
import pandas as pd
from scipy.stats import yeojohnson, yeojohnson_normmax

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_FIT, MIN_FIRMS, NA_MAX, T_EXCLUDE, SHRINK_PP = 8, 8, 0.40, -1.5, 0.005
STEP, LO, HI = 0.05, 0.10, 0.35

_data = pickle.load(open(os.path.join(HERE, "mr_panel.pkl"), "rb"))
PANEL = _data["panel"]
PERIODS = sorted(PANEL)
TEST = PERIODS[MIN_FIT:]
METRICS = [c for c in PANEL[PERIODS[0]]["df"].columns if c not in ("ticker_c", "industry", "fwd3", "fwd12")]
DIRECTION = {}
for p in PERIODS:
    DIRECTION.update(PANEL[p]["direction"])
SIGN = np.array([1.0 if DIRECTION.get(m, True) else -1.0 for m in METRICS])
COL = {m: i for i, m in enumerate(METRICS)}
_firms = pd.DataFrame({p: PANEL[p]["df"].industry.value_counts() for p in PERIODS}).fillna(0)
INDUSTRIES = sorted(_firms.index[_firms.mean(axis=1) >= MIN_FIRMS])
ALL_INDUSTRIES = sorted(_firms.index)


def _lambdas(p):
    out = {}
    df = PANEL[p]["df"]
    for m in METRICS:
        x = df[m].dropna().to_numpy(float)
        if len(x) >= 30 and np.std(x) > 0:
            lo, hi = np.quantile(x, [0.005, 0.995])
            out[m] = float(np.clip(yeojohnson_normmax(np.clip(x, lo, hi)), -3, 3))
    return out


def _transform(raw: pd.DataFrame, lam: dict) -> np.ndarray:
    df = raw.copy()
    for m in df.columns:
        s = df[m]
        if m in lam and s.notna().any():
            df.loc[s.notna(), m] = yeojohnson(s[s.notna()].to_numpy(float), lmbda=lam[m])
    w = df.clip(lower=df.quantile(.05), upper=df.quantile(.95), axis=1)
    mu, sd = w.mean(), w.std(ddof=0)
    return ((w.fillna(mu) - mu) / sd.replace(0, np.nan)).fillna(0.0).to_numpy() * SIGN


def _rank_ic(score, fwd):
    ok = ~np.isnan(fwd)
    if ok.sum() < 5 or np.std(score[ok]) == 0:
        return np.nan
    return float(pd.Series(score[ok]).rank().corr(pd.Series(fwd[ok]).rank()))


def long_short(score: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """score n x k -> k long-short returns (top half minus bottom half, missing TR = 0)."""
    size = int(np.ceil(score.shape[0] / 2))
    order = np.argsort(-score, axis=0, kind="stable")
    r = np.nan_to_num(fwd)[order]
    return r[:size].mean(axis=0) - r[-size:].mean(axis=0)


CELLS = []
for p in PERIODS:
    lam = _lambdas(p)
    df = PANEL[p]["df"]
    for ind in ALL_INDUSTRIES:
        sub = df[df.industry == ind]
        if len(sub) < 2:
            continue
        z = _transform(sub[METRICS].astype(float), lam)
        fwd = sub.fwd3.to_numpy(float)
        CELLS.append(dict(p=p, ind=ind, z=z, fwd=fwd, na=PANEL[p]["raw_na"].loc[sub.index].reindex(
            columns=METRICS).fillna(False).to_numpy(float).mean(axis=0),
            ic=np.array([_rank_ic(z[:, j], fwd) for j in range(len(METRICS))])))
IDX = pd.MultiIndex.from_tuples([(c["p"], c["ind"]) for c in CELLS], names=["period", "industry"])
IC = pd.DataFrame([c["ic"] for c in CELLS], index=IDX, columns=METRICS)
IN_UNIVERSE = np.array([c["ind"] in INDUSTRIES for c in CELLS])


def metric_series(metric) -> pd.Series:
    j = COL[metric]
    return pd.Series([long_short(c["z"][:, [j]], c["fwd"])[0] for c in CELLS], index=IDX)


def weights_series(weights: dict) -> tuple[pd.Series, pd.Series]:
    w = np.zeros(len(METRICS))
    for m, v in weights.items():
        w[COL[m]] = v
    s = [c["z"] @ w for c in CELLS]
    return (pd.Series([long_short(x[:, None], c["fwd"])[0] for x, c in zip(s, CELLS)], index=IDX),
            pd.Series([_rank_ic(x, c["fwd"]) for x, c in zip(s, CELLS)], index=IDX))


def make_grid(n_pillars):
    vals = np.round(np.arange(LO, HI + 1e-9, STEP), 2)
    return np.array([g for g in itertools.product(vals, repeat=n_pillars) if abs(sum(g) - 1) < 1e-9])


class Profile:
    """pillars: {name: [candidate metrics]}; benchmark: single metric.

    mode "grid": pillar weights from the bounded grid with shrinkage to equal pillars; "equal": equal pillars.
    pillar_t: if set, a pillar enters only when its best metric has pooled IC t >= pillar_t in the fit window.
    """

    def __init__(self, name, pillars: dict, benchmark: str, mode="grid", pillar_t=None):
        self.name, self.pillars, self.benchmark = name, pillars, benchmark
        self.mode, self.pillar_t = mode, pillar_t
        self.bench = metric_series(benchmark)
        self.bench_ic = IC[benchmark]
        self._cache = {}

    def select(self, fit_periods):
        mask = IC.index.get_level_values("period").isin(fit_periods) & IN_UNIVERSE
        q = IC[mask].groupby(level="period").mean()
        ic, t = q.mean(), q.mean() / q.std() * np.sqrt(len(q))
        sel = []
        for ms in self.pillars.values():
            if self.pillar_t is not None and not (t.reindex(ms).max() >= self.pillar_t):
                sel.append(())
                continue
            pos = [m for m in ms if ic.get(m, np.nan) > 0]
            sel.append(tuple(pos) if pos else (ic.reindex(ms).idxmax(),))
        return tuple(sel)

    def weight_candidates(self, sel):
        """Full-length weight vectors: bounded grid over active pillars, last row = equal active pillars."""
        active = [j for j, ms in enumerate(sel) if ms]
        eq = np.zeros(len(sel))
        eq[active] = 1 / len(active)
        rows = []
        if self.mode == "grid":
            for g in make_grid(len(active)):
                w = np.zeros(len(sel))
                w[active] = g
                rows.append(w)
        return np.vstack(rows + [eq])

    def pillar_scores(self, cell, sel):
        n = cell["z"].shape[0]
        return np.column_stack([cell["z"][:, [COL[m] for m in ms]].mean(axis=1) if ms else np.zeros(n)
                                for ms in sel])

    def grid_returns(self, sel):
        if sel not in self._cache:
            w = self.weight_candidates(sel)
            self._cache[sel] = (w, pd.DataFrame(
                np.vstack([long_short(self.pillar_scores(c, sel) @ w.T, c["fwd"]) for c in CELLS]), index=IDX))
        return self._cache[sel]

    def fit(self, k):
        """Selection, pillar weights and per-industry exclusion using PERIODS[:k] only."""
        fit = PERIODS[:k]
        sel = self.select(fit)
        cand, g = self.grid_returns(sel)
        mask = IDX.get_level_values("period").isin(fit) & IN_UNIVERSE
        excess = g[mask].sub(self.bench[mask], axis=0).groupby(level="period").mean().mean()
        eq_col = g.columns[-1]
        best = int(excess.iloc[:-1].idxmax()) if len(excess) > 1 else eq_col
        shrunk = best == eq_col or (excess[best] - excess[eq_col]) * 4 <= SHRINK_PP
        col = eq_col if shrunk else best
        w, prof = cand[col], g[col]
        exc_ind = (prof - self.bench)[mask].groupby(level="industry")
        t_ind = exc_ind.apply(lambda s: s.mean() / s.std() * np.sqrt(len(s)) if len(s) >= MIN_FIT and s.std() > 0
                              else np.nan)
        excluded = set(t_ind.index[t_ind < T_EXCLUDE])
        return sel, w, shrunk, excluded

    def metric_weights(self, sel, w):
        return {m: wp / len(ms) for ms, wp in zip(sel, w) for m in ms}

    def cell_na(self, cell, sel, w):
        return sum(wp * cell["na"][[COL[m] for m in ms]].mean() for ms, wp in zip(sel, w) if ms)

    def walk_forward(self):
        rows, fits = [], {}
        for k in range(MIN_FIT, len(PERIODS)):
            sel, w, shrunk, excluded = self.fit(k)
            fits[PERIODS[k]] = dict(sel=sel, w=w, shrunk=shrunk, excluded=excluded)
            for i, c in enumerate(CELLS):
                if c["p"] != PERIODS[k] or c["ind"] not in INDUSTRIES:
                    continue
                score = self.pillar_scores(c, sel) @ w
                na = self.cell_na(c, sel, w)
                reason = "NA" if na > NA_MAX else ("t" if c["ind"] in excluded else "")
                r = long_short(score[:, None], c["fwd"])[0]
                rows.append(dict(period=c["p"], industry=c["ind"], prof=r, bench=self.bench.iloc[i],
                                 ic=_rank_ic(score, c["fwd"]), bench_ic=self.bench_ic.iloc[i], na=na, reason=reason))
        wf = pd.DataFrame(rows).set_index(["period", "industry"])
        wf["rule"] = np.where(wf.reason == "", wf.prof, wf.bench)
        return wf, fits


def ann_by_industry(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.groupby(level="industry").apply(lambda x: (1 + x).prod() ** (4 / len(x)) - 1).mean())


def paired_t(a: pd.Series, b: pd.Series) -> float:
    d = (a - b).groupby(level="period").mean().dropna()
    return float(d.mean() / d.std() * np.sqrt(len(d)))


def summary(prof: pd.Series, bench: pd.Series, ic=None, bench_ic=None) -> dict:
    q = (prof - bench).groupby(level="period").mean().dropna()
    roll = q.rolling(12).sum().dropna()
    by_ind = prof.groupby(level="industry").apply(lambda x: (1 + x.dropna()).prod() ** (4 / x.notna().sum()) - 1) - \
        bench.groupby(level="industry").apply(lambda x: (1 + x.dropna()).prod() ** (4 / x.notna().sum()) - 1)
    out = dict(prof=ann_by_industry(prof), bench=ann_by_industry(bench))
    out.update(excess=out["prof"] - out["bench"], t=paired_t(prof, bench), hit_q=float((q > 0).mean()),
               ind_won=int((by_ind > 0).sum()), n_ind=int(by_ind.notna().sum()),
               roll3y_won=int((roll > 0).sum()), n_roll=len(roll))
    if ic is not None:
        qi = ic.groupby(level="period").mean()
        qb = bench_ic.groupby(level="period").mean()
        out.update(ic=float(qi.mean()), bench_ic=float(qb.mean()),
                   t_ic=float((qi - qb).mean() / (qi - qb).std() * np.sqrt(len(qi))))
    return out
