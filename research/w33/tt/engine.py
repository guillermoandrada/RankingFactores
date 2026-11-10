"""Common train/test engine (pre-registration mejora/tt/spec.md + implementacion.md).

A factor is described by FactorData: cells (one per period x industry) with signed z-scores, forward 3m return,
raw NA share per metric, benchmark long-short return and IC. The profile is trained once on the first TRAIN_N
periods (metric weights inside fixed theory pillars) and evaluated once on the rest."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRAIN_N, NA_MAX, SHRINK, CAP = 18, 0.40, 0.5, 0.5
VARIANTS = ["A", "B", "C", "D"]


@dataclass
class FactorData:
    name: str
    periods: list
    cells: list                      # dicts: p, ind, z (n x k), fwd (n), na (k), bench, bench_ic
    col: dict                        # metric -> column index
    pillars: dict                    # pillar -> candidate metrics
    pillar_w: dict                   # pillar -> theory weight (sums to 1)
    canonical: set
    bench_name: str
    subs: dict = field(default_factory=dict)    # industry -> {pillar: metrics} (theory substitution, equal wt)
    extra: dict = field(default_factory=dict)   # industry -> {pillar: extra metrics}

    @property
    def train(self):
        return self.periods[:TRAIN_N]

    @property
    def test(self):
        return self.periods[TRAIN_N:]


def long_short(score, fwd):
    size = int(np.ceil(len(score) / 2))
    order = np.argsort(-score, kind="stable")
    r = np.nan_to_num(fwd)[order]
    return r[:size].mean() - r[-size:].mean()


def spearman(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 5 or np.std(a[ok]) == 0:
        return np.nan
    return float(pd.Series(a[ok]).rank().corr(pd.Series(b[ok]).rank()))


def cell_ic(c, j):
    key = ("ic", j)
    if key not in c:
        c[key] = spearman(c["z"][:, j], c["fwd"])
    return c[key]


# ---------------------------------------------------------------- training statistics (train window only)
def train_stats(fd: FactorData):
    tr = [c for c in fd.cells if c["p"] in fd.train]
    nonsub = [c for c in tr if c["ind"] not in fd.subs]
    metrics = sorted({m for ms in fd.pillars.values() for m in ms})
    na_glob = {m: float(np.mean([c["na"][fd.col[m]] for c in nonsub])) for m in metrics}
    ic_q = pd.DataFrame({p: {m: np.nanmean([cell_ic(c, fd.col[m]) for c in tr if c["p"] == p]) for m in metrics}
                         for p in fd.train}).T
    inds = sorted({c["ind"] for c in fd.cells})
    all_m = sorted({m for ms in fd.pillars.values() for m in ms}
                   | {m for d in list(fd.subs.values()) + list(fd.extra.values()) for ms in d.values() for m in ms})
    na_ind = {ind: {m: float(np.mean([c["na"][fd.col[m]] for c in tr if c["ind"] == ind] or [1.0])) for m in all_m}
              for ind in inds}
    corr = {}
    for pil, ms in fd.pillars.items():
        idx = [fd.col[m] for m in ms]
        if len(ms) == 1:
            corr[pil] = pd.DataFrame([[1.0]], index=ms, columns=ms)
            continue
        acc, n = np.zeros((len(ms), len(ms))), 0
        for c in tr:
            z = c["z"][:, idx]
            if z.shape[0] < 5:
                continue
            sd = z.std(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.corrcoef(z, rowvar=False)
            r = np.where(np.isnan(r), 0.0, r)
            r[sd == 0, :] = 0.0
            r[:, sd == 0] = 0.0
            np.fill_diagonal(r, 1.0)
            acc += r
            n += 1
        corr[pil] = pd.DataFrame(acc / max(n, 1), index=ms, columns=ms)
    return dict(ic=ic_q.mean(), ic_sd=ic_q.std(), ic_t=ic_q.mean() / ic_q.std() * np.sqrt(len(ic_q)),
                na_glob=na_glob, na_ind=na_ind, corr=corr)


def _max_ir(ic, sd, rho):
    n = len(ic)
    if n == 1:
        return np.array([1.0])
    rho_s = SHRINK * rho + (1 - SHRINK) * np.eye(n)
    sigma = np.outer(sd, sd) * rho_s
    cap = max(CAP, 1.0 / n)
    x0 = np.linalg.solve(sigma, ic)
    x0 = np.clip(x0, 1e-6, None)
    x0 = np.minimum(x0 / x0.sum(), cap)
    x0 /= x0.sum()
    res = minimize(lambda w: -(w @ ic) / np.sqrt(w @ sigma @ w), x0, method="SLSQP",
                   bounds=[(0.0, cap)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options=dict(ftol=1e-12, maxiter=500))
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def within_weights(fd: FactorData, st):
    """{variant: {pillar: pd.Series(metric -> weight inside pillar)}} on the globally available candidates."""
    out = {v: {} for v in VARIANTS}
    for pil, ms in fd.pillars.items():
        avail = [m for m in ms if st["na_glob"][m] <= NA_MAX]
        if not avail:  # implementacion.md item 4: empty pillar contributes 0 (pro rata to the others)
            print(f"  {fd.name}: pilar {pil} vacío por NA > 40% en entrenamiento: "
                  + ", ".join(f"{m} {st['na_glob'][m]:.0%}" for m in ms), flush=True)
            for v in VARIANTS:
                out[v][pil] = pd.Series(dtype=float)
            continue
        a = pd.Series(1.0 / len(avail), index=avail)
        b = pd.Series([2.0 if m in fd.canonical else 1.0 for m in avail], index=avail)
        b /= b.sum()
        pos = [m for m in avail if st["ic"][m] > 0]
        if pos:
            w = _max_ir(st["ic"][pos].to_numpy(), st["ic_sd"][pos].to_numpy(),
                        st["corr"][pil].loc[pos, pos].to_numpy())
            c = pd.Series(0.0, index=avail)
            c[pos] = w
        else:
            c = a.copy()
        out["A"][pil], out["B"][pil], out["C"][pil] = a, b, c
        out["D"][pil] = 0.5 * b + 0.5 * c
    return out


def industry_vector(fd: FactorData, within, st, ind):
    """Full metric weight vector for one industry (theory substitutions at equal weight, NA filter)."""
    vec = np.zeros(len(fd.col))
    na = st["na_ind"].get(ind, {})
    for pil, wp in fd.pillar_w.items():
        if pil in fd.subs.get(ind, {}):
            ms = [m for m in fd.subs[ind][pil] if na.get(m, 1.0) <= NA_MAX]
            w = pd.Series(1.0 / len(ms), index=ms) if ms else pd.Series(dtype=float)
        else:
            w = within[pil]
            extra = fd.extra.get(ind, {}).get(pil, [])
            if extra:
                w = pd.concat([w, pd.Series(w.mean() if len(w) else 1.0, index=extra)])
            w = w[[na.get(m, 1.0) <= NA_MAX for m in w.index]]
            w = w / w.sum() if w.sum() > 0 else w
        for m, x in w.items():
            vec[fd.col[m]] += wp * x
    return vec


def evaluate(fd: FactorData, within, st):
    vecs, rows = {}, []
    for c in fd.cells:
        if c["ind"] not in vecs:
            vecs[c["ind"]] = industry_vector(fd, within, st, c["ind"])
        s = c["z"] @ vecs[c["ind"]]
        rows.append(dict(p=c["p"], ind=c["ind"], r=long_short(s, c["fwd"]), bench=c["bench"],
                         ic=spearman(s, c["fwd"]), bench_ic=c["bench_ic"]))
    return pd.DataFrame(rows)


def ann_by_ind(d, col):
    return d.groupby("ind")[col].apply(lambda s: (1 + s.dropna()).prod() ** (4 / max(1, s.notna().sum())) - 1)


def paired_t(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def summarize(df, periods, window=8):
    d = df[df.p.isin(periods)].dropna(subset=["r", "bench"])
    ex_q = (d.r - d.bench).groupby(d.p).mean()
    exic_q = (d.ic - d.bench_ic).groupby(d.p).mean()
    r_i, b_i = ann_by_ind(d, "r"), ann_by_ind(d, "bench")
    roll = ex_q.rolling(window).mean().dropna()
    return dict(r=r_i.mean(), bench=b_i.mean(), excess=r_i.mean() - b_i.mean(), t=paired_t(ex_q),
                t_ic=paired_t(exic_q), ic=d.groupby("p").ic.mean().mean(), bench_ic=d.groupby("p").bench_ic.mean().mean(),
                q_win=float((ex_q > 0).mean()), ind_win=int(((r_i - b_i) > 0).sum()), n_ind=int(len(r_i)),
                roll=f"{int((roll > 0).sum())}/{len(roll)}", ex_q=ex_q, by_ind=r_i - b_i)


def run(fd: FactorData):
    st = train_stats(fd)
    within = within_weights(fd, st)
    res = {}
    for v in VARIANTS:
        df = evaluate(fd, within[v], st)
        res[v] = dict(df=df, test=summarize(df, fd.test), train=summarize(df, fd.train),
                      weights={pil: (fd.pillar_w[pil] * w).round(4) for pil, w in within[v].items()})
    for v in ("B", "C", "D"):
        for part, per in (("test", fd.test), ("train", fd.train)):
            a = res["A"]["df"][res["A"]["df"].p.isin(per)]
            b = res[v]["df"][res[v]["df"].p.isin(per)]
            diff = (b.r.to_numpy() - a.r.to_numpy())
            res[v][f"vsA_{part}"] = dict(diff=float(np.nanmean(pd.Series(diff).groupby(a.p.to_numpy()).mean()) * 4),
                                         t=paired_t(pd.Series(diff).groupby(a.p.to_numpy()).mean()))
    return dict(stats=st, within=within, res=res)
