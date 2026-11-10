"""Robustness of the final profiles under two other portfolio constructions (test 2018/03-2026/03):
A) top 30 vs bottom 30 of the whole universe (composite z re-standardised within industry), equal weight.
B) long-only legacy rebalance (app function, default params: floor 5 on 0-10 score, top half per industry,
   3% per name ordered by score), new portfolio each quarter. Score on the app 0-10 scale (min-max per metric
   within industry, then weighted sum). Benchmarks built identically from the FF/ST_Rev single metric.
Run from repo root."""
import os, sys, pickle
import numpy as np, pandas as pd
REPO = os.getcwd(); sys.path.insert(0, REPO)
OLD = r"C:\Users\usuario\AppData\Local\Temp\claude\33de0e63510b7b04\5103b39d-3463-42e5-a5a9-77cdb0e23eb2\scratchpad\mejora\w33"
sys.path.insert(0, OLD)
import final_profiles as FP
from modules.domain.portfolio.legacy_rebalance import build_legacy_rebalance_target
from modules.domain.portfolio.models import SecurityCandidate

TEST0, SPLIT, N = "2018/03/30", "2021/09/30", 30


def weight_vector(fd, spec, c):
    w, sets = spec(c["ind"]); tot = sum(w.values()); v = np.zeros(len(fd.col))
    for pil, wp in w.items():
        ms = [m for m in sets.get(pil, []) if c["na"][fd.col[m]] <= FP.NA_MAX]
        for m in ms:
            v[fd.col[m]] += wp / tot / len(ms)
    return v


def minmax10(z):
    lo, hi = z.min(axis=0), z.max(axis=0); rng = np.where(hi > lo, hi - lo, np.nan)
    return np.nan_to_num((z - lo) / rng * 10, nan=5.0)


def std(x):
    s = x.std(); return (x - x.mean()) / s if s > 0 else x * 0


def quarter_frames(fd, spec, bench_col):
    out = {}
    for c in fd.cells:
        if c["p"] < TEST0:
            continue
        v = weight_vector(fd, spec, c); jb = fd.col[bench_col]
        mm = minmax10(c["z"])
        df = pd.DataFrame(dict(ind=c["ind"], fwd=np.nan_to_num(c["fwd"]),
                               z_prof=std(c["z"] @ v), z_bench=std(c["z"][:, jb]),
                               app_prof=mm @ v, app_bench=mm[:, jb]))
        out.setdefault(c["p"], []).append(df)
    return {p: pd.concat(v, ignore_index=True) for p, v in out.items()}


def top_bottom(df, col):
    o = df.sort_values(col, ascending=False, kind="stable")
    return o.fwd.iloc[:N].mean() - o.fwd.iloc[-N:].mean()


def legacy(df, col):
    cands = [SecurityCandidate(ticker=str(i), score=float(s), industry=ind) for i, (s, ind) in enumerate(zip(df[col], df.ind))]
    pos, _ = build_legacy_rebalance_target(cands)
    w = pd.Series({int(p.ticker): p.target_weight for p in pos})
    return float((df.fwd.loc[w.index] * w).sum() / w.sum()), len(w)


def stats(a, b):
    a, b = pd.Series(a), pd.Series(b); d = a - b
    ann = lambda s: (1 + s).prod() ** (4 / len(s)) - 1
    t = d.mean() / d.std() * np.sqrt(len(d))
    h1, h2 = d[d.index < SPLIT], d[d.index >= SPLIT]
    return dict(prof=ann(a) * 100, bench=ann(b) * 100, excess=(ann(a) - ann(b)) * 100, t=t, win=(d > 0).mean() * 100,
                sharpe=a.mean() * 4 / (a.std() * 2), sharpe_b=b.mean() * 4 / (b.std() * 2),
                h1=h1.mean() * 400, h2=h2.mean() * 400)


FACTORS = {"Quality": (FP.quality, "Current ROE", "FF ROE"),
           "Value": (FP.value, "Y:Current P/B Ratio", "FF B/M"),
           "Reversion": (FP.reversion, "1m Return Reversal", "ST_Rev")}
res = {}
for name, (loader, bcol, bname) in FACTORS.items():
    fd, spec, _ = loader()
    Q = quarter_frames(fd, spec, bcol)
    tb = {p: top_bottom(d, "z_prof") for p, d in Q.items()}; tbb = {p: top_bottom(d, "z_bench") for p, d in Q.items()}
    lg = {p: legacy(d, "app_prof") for p, d in Q.items()}; lgb = {p: legacy(d, "app_bench") for p, d in Q.items()}
    ew = {p: d.fwd.mean() for p, d in Q.items()}
    lgr = {p: v[0] for p, v in lg.items()}; lgbr = {p: v[0] for p, v in lgb.items()}
    res[name] = dict(bench=bname, top_bottom=stats(tb, tbb), legacy_vs_bench=stats(lgr, lgbr),
                     legacy_vs_ew=stats(lgr, ew), bench_legacy_vs_ew=stats(lgbr, ew),
                     n_names=np.mean([v[1] for v in lg.values()]), n_univ=np.mean([len(d) for d in Q.values()]))
    print(name, {k: (v if not isinstance(v, dict) else {a: round(b, 2) for a, b in v.items()}) for k, v in res[name].items()}, flush=True)
pickle.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "robust_portfolios.pkl"), "wb"))
