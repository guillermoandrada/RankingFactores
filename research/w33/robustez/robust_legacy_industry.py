"""Legacy rebalance con restricción por industria: objetivo = peso de cada industria en el índice (Market Cap)
en ese mismo trimestre, renormalizado sobre las industrias del universo del factor. Referencia pasiva = índice
a peso igual dentro de industria, industrias a su peso de capitalización del trimestre. Run from repo root."""
import glob, os, sys, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import robust_portfolios as RP  # reuses loaders and helpers (runs its own tests first)
from modules.domain.portfolio.legacy_rebalance import build_legacy_rebalance_target
from modules.domain.portfolio.models import SecurityCandidate

CAP = {}
for f in glob.glob(os.path.join(os.path.dirname(RP.OLD), "value", "rpanel_v33", "*.pkl")):
    d = pickle.load(open(f, "rb"))
    CAP[d["period"]] = {i: pd.to_numeric(fr["Market Cap"], errors="coerce").sum() for i, fr in d["frames"].items()}


def targets(p, inds):
    w = pd.Series({i: CAP[p].get(i, np.nan) for i in inds}).dropna()
    return (w / w.sum()).to_dict()


def legacy_ind(df, col, tg):
    cands = [SecurityCandidate(ticker=str(i), score=float(s), industry=ind) for i, (s, ind) in enumerate(zip(df[col], df.ind))]
    pos, _ = build_legacy_rebalance_target(cands, constraint_type="industry", industry_targets=tg)
    w = pd.Series({int(p.ticker): p.target_weight for p in pos})
    return float((df.fwd.loc[w.index] * w).sum() / w.sum()), len(w), w.sum()


res = {}
for name, (loader, bcol, bname) in RP.FACTORS.items():
    fd, spec, _ = loader()
    Q = RP.quarter_frames(fd, spec, bcol)
    pr, br, ref, nn, alloc = {}, {}, {}, [], []
    for p, d in Q.items():
        tg = targets(p, d.ind.unique())
        d = d[d.ind.isin(tg)]
        pr[p], n, a = legacy_ind(d, "app_prof", tg); br[p] = legacy_ind(d, "app_bench", tg)[0]
        ref[p] = sum(tg[i] * g.fwd.mean() for i, g in d.groupby("ind")); nn.append(n); alloc.append(a)
    res[name] = dict(vs_bench=RP.stats(pr, br), vs_ref=RP.stats(pr, ref), bench_vs_ref=RP.stats(br, ref),
                     n_names=np.mean(nn), alloc=np.mean(alloc))
    r = res[name]
    print(f"\n=== {name} vs {bname} legacy por industria  (nombres {r['n_names']:.0f}, peso asignado {r['alloc']:.2f})")
    print(pd.DataFrame({k: v for k, v in r.items() if isinstance(v, dict)}).T.round(2).to_string(), flush=True)
pickle.dump(res, open(os.path.join(HERE, "robust_legacy_industry.pkl"), "wb"))
