import os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.dirname(HERE)
W = pickle.load(open(os.path.join(HERE, "results.pkl"), "rb"))
T = pickle.load(open(os.path.join(M, "tt", "results.pkl"), "rb"))
f = lambda s: f"{s['excess']*100:+.2f}pp t {s['t']:.2f} tIC {s['t_ic']:.2f} r {s['r']*100:+.2f} bench {s['bench']*100:+.2f} q {s['q_win']:.0%} 2a {s['roll']} ind {s['ind_win']}/{s['n_ind']}"
for k, w in W.items():
    print(f"\n=== {k} vs {w['bench']}")
    print(" 33q test      ", f(w["test_sum"]))
    for h, s in w["halves"].items(): print(f" {h}", f(s))
    print(" 18/18 test A  ", f(T[k]["res"]["A"]["test"]))
    print(" fuera NA global:", w["dropped"])
    print(" pilares solos:", {p: f"{v['excess']*100:+.2f} (t {v['t']:.2f})" for p, v in w["pillars"].items()})
    # weights difference vs 18/18
    old = T[k]["res"]["A"]["weights"]
    for pil, ws in w["weights"].items():
        o = set(old[pil].index); n = set(ws.index)
        if o != n: print(f"  {pil}: entra {sorted(n-o)} sale {sorted(o-n)}")
    ex = w["test_sum"]["ex_q"]
    print(" exceso por año:", (ex.groupby(ex.index.str[:4]).mean()*4*100).round(1).to_dict())
# industries momentum
MI = pickle.load(open(os.path.join(M, "momentum_industrias", "series.pkl"), "rb"))
ls = MI["lin"]["ls"]["6-1m (benchmark)"]
ann = lambda s: (1 + s.dropna()).prod() ** (4 / s.notna().sum()) - 1
tst = lambda s: s.dropna().mean() / s.dropna().std() * np.sqrt(s.notna().sum())
TEST = W["Reversion"]["test"]
s = ls.loc[[p for p in TEST if p in ls.index]]
print(f"\n6-1m industrias {len(s)}q: {ann(s)*100:+.1f}% t {tst(s):.2f} pos {(s>0).mean():.0%}")
d = W["Reversion"]["df"].dropna(subset=["r", "bench"])
rev, strev = d.groupby("p").r.mean(), d.groupby("p").bench.mean()
two, twob = (ls/2 + rev).loc[TEST].dropna(), (ls/2 + strev).loc[TEST].dropna()
print(f"2 capas 6-1m+RevA: {ann(two)*100:+.1f}% t {tst(two):.2f} | 6-1m+ST_Rev {ann(twob)*100:+.1f}% t {tst(twob):.2f} | dif t {tst(two-twob):.2f}")
