"""Exceso sobre FF ponderando industrias por su capitalización en el S&P 500 (vs peso igual)."""
import glob, os, pickle
import numpy as np, pandas as pd
OLD = r"C:\Users\usuario\AppData\Local\Temp\claude\33de0e63510b7b04\5103b39d-3463-42e5-a5a9-77cdb0e23eb2\scratchpad\mejora"
rows = pickle.load(open(os.path.join(OLD, "w33", "final_rows.pkl"), "rb"))
cap, nf = {}, {}
for f in glob.glob(os.path.join(OLD, "value", "rpanel_v33", "*.pkl")):
    d = pickle.load(open(f, "rb"))
    cap[d["period"]] = {i: pd.to_numeric(fr["Market Cap"], errors="coerce").sum() for i, fr in d["frames"].items()}
    nf[d["period"]] = {i: len(fr) for i, fr in d["frames"].items()}
cap, nf = pd.DataFrame(cap).T, pd.DataFrame(nf).T
T0 = "2018/03/30"
for name, key, bench in (("Quality", "q", "FF ROE"), ("Value", "v", "FF B/M")):
    df = rows[key].dropna(subset=["r", "bench"]); df = df[df.p >= T0]
    df["ex"] = df.r - df.bench
    df["cap"] = [cap.loc[p, i] for p, i in zip(df.p, df.ind)]
    df["nf"] = [nf.loc[p, i] for p, i in zip(df.p, df.ind)]
    out = {}
    for wname, wcol in (("igual", None), ("capitalización", "cap"), ("nº empresas", "nf")):
        g = df.groupby("p")
        f = (lambda c: g.apply(lambda x: x[c].mean())) if wcol is None else (lambda c: g.apply(lambda x: np.average(x[c], weights=x[wcol])))
        pq, bq = f("r"), f("bench"); ex = pq - bq
        ann = lambda s: (1 + s).prod() ** (4 / len(s)) - 1
        out[wname] = dict(perfil=ann(pq) * 100, bench=ann(bq) * 100, exceso=(ann(pq) - ann(bq)) * 100,
                          t=ex.mean() / ex.std() * np.sqrt(len(ex)), gana_trim=(ex > 0).mean() * 100,
                          v2a=f"{int((ex.rolling(8).mean().dropna() > 0).sum())}/{len(ex) - 7}")
    print(f"\n=== {name} vs {bench} (2018/03–2026/03)")
    print(pd.DataFrame(out).T.round(2).to_string())
    # peso medio de cada industria en el índice (testables) y su exceso anual medio
    w = df.groupby("ind").cap.mean(); w = w / w.sum() * 100
    exi = df.groupby("ind").ex.mean() * 400
    t = pd.DataFrame({"peso_%": w.round(1), "exceso_pp": exi.round(1)}).sort_values("peso_%", ascending=False)
    print(t.to_string())
    print(f"peso del índice (testables) en industrias que gana: {w[exi > 0].sum():.0f}%;  corr(peso, exceso) = {np.corrcoef(w, exi)[0, 1]:+.2f}")
