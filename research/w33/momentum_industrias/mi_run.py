"""Nested walk-forward search for an industry-selection profile that beats 6-1m. Saves series.pkl and prints tables."""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mi_lib import HERE, MIN_FIT, SENTIMENT, ann, book, build_panel, t_stat  # noqa: E402

PANEL = build_panel()
PERIODS = sorted(PANEL)
TEST = PERIODS[MIN_FIT:]
BENCH = {"6-1m Total Return": 1.0}
STEPS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]


def sentiment(w):
    return {"6-1m Total Return": round(1 - w, 2), **{m: w / len(SENTIMENT) for m in SENTIMENT}}


CANDIDATES = {
    "6-1m (benchmark)": BENCH,
    "12-1m": {"12-1m Total Return": 1.0},
    "6m con último mes": {"6m Total Return": 1.0},
    "Horizontes 3/6/12 (−1m)": {"3-1m Total Return": 1 / 3, "6-1m Total Return": 1 / 3, "12-1m Total Return": 1 / 3},
    "Horizontes 3/6/12 con último mes": {"3m Total Return": 1 / 3, "6m Total Return": 1 / 3, "12m Total Return": 1 / 3},
    **{f"6-1m + aceleración {w:.0%}": {"6-1m Total Return": round(1 - w, 2), "Aceleración 6-1m − 12-6m": w} for w in STEPS},
    **{f"6-1m + sentimiento {w:.0%}": sentiment(w) for w in STEPS},
}


def run(mode, industries=None):
    rows = [dict(period=p, signal=c, **book(PANEL, p, w, mode, industries)) for p in PERIODS for c, w in CANDIDATES.items()]
    det = pd.DataFrame(rows)
    ls = det.pivot(index="period", columns="signal", values="ls")
    ic = det.pivot(index="period", columns="signal", values="ic")
    picks = {}
    for k in range(MIN_FIT, len(PERIODS)):
        fit = PERIODS[:k]
        excess = (ls.loc[fit].sub(ls.loc[fit, "6-1m (benchmark)"], axis=0)).mean() * 4
        best = excess.drop("6-1m (benchmark)").idxmax()
        picks[PERIODS[k]] = best if excess[best] > 0.005 else "6-1m (benchmark)"
    ls["Selección WF anidada"] = [ls.loc[p, picks[p]] if p in picks else np.nan for p in ls.index]
    ic["Selección WF anidada"] = [ic.loc[p, picks[p]] if p in picks else np.nan for p in ic.index]
    for fam in ("aceleración", "sentimiento"):  # nested choice of the add-on weight only (0% = benchmark)
        cols = ["6-1m (benchmark)"] + [c for c in CANDIDATES if fam in c]
        name, sel_ls, sel_ic = f"6-1m + {fam}, peso WF", [], []
        for p in ls.index:
            if p not in picks:
                sel_ls.append(np.nan); sel_ic.append(np.nan); continue
            fit = PERIODS[:PERIODS.index(p)]
            ex = ls.loc[fit, cols].sub(ls.loc[fit, "6-1m (benchmark)"], axis=0).mean() * 4
            c = ex.idxmax() if ex.max() > 0.005 else "6-1m (benchmark)"
            sel_ls.append(ls.loc[p, c]); sel_ic.append(ic.loc[p, c])
        ls[name], ic[name] = sel_ls, sel_ic
    return ls, ic, pd.Series(picks)


def compare(ls, ic, periods, bench="6-1m (benchmark)"):
    out = {}
    b = ls.loc[periods, bench]
    for c in ls.columns:
        s, d = ls.loc[periods, c], ls.loc[periods, c] - b
        dic = ic.loc[periods, c] - ic.loc[periods, bench]
        roll = [(1 + s.iloc[i:i + 12]).prod() > (1 + b.iloc[i:i + 12]).prod() for i in range(len(s) - 11)]
        out[c] = dict(ann=ann(s), bench=ann(b), excess=ann(s) - ann(b), t_ls=t_stat(d), t_ic=t_stat(dic), ic=ic.loc[periods, c].mean(),
                      hit_q=(d > 0).mean() if c != bench else np.nan, win_3y=np.mean(roll) if c != bench else np.nan,
                      t_abs=t_stat(s))
    return pd.DataFrame(out).T


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    res = {}
    for mode in ("yj", "wz", "lin"):
        ls, ic, picks = run(mode)
        res[mode] = dict(ls=ls, ic=ic, picks=picks, wf=compare(ls, ic, TEST), full=compare(ls, ic, PERIODS))
        print(f"\n=== {mode}: WF {len(TEST)} trimestres ===\n", res[mode]["wf"].round(3).to_string())
        print("picks\n", picks.value_counts())
    firms = pd.DataFrame({p: PANEL[p]["firms"] for p in PERIODS}).mean(axis=1)
    big = list(firms[firms >= 8].index)
    for mode in ("yj", "wz", "lin"):
        ls, ic, picks = run(mode, big)
        res[mode + "_8"] = dict(ls=ls, ic=ic, picks=picks, wf=compare(ls, ic, TEST))
        print(f"\n=== {mode}, industrias >= 8 empresas ({len(big)}) ===\n",
              res[mode + "_8"]["wf"].loc[["6-1m (benchmark)", "Selección WF anidada", "6-1m + sentimiento, peso WF",
                                          "6-1m + aceleración, peso WF", "12-1m", "6m con último mes"]].round(3).to_string())
    res["firms"] = firms
    res["TEST"] = TEST
    res["PERIODS"] = PERIODS
    pickle.dump(res, open(os.path.join(HERE, "series.pkl"), "wb"))
