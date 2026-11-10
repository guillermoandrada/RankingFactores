"""Industry-selection momentum, mejora 06-11: shared panel, signals, books and stats.

Stock z SPX-wide (Yeo-Johnson -> winsor 5/95 -> Z(0,1), NA neutral) or linear (Z only, = industry return ranking)
-> industry mean -> z across industries -> score = sum(w * z). Top third minus bottom third, 3m BBG TR equal weight.
"""
import glob
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import yeojohnson, yeojohnson_normmax

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = r"C:/Users/usuario/AppData/Local/Temp/claude/C--Users-usuario-PycharmProjects-RankingFactores/35a691c6-8f31-454e-8c45-92d497426d38/scratchpad"
MIN_FIT = 8
SENTIMENT = {"Best Analyst Rating": 1, "Net Buy Rec %": 1, "Buy Rec %": 1, "Sell Rec %": -1}
BASE = ["1m Total Return", "3m Total Return", "6m Total Return", "12m Total Return", "6-1m Total Return",
        "12-1m Total Return", "12-6m Total Return"]
DERIVED = ["3-1m Total Return", "Aceleración 6-1m − 12-6m"]
METRICS = BASE + DERIVED + list(SENTIMENT)
SIGN = {m: SENTIMENT.get(m, 1) for m in METRICS}

INDUSTRIES = pickle.load(open(os.path.join(OLD, "mom_results.pkl"), "rb"))["INDUSTRIES"]


def _derive(raw):
    raw = raw.copy()
    raw["3-1m Total Return"] = (1 + raw["3m Total Return"]) / (1 + raw["1m Total Return"]) - 1
    raw["Aceleración 6-1m − 12-6m"] = raw["6-1m Total Return"] - raw["12-6m Total Return"]
    return raw


def _stock_z(raw, mode):
    """mode: yj = Yeo-Johnson + winsor 5/95 + Z; wz = winsor 5/95 + Z (app today, 0-10 rescale is affine); lin = Z only."""
    out = {}
    for m in METRICS:
        s = pd.to_numeric(raw[m], errors="coerce").astype(float)
        x = s.dropna().to_numpy(float)
        if mode == "yj" and len(x) >= 30 and np.std(x) > 0:
            lo, hi = np.quantile(x, [0.005, 0.995])
            lam = float(np.clip(yeojohnson_normmax(np.clip(x, lo, hi)), -3, 3))
            s = s.copy()
            s[s.notna()] = yeojohnson(x, lmbda=lam)
        if mode in ("yj", "wz"):
            s = s.clip(s.quantile(0.05), s.quantile(0.95))
        z = (s - s.mean()) / s.std(ddof=0)
        out[m] = z.fillna(0.0) * SIGN[m]
    return pd.DataFrame(out, index=raw.index)


def _standardize(s):
    return (s - s.mean()) / s.std(ddof=0)


def build_panel(cache=os.path.join(HERE, "panel.pkl")):
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    out = {}
    for f in sorted(glob.glob(os.path.join(OLD, "panel", "*.pkl"))):
        d = pickle.load(open(f, "rb"))
        raw = _derive(d["raw"][d["raw"].industry.isin(INDUSTRIES)])
        fwd = raw.ticker_c.map(d["fwd"][3])
        entry = {"fwd": fwd.groupby(raw.industry).mean(), "firms": raw.groupby("industry").size()}
        for mode in ("yj", "wz", "lin"):
            z = _stock_z(raw, mode)
            entry[mode] = z.groupby(raw.industry).mean().apply(_standardize)
        out[d["period"]] = entry
    pickle.dump(out, open(cache, "wb"))
    return out


def book(panel, p, weights, mode, industries=None):
    e = panel[p]
    z = e[mode] if industries is None else e[mode].loc[e[mode].index.isin(industries)]
    score = sum(w * z[m] for m, w in weights.items())
    fwd = e["fwd"].reindex(score.index)
    ok = score.notna() & fwd.notna()
    a, b = score[ok], fwd[ok]
    k = len(a) // 3
    return dict(ls=b[a.nlargest(k).index].mean() - b[a.nsmallest(k).index].mean(),
                ic=a.rank().corr(b.rank()))


def t_stat(s):
    s = pd.Series(s).dropna()
    return s.mean() / s.std() * np.sqrt(len(s)) if len(s) > 2 and s.std() > 0 else np.nan


def ann(r):
    r = pd.Series(r).dropna()
    return (1 + r).prod() ** (4 / len(r)) - 1
