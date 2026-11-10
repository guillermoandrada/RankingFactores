"""Value improvement (protocol 2026-11-06): cell construction (DB NA convention -> Yeo-Johnson -> winsor 5/95 ->
Z(0,1), signed), FF B/M benchmark, long-short, IC and summary statistics."""
import os
import glob
import pickle

import numpy as np
import pandas as pd
from scipy.stats import yeojohnson, yeojohnson_normmax

HERE = os.path.dirname(os.path.abspath(__file__))
PERIODS = [f"{y}/{m}" for y in range(2017, 2027) for m in ("03/30", "06/30", "09/30", "12/30")]
PERIODS = [p for p in PERIODS if "2017/03/30" <= p <= "2026/03/30" and (p != "2024/06/30" or os.environ.get("W33"))]
MIN_FIT = 8
TEST = PERIODS[MIN_FIT:]
FF = "Current Book to Market"
NOT_SCORED = {FF, "Market Cap"}

PANEL_DIR = os.environ.get("VPANEL", "rpanel")
# Phase 2 data fix: quarters where a metric is missing for the whole S&P use a proxy (Est EV/EBITDA -> Current)
PROXY = {"Estimated EV / EBITDA": "Current EV / EBITDA"} if PANEL_DIR != "rpanel" else {}
PANEL = {}
for f in glob.glob(os.path.join(HERE, PANEL_DIR, "*.pkl")):
    d = pickle.load(open(f, "rb"))
    if d["period"] in PERIODS:
        PANEL[d["period"]] = d
NA_MAP = pickle.load(open(os.path.join(HERE, "na_map.pkl" if PANEL_DIR == "rpanel" else PANEL_DIR + "_na_map.pkl"), "rb"))
DIRECTION = {}
for p in PERIODS:
    DIRECTION.update(PANEL[p]["direction"])
FIRMS = pd.DataFrame({p: {i: len(z) for i, z in PANEL[p]["frames"].items()} for p in PERIODS}).fillna(0)


def apply_na(df: pd.DataFrame) -> pd.DataFrame:
    """DB NA convention per industry: high -> industry max, zero -> 0, None -> left NaN (neutral after Z)."""
    out = df.copy()
    for m in out.columns:
        rule = NA_MAP.get(m)
        if rule == "replace_with_high":
            out[m] = out[m].fillna(out[m].max() if out[m].notna().any() else 0.0)
        elif rule == "replace_with_zero":
            out[m] = out[m].fillna(0.0)
        elif rule == "replace_with_low":
            out[m] = out[m].fillna(out[m].min() if out[m].notna().any() else 0.0)
    return out


def yj_lambdas(metrics):
    lam = {}
    for p in PERIODS:
        pooled = pd.concat([apply_na(fr[metrics].apply(pd.to_numeric, errors="coerce").astype(float))
                            for fr in PANEL[p]["frames"].values()])
        for m in metrics:
            x = pooled[m].dropna().to_numpy(float)
            if len(x) >= 30 and np.std(x) > 0:
                lo, hi = np.quantile(x, [0.005, 0.995])
                lam[(p, m)] = float(np.clip(yeojohnson_normmax(np.clip(x, lo, hi)), -3, 3))
    return lam


def standardize(filled: pd.DataFrame, p: str, lam: dict) -> np.ndarray:
    df = filled.copy()
    for m in df.columns:
        lm, s = lam.get((p, m)), df[m]
        if lm is not None and s.notna().any():
            df.loc[s.notna(), m] = yeojohnson(s[s.notna()].to_numpy(float), lmbda=lm)
    w = df.clip(lower=df.quantile(.05), upper=df.quantile(.95), axis=1)
    mu, sd = w.mean(), w.std(ddof=0)
    return ((w.fillna(mu) - mu) / sd.replace(0, np.nan)).fillna(0.0).to_numpy()


def long_short(score: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """score n x k, fwd n (NaN -> 0). Top half minus bottom half, equal weight."""
    size = int(np.ceil(score.shape[0] / 2))
    order = np.argsort(-score, axis=0, kind="stable")
    r = np.nan_to_num(fwd)[order]
    return r[:size].mean(axis=0) - r[-size:].mean(axis=0)


def rank_ic(score: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """Spearman IC of each score column with fwd (rows with fwd NaN dropped)."""
    ok = ~np.isnan(fwd)
    if ok.sum() < 5:
        return np.full(score.shape[1], np.nan)
    s = pd.DataFrame(score[ok]).rank().to_numpy()
    f = pd.Series(fwd[ok]).rank().to_numpy()
    s, f = s - s.mean(axis=0), f - f.mean()
    den = np.sqrt((s ** 2).sum(axis=0) * (f ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s * f[:, None]).sum(axis=0) / den


def proxy_missing():
    """Replace a metric by its proxy in quarters where > 50% of the S&P lacks it (whole-download gaps)."""
    for p in PERIODS:
        frames = PANEL[p]["frames"]
        for m, proxy in PROXY.items():
            pooled = pd.concat([fr[m] for fr in frames.values()])
            if pd.to_numeric(pooled, errors="coerce").isna().mean() > 0.5:
                for fr in frames.values():
                    fr[m] = fr[proxy]
                PANEL[p].setdefault("proxied", []).append(m)


proxy_missing()
if PANEL_DIR != "rpanel":
    for _p in PERIODS:
        for _fr in PANEL[_p]["frames"].values():
            _fr["OCF Yield"] = (pd.to_numeric(_fr["Current Operating Cash Flow"], errors="coerce")
                                / pd.to_numeric(_fr["Market Cap"], errors="coerce").replace(0, np.nan))


def build_cells(metrics, min_firms=5):
    lam = yj_lambdas(metrics)
    sign = np.array([1.0 if DIRECTION.get(m, True) else -1.0 for m in metrics])
    industries = sorted(FIRMS.index[FIRMS.mean(axis=1) >= min_firms])
    cells = []
    for p in PERIODS:
        for ind in industries:
            fr = PANEL[p]["frames"].get(ind)
            if fr is None or len(fr) < 2:
                continue
            raw = fr[metrics].apply(pd.to_numeric, errors="coerce").astype(float)
            fwd = fr.ticker.map(PANEL[p]["fwd"][3]).to_numpy(float)
            bm = pd.to_numeric(fr[FF], errors="coerce").to_numpy(float)
            ok = ~np.isnan(bm)
            ff_ret = long_short(bm[ok][:, None], fwd[ok])[0] if ok.sum() >= 2 else np.nan
            ff_ic = rank_ic(bm[ok][:, None], fwd[ok])[0] if ok.sum() >= 2 else np.nan
            cells.append(dict(p=p, ind=ind, z=standardize(apply_na(raw), p, lam) * sign, raw=raw,
                              na=raw.isna().to_numpy(), fwd=fwd, ff=ff_ret, ff_ic=ff_ic, n=len(fr)))
    return cells, industries


def ann(s: pd.Series, periods) -> float:
    """Geometric annualized return per industry, averaged across industries (project convention)."""
    r = s[s.index.get_level_values("period").isin(periods)].dropna()
    if r.empty:
        return np.nan
    return float(r.groupby(level="industry").apply(lambda x: (1 + x).prod() ** (4 / len(x)) - 1).mean())


def ann_by_industry(s: pd.Series, periods) -> pd.Series:
    r = s[s.index.get_level_values("period").isin(periods)].dropna()
    return r.groupby(level="industry").apply(lambda x: (1 + x).prod() ** (4 / len(x)) - 1)


def paired_t(diff_by_period: pd.Series) -> float:
    d = diff_by_period.dropna()
    return float(d.mean() / d.std() * np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan
