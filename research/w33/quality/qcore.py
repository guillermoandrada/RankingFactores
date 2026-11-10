"""Quality improvement (protocol 2026-11-06): panel -> YJ+Z cells, pillar definitions, LS/IC helpers.
Builds and caches qcells.pkl on first import."""
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
MIN_AVG_FIRMS = 8
NA_MAX = 0.40
FF = "Current ROE"

# Stability metrics built here (lower = more stable); all are squares of DB/derived metrics so the app
# can reproduce them with a derived formula x * x.
SQUARES = {"EBIT Margin Change Sq": ("EBIT Margin Change", None), "ROA Change Sq": ("ROA Change", None),
           "GP/Assets Change Sq": ("GP/Assets Change", None),
           "ROIC Dev 5Y Sq": ("Current ROIC", "5Y Average ROIC")}
# Fase 2 scale-free stability ratios: name -> (numerator, denominator, squared); lower = more stable
RATIOS = {"EPS Var / Price": ("EPS Variability", "Last Price", False),
          "NI Vol / Assets": ("Net Income Volatility", "Current Total Assets", False),
          "NI CV Squared": ("Net Income Volatility", "Current Net Income", True)}

PILLARS = {
    "Rentabilidad": ["Current ROE", "Current ROA", "Current ROIC", "Estimated ROE", "Estimated ROA",
                     "5Y Average ROIC", "Current GP/Assets", "Estimated GP/Assets"],
    "Calidad de beneficios": ["Current Accruals/Assets", "Current CFO/Assets", "Estimated CFO/Assets",
                              "Current OCF/Net Income", "Estimated OCF/Net Income"],
    "Endeudamiento": ["Current Net Debt / EBITDA", "Estimated Net Debt / EBITDA", "Current Net Debt/Assets",
                      "Estimated Net Debt/Assets", "Current Net Debt/OCF", "Current OCF/LT Debt",
                      "Current LT Debt/Assets"],
    "Estabilidad": ["EPS CV Squared", "EBIT Margin Change Sq", "ROA Change Sq", "GP/Assets Change Sq",
                    "ROIC Dev 5Y Sq"],
    "Márgenes": ["Current EBIT Margin", "Estimated EBIT Margin", "Current Gross Margin", "Estimated Gross Margin"],
    "Change/Trend": ["ROA Trend", "ROA Change", "CFO/Assets Change", "CFO/Assets Trend", "OCF/Net Income Change",
                     "ROE Change", "ROE Trend", "ROIC Change", "EBIT Margin Change", "EBIT Margin Trend",
                     "GP/Assets Change", "GP/Assets Trend", "Gross Margin Change", "OCF/LT Debt Change"],
}
# Theory substitutions (fixed before results): pillar -> metrics at equal weight; [] = pillar does not apply.
SUBSTITUTIONS = {
    "Banks": {"Rentabilidad": ["Current ROE", "Current ROA", "Estimated ROE", "Estimated ROA"],
              "Calidad de beneficios": [],
              "Endeudamiento": ["Current Equity/Assets", "Estimated Equity/Assets"],
              "Estabilidad": ["EPS CV Squared", "ROA Change Sq"],
              "Márgenes": ["Current Net Interest Margin", "Past Efficiency Ratio"],
              "Change/Trend": ["ROE Change", "ROE Trend", "ROA Change", "Net Interest Margin Change"]},
    "Insurance": {"Rentabilidad": ["Current ROE", "Current ROA", "Estimated ROE"],
                  "Calidad de beneficios": [],
                  "Endeudamiento": ["Current Equity/Assets", "Current LT Debt/Assets"],
                  "Estabilidad": ["EPS CV Squared", "ROA Change Sq"],
                  "Márgenes": [],
                  "Change/Trend": ["ROE Change", "ROE Trend", "ROA Change"]},
}
EXTRA = {"Financial Services": {"Endeudamiento": ["Current Equity/Assets"]}}
FIN_EXTRA = ["Current Net Interest Margin", "Past Efficiency Ratio", "Current Equity/Assets",
             "Estimated Equity/Assets", "Net Interest Margin Change"]
BASE_ALL = list(dict.fromkeys([m for ms in PILLARS.values() for m in ms] + FIN_EXTRA + [FF]))


def _fill(x: pd.DataFrame, na: dict) -> pd.DataFrame:
    x = x.copy()
    for m in x.columns:
        h = na.get(m)
        if h == "replace_with_zero":
            x[m] = x[m].fillna(0.0)
        elif h == "replace_with_high":
            x[m] = x[m].fillna(x[m].max() if x[m].notna().any() else 0.0)
        elif h == "replace_with_low":
            x[m] = x[m].fillna(x[m].min() if x[m].notna().any() else 0.0)
    return x


def _build():
    panel = {}
    for f in glob.glob(os.path.join(HERE, os.environ.get("QPANEL", "qpanel"), "*.pkl")):
        d = pickle.load(open(f, "rb"))
        if d["period"] in PERIODS:
            panel[d["period"]] = d
    direction = {}
    for p in PERIODS:
        direction.update(panel[p]["direction"])
    direction.update({m: False for m in list(SQUARES) + list(RATIOS)})
    cols = [c for c in panel[PERIODS[-1]]["frames"][next(iter(panel[PERIODS[-1]]["frames"]))].columns
            if c not in ("security_id", "ticker")]
    ALL = list(dict.fromkeys(BASE_ALL + cols + list(RATIOS)))
    sign = np.array([1.0 if direction.get(m, True) else -1.0 for m in ALL])
    firms = pd.DataFrame({p: {i: len(z) for i, z in panel[p]["frames"].items()} for p in PERIODS}).fillna(0)
    industries = sorted(firms.index[firms.mean(axis=1) >= MIN_AVG_FIRMS])
    na_conv = panel[PERIODS[-1]]["na"]
    universe = {}
    for f in glob.glob(os.path.join(HERE, "qpanel_f1", "*.pkl")):
        d = pickle.load(open(f, "rb"))
        universe.update({(d["period"], i): set(z.ticker) for i, z in d["frames"].items()})
    raw, filled = {}, {}
    for p in PERIODS:
        for ind in industries:
            fr = panel[p]["frames"].get(ind)
            if fr is None or len(fr) < 4:
                continue
            x = fr.drop(columns=["security_id", "ticker"]).apply(pd.to_numeric, errors="coerce").astype(float)
            # Universe = the fase-1 universe (firms returned for the quality metric set); the extra fase-2
            # price/volatility columns must not add firms without fundamentals
            keep = fr.ticker.isin(universe[(p, ind)]).to_numpy()
            fr, x = fr[keep], x[keep]
            for name, (a, b) in SQUARES.items():
                d = x[a] if b is None else x[a] - x[b]
                x[name] = d * d
            for name, (a, b, sq) in RATIOS.items():
                r = x[a] / x[b].where(x[b] != 0)
                x[name] = r * r if sq else r
            x = x.reindex(columns=ALL)
            raw[(p, ind)] = (fr.ticker.to_numpy(), x)
            filled[(p, ind)] = _fill(x, na_conv)
    lam = {}
    for p in PERIODS:
        pooled = pd.concat([filled[k] for k in filled if k[0] == p])
        for m in ALL:
            v = pooled[m].dropna().to_numpy(float)
            if len(v) >= 30 and np.std(v) > 0:
                lo, hi = np.quantile(v, [0.005, 0.995])
                lam[(p, m)] = float(np.clip(yeojohnson_normmax(np.clip(v, lo, hi)), -3, 3))
    cells = []
    for (p, ind), (tick, x) in raw.items():
        df = filled[(p, ind)].copy()
        for m in ALL:
            s, lmb = df[m], lam.get((p, m))
            if lmb is not None and s.notna().any():
                df.loc[s.notna(), m] = yeojohnson(s[s.notna()].to_numpy(float), lmbda=lmb)
        w = df.clip(lower=df.quantile(.05), upper=df.quantile(.95), axis=1)
        mu, sd = w.mean(), w.std(ddof=0)
        z = ((w.fillna(mu) - mu) / sd.replace(0, np.nan)).fillna(0.0).to_numpy() * sign
        fwd = pd.Series(tick).map(panel[p]["fwd"]).to_numpy(float)
        ff_raw = x[FF].to_numpy(float)
        ok = ~np.isnan(ff_raw)
        ff = long_short(ff_raw[ok][:, None], fwd[ok])[0] if ok.sum() >= 4 else np.nan
        ff_ic = spearman(ff_raw[ok], fwd[ok])
        ic = np.array([spearman(z[:, j], fwd) for j in range(len(ALL))])
        cells.append(dict(p=p, ind=ind, z=z, fwd=fwd, na=x.isna().mean().to_numpy(), ff=ff, ff_ic=ff_ic, ic=ic))
    cells.sort(key=lambda c: (PERIODS.index(c["p"]), c["ind"]))
    return dict(cells=cells, industries=industries, sign=sign, lam=lam, all=ALL)


def long_short(score: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """score n x k, fwd n (NaN = 0): top half minus bottom half, equal weight (app convention)."""
    size = int(np.ceil(score.shape[0] / 2))
    order = np.argsort(-score, axis=0, kind="stable")
    r = np.nan_to_num(fwd)[order]
    return r[:size].mean(axis=0) - r[-size:].mean(axis=0)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 5 or np.std(a[ok]) == 0:
        return np.nan
    return float(pd.Series(a[ok]).rank().corr(pd.Series(b[ok]).rank()))


_CACHE = os.path.join(HERE, os.environ.get("QCELLS", "qcells.pkl"))
if os.path.exists(_CACHE):
    DATA = pickle.load(open(_CACHE, "rb"))
else:
    DATA = _build()
    pickle.dump(DATA, open(_CACHE, "wb"))
CELLS, INDUSTRIES, ALL = DATA["cells"], DATA["industries"], DATA["all"]
COL = {m: i for i, m in enumerate(ALL)}
