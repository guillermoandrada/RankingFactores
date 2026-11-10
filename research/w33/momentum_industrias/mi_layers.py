"""Two layers: industry selection (gross 1 = L/S / 2) + within-industry Reversion (gross 1). WF quarters.

Reversion source: mejora/momentum_reversion/series.pkl if it exposes a quarterly WF Reversion series, else Reversion_04-11 WF.
"""
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mi_lib import HERE, OLD, ann, t_stat  # noqa: E402

R = pickle.load(open(os.path.join(HERE, "series.pkl"), "rb"))
TEST, B = R["TEST"], "6-1m (benchmark)"
old = pickle.load(open(os.path.join(OLD, "rev_results.pkl"), "rb"))["DET"]
q = old.groupby(["period", "profile"]).ret.mean().unstack()
REV = {"Reversion_04-11 (WF)": q["Walk-forward"], "FF ST_Rev (1m) dentro": q["FF ST_Rev (1m)"]}
new_path = os.path.join(os.path.dirname(HERE), "momentum_reversion", "series.pkl")
if os.path.exists(new_path):
    frame = pickle.load(open(new_path, "rb"))["Reversion"]["res"]["R2 sin pilar de revisiones"]["wf_frame"]
    REV["Reversion R2 mejorada (WF, regla de industrias)"] = frame["rule"].groupby(level="period").mean()
    REV["ST_Rev (pipeline agente D)"] = frame["bench"].groupby(level="period").mean()

IND = {"6-1m simple": R["lin"]["ls"][B] / 2, "Perfil YJ 6-1m 65 + sentimiento 35": R["yj"]["ls"]["6-1m + sentimiento 35%"] / 2}
rows, L = {}, {}
for i, si in IND.items():
    for r, sr in REV.items():
        s = (si + sr).loc[TEST].dropna()
        L[f"{i} + {r}"] = s
        rows[f"{i} + {r}"] = dict(n=len(s), ann=ann(s), t=t_stat(s), vol=s.std() * 2, sharpe=ann(s) / (s.std() * 2),
                                 hit=(s > 0).mean(), corr_capas=si.loc[TEST].corr(sr.loc[TEST]))
for name, s in {**IND, **REV}.items():
    s = s.loc[TEST].dropna()
    rows[f"Solo {name}"] = dict(n=len(s), ann=ann(s), t=t_stat(s), vol=s.std() * 2, sharpe=ann(s) / (s.std() * 2), hit=(s > 0).mean())
SUMM = pd.DataFrame(rows).T
L = pd.DataFrame(L)
diffs = {}
for r in REV:
    if not r.startswith(("FF ST_Rev", "ST_Rev")):
        ref = "ST_Rev (pipeline agente D)" if "R2" in r else "FF ST_Rev (1m) dentro"
        d = L[f"6-1m simple + {r}"] - L[f"6-1m simple + {ref}"]
        diffs[f"{r} − ST_Rev (con 6-1m simple)"] = (d.mean() * 4, t_stat(d))
pickle.dump(dict(SUMM=SUMM, L=L, diffs=diffs, REV_USED=list(REV)), open(os.path.join(HERE, "layers.pkl"), "wb"))
pd.set_option("display.width", 250)
print(SUMM.round(3).to_string())
print(diffs)
