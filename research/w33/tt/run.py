"""Builds the four FactorData objects from the existing mejora/* panels and runs the common engine.
Saves results.pkl (no weights or variants are tuned here: see spec.md and implementacion.md)."""
import os
import sys
import pickle
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEJORA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from engine import FactorData, run  # noqa: E402


def quality():
    sys.path.insert(0, os.path.join(MEJORA, "quality"))
    import qcore as Q
    cells = [dict(p=c["p"], ind=c["ind"], z=c["z"], fwd=c["fwd"], na=np.asarray(c["na"], float),
                  bench=c["ff"], bench_ic=c["ff_ic"]) for c in Q.CELLS]
    canon = {"Current ROE", "Estimated ROE", "Current ROA", "Estimated ROA", "Current GP/Assets",
             "Estimated GP/Assets", "Current CFO/Assets", "Estimated CFO/Assets", "Current Net Debt / EBITDA",
             "Estimated Net Debt / EBITDA", "EPS CV Squared", "Current EBIT Margin", "Estimated EBIT Margin"}
    pw = {"Rentabilidad": .30, "Calidad de beneficios": .20, "Endeudamiento": .20, "Estabilidad": .10,
          "Márgenes": .10, "Change/Trend": .10}
    return FactorData("Quality", Q.PERIODS, cells, Q.COL, Q.PILLARS, pw, canon, "FF ROE (Current ROE)",
                      subs=Q.SUBSTITUTIONS, extra=Q.EXTRA)


def value():
    os.environ.setdefault("VPANEL", "rpanel_f2")
    vdir = os.path.join(MEJORA, "value")
    sys.path.insert(0, vdir)
    cwd = os.getcwd()
    os.chdir(vdir)
    import vwf as V
    os.chdir(cwd)
    cells = [dict(p=c["p"], ind=c["ind"], z=c["z"], fwd=c["fwd"], na=np.asarray(c["na"], float).mean(axis=0),
                  bench=c["ff"], bench_ic=c["ff_ic"]) for c in V.CELLS]
    pillars = dict(V.BASE_SETS)
    pw = {"Beneficios": .35, "Caja": .25, "Valor contable": .15, "Ventas": .10, "Dividendo": .05,
          "Change/Trend": .10}
    canon = {"Estimated P/E Ratio", "Current EV / EBITDA", "Estimated EV / EBITDA", "Current P/B Ratio",
             "Estimated P/B Ratio", "Current P/S Ratio", "Estimated P/S Ratio", "Current Dividend Yield"}
    book = {p: list(V.BOOK_SUB) for p in ("Caja", "Ventas", "Valor contable")}
    return FactorData("Value", V.PERIODS, cells, V.COL, pillars, pw, canon, "FF B/M (Current Book to Market)",
                      subs={"Banks": book, "Insurance": book})


def momentum_reversion():
    sys.path.insert(0, os.path.join(MEJORA, "momentum_reversion"))
    import mr_lib as L
    import mr_run as R

    def cells_for(bench):
        b, bic = L.metric_series(bench), L.IC[bench]
        return [dict(p=c["p"], ind=c["ind"], z=c["z"], fwd=c["fwd"], na=np.asarray(c["na"], float),
                     bench=b.iloc[i], bench_ic=bic.iloc[i])
                for i, c in enumerate(L.CELLS) if c["ind"] in L.INDUSTRIES]

    mom_p = {"Precio": R.PRECIO + R.RESID, "Revisiones": R.REVIS, "Trayectoria": R.TRAY, "Sentimiento": R.SENT,
             "Volumen": R.VOLUM}
    mom_w = {"Precio": .25 / .85, "Revisiones": .25 / .85, "Trayectoria": .15 / .85, "Sentimiento": .10 / .85,
             "Volumen": .10 / .85}
    rev_p = {"Reversión": R.REVERS, "Sentimiento": R.SENT, "Volumen": R.VOLUM}
    rev_w = {"Reversión": .35 / .75, "Sentimiento": .25 / .75, "Volumen": .15 / .75}
    mom = FactorData("Momentum M2", L.PERIODS, cells_for(R.MOM_B), L.COL, mom_p, mom_w,
                     {"12-1m Total Return", "6-1m Total Return", "Estimated EPS Revision", "Best Analyst Rating"},
                     "12-1m (UMD)")
    rev = FactorData("Reversion", L.PERIODS, cells_for(R.REV_B), L.COL, rev_p, rev_w, {"1m Return Reversal"},
                     "ST_Rev (1m reversal)")
    return mom, rev


if __name__ == "__main__":
    t0 = time.time()
    out = {}
    factors = [quality(), value(), *momentum_reversion()]
    print(f"panels loaded [{time.time() - t0:.0f}s]", flush=True)
    for fd in factors:
        res = run(fd)
        out[fd.name] = dict(fd=dict(name=fd.name, periods=fd.periods, train=fd.train, test=fd.test,
                                    pillars=fd.pillars, pillar_w=fd.pillar_w, canonical=sorted(fd.canonical),
                                    bench=fd.bench_name, subs=fd.subs), **res)
        print(fd.name, f"train {fd.train[0]}-{fd.train[-1]} ({len(fd.train)}), test {fd.test[0]}-{fd.test[-1]} "
              f"({len(fd.test)}) [{time.time() - t0:.0f}s]", flush=True)
    pickle.dump(out, open(os.path.join(HERE, "results.pkl"), "wb"))
