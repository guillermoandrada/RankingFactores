"""Value, convención FF (pre-registrado 2026-11-09). Pilares teoría sin DY (35/25/15/10/10 renorm.), C/T entero,
NA por trimestre. Tratamientos: base (múltiplos, NA P/E y EV/EBITDA = caro), Y (yields 1/x), N (NA neutral), YN.
Conjuntos de Beneficios: cur+est P/E, solo Est P/E."""
import os, sys
import numpy as np, pandas as pd
os.environ.update(W33="1", VPANEL="rpanel_v33")
HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(M, "tt")); sys.path.insert(0, os.path.join(M, "value"))
import vlib as L
import engine as E
from unified_lib import evaluate, line, HDR, CT

MULT = ["Current P/E Ratio", "Estimated P/E Ratio", "Current EV / EBITDA", "Estimated EV / EBITDA", "Current P/B Ratio",
        "Estimated P/B Ratio", "Current P/S Ratio", "Estimated P/S Ratio", "Current Price to Tangible Book"]
CTM = ["P/E Change", "P/E Trend", "P/B Change", "P/B Trend", "P/S Change", "P/S Trend", "EV/EBITDA Change",
       "EV/EBITDA Trend", "FCF Yield Change", "FCF Yield Trend"]
NEUTRAL = ["Current P/E Ratio", "Estimated P/E Ratio", "Current EV / EBITDA", "Estimated EV / EBITDA"]
Y = lambda m: "Y:" + m
for p in L.PERIODS:
    for fr in L.PANEL[p]["frames"].values():
        for m in MULT:
            x = pd.to_numeric(fr[m], errors="coerce")
            fr[Y(m)] = 1.0 / x.where(x != 0)
for m in MULT:
    L.DIRECTION[Y(m)] = True
    L.DIRECTION["N:" + m] = L.DIRECTION[m]
    L.DIRECTION["YN:" + m] = True
    L.NA_MAP[Y(m)] = "replace_with_low" if L.NA_MAP.get(m) == "replace_with_high" else L.NA_MAP.get(m)
    L.NA_MAP["N:" + m] = None if m in NEUTRAL else L.NA_MAP.get(m)
    L.NA_MAP["YN:" + m] = None if m in NEUTRAL else L.NA_MAP[Y(m)]
for p in L.PERIODS:
    for fr in L.PANEL[p]["frames"].values():
        for m in MULT:
            fr["N:" + m] = fr[m]
            fr["YN:" + m] = fr[Y(m)]
cols = MULT + [Y(m) for m in MULT] + ["N:" + m for m in MULT] + ["YN:" + m for m in MULT] + CTM
cells, industries = L.build_cells(cols, min_firms=8)


class FD:  # minimal FactorData for unified_lib.evaluate / line
    pass


fd = FD()
fd.col = {m: i for i, m in enumerate(cols)}
fd.periods = L.PERIODS
fd.cells = [dict(p=c["p"], ind=c["ind"], z=c["z"], fwd=c["fwd"], na=c["na"].mean(axis=0), bench=c["ff"],
                 bench_ic=c["ff_ic"]) for c in cells]
fd.extra = {}
