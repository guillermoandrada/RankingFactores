"""Perfiles definitivos (cerrados 2026-11-09). Método único: pilares y pesos por teoría, 2 métricas canónicas por
pilar (actual + estimada), peso igual dentro del pilar, NA por trimestre (celda > 40% NA sale ese trimestre),
pilares que no aplican -> peso a prorrata, YJ -> winsor 5/95 -> Z dentro de la industria, long-short mitad alta - baja.
Value en yields (1/múltiplo). Sustituciones por teoría: financieras (Quality y Value) y cíclicas (Value).
Cada evaluador devuelve un DataFrame por celda (p, ind, r, bench, ic, bench_ic) en los 37 trimestres."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.dirname(HERE)
os.environ.update(W33="1", QPANEL="qpanel_v33", QCELLS="qcells_v33.pkl", VPANEL="rpanel_v33")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(M, "tt"))
import engine as E  # noqa: E402
CT = "Change/Trend"
NA_MAX = 0.40


def score_rows(fd, spec, drop=None):
    """spec(ind) -> (pillar weights, pillar sets). drop = pillar left out (leave-one-out contribution)."""
    rows = []
    for c in fd.cells:
        w, sets = spec(c["ind"])
        w = {k: v for k, v in w.items() if k != drop}
        tot = sum(w.values())
        v = np.zeros(len(fd.col))
        for pil, wp in w.items():
            ms = [m for m in sets.get(pil, []) if c["na"][fd.col[m]] <= NA_MAX]
            for m in ms:
                v[fd.col[m]] += wp / tot / len(ms)
        s = c["z"] @ v
        rows.append(dict(p=c["p"], ind=c["ind"], r=E.long_short(s, c["fwd"]), bench=c["bench"],
                         ic=E.spearman(s, c["fwd"]), bench_ic=c["bench_ic"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Quality
Q_W = {"Rentabilidad": .30, "Calidad de beneficios": .20, "Endeudamiento": .20, "Estabilidad": .10, "Márgenes": .10, CT: .10}


def quality():
    import run as TT
    fd = TT.quality()
    sets = {"Rentabilidad": ["Current ROE", "Current GP/Assets"],
            "Calidad de beneficios": ["Current CFO/Assets", "Estimated CFO/Assets"],
            "Endeudamiento": ["Current Net Debt / EBITDA", "Estimated Net Debt / EBITDA"],
            "Estabilidad": ["EPS CV Squared", "ROA Change Sq"],
            "Márgenes": ["Current EBIT Margin", "Estimated EBIT Margin"], CT: list(fd.pillars[CT])}

    def spec(ind):
        s = dict(sets)
        for pil, ms in fd.subs.get(ind, {}).items():
            s[pil] = list(ms)
        for pil, ms in fd.extra.get(ind, {}).items():
            s[pil] = s[pil] + list(ms)
        return Q_W, s
    return fd, spec, Q_W


# ------------------------------------------------------------------ Reversión
def reversion():
    import run as TT
    fd = TT.momentum_reversion()[1]
    return fd, (lambda ind: (fd.pillar_w, fd.pillars)), fd.pillar_w


# ------------------------------------------------------------------ Value
V_W = {"Beneficios": .35, "Caja": .25, "Valor contable": .15, "Ventas": .10, CT: .10}
FIN = ("Banks", "Insurance")
CYC = ("Materials", "Energy", "Transportation", "Capital Goods")


def value():
    from value_ff_lib2 import fd, Y, CTM
    fd.subs, fd.extra = {}, {}
    gen = {"Beneficios": [Y("Estimated P/E Ratio")], "Caja": [Y("Current EV / EBITDA"), Y("Estimated EV / EBITDA")],
           "Valor contable": [Y("Current P/B Ratio"), Y("Estimated P/B Ratio")],
           "Ventas": [Y("Current P/S Ratio"), Y("Estimated P/S Ratio")], CT: list(CTM)}

    def spec(ind):
        if ind in FIN:
            return {"Valor contable": V_W["Valor contable"]}, gen
        if ind in CYC:
            # Beneficios y Caja no aplican: su 60% va a Valor contable y Ventas a prorrata 15:10; C/T sigue en 10%
            return {"Valor contable": .15 + .60 * .6, "Ventas": .10 + .60 * .4, CT: .10}, gen
        return V_W, gen
    fd.periods = list(fd.periods)
    return fd, spec, V_W
