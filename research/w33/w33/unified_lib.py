"""Regla única (pre-registrada 2026-11-09): 2 métricas canónicas por pilar por teoría (sin IC), peso igual dentro,
pilares iguales, NA por trimestre (celda > 40% sale). C/T = pilar entero en Quality y Value. Test 2018/03-2026/03.
Diagnóstico de Value: por qué parecía restar C/T (V4 incluía DY 1/6) y cada métrica C/T sola."""
import os, sys
import numpy as np, pandas as pd
os.environ.update(W33="1", QPANEL="qpanel_v33", QCELLS="qcells_v33.pkl", VPANEL="rpanel_v33")
HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(M, "tt"))
import engine as E
from variants_risk import risk
CT = "Change/Trend"


def evaluate(fd, pw, sets):
    rows = []
    for c in fd.cells:
        v = np.zeros(len(fd.col))
        for pil, wp in pw.items():
            ms = fd.subs[c["ind"]][pil] if pil in fd.subs.get(c["ind"], {}) else \
                list(sets[pil]) + fd.extra.get(c["ind"], {}).get(pil, [])
            ms = [m for m in ms if c["na"][fd.col[m]] <= 0.40]
            for m in ms:
                v[fd.col[m]] += wp / len(ms)
        s = c["z"] @ v
        rows.append(dict(p=c["p"], ind=c["ind"], r=E.long_short(s, c["fwd"]), bench=c["bench"],
                         ic=E.spearman(s, c["fwd"]), bench_ic=c["bench_ic"]))
    return pd.DataFrame(rows)


def line(name, fd, d):
    T = fd.periods[4:]
    s, r = E.summarize(d, T), risk(d, T)
    e = E.summarize(d, [p for p in T if p < "2021/09/30"]); l = E.summarize(d, [p for p in T if p >= "2021/09/30"])
    print(f"{name:44}{s['excess']*100:+7.2f}{s['t']:6.2f}{s['t_ic']:6.2f}{s['q_win']:6.0%}{s['roll']:>7}"
          f"{s['ind_win']:>4}/{s['n_ind']:<2}{e['excess']*100:+7.2f}{l['excess']*100:+7.2f}{r['sharpe']:7.2f}{r['sharpe_b']:6.2f}{r['corr']:6.2f}", flush=True)


HDR = f"{'':44}{'exceso':>7}{'t':>6}{'tIC':>6}{'%trim':>6}{'2a':>7}{'ind':>7}{'18-21':>7}{'21-26':>7}{'Sharpe':>7}{'SR_b':>6}{'corr':>6}"
