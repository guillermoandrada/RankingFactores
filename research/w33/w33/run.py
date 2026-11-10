"""Ventana ampliada (2026-11-09): 1er año (2017/03-2017/12) solo para el filtro de NA; test 2018/03-2026/03.
Variante A (pesos iguales dentro de pilares por teoría). Mismo motor que mejora/tt. Paneles regenerados de la DB.
Nada se ajusta tras ver resultados. Run from repo root."""
import os
import sys
import pickle
import time

import numpy as np
import pandas as pd

os.environ.update(W33="1", QPANEL="qpanel_v33", QCELLS="qcells_v33.pkl", VPANEL="rpanel_v33")
HERE = os.path.dirname(os.path.abspath(__file__))
MEJORA = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MEJORA, "tt"))
import engine as E  # noqa: E402
import run as TT  # noqa: E402  (FactorData builders of the 18/18 test)

E.TRAIN_N = 4


def pillar_alone(fd, within, st):
    """Excess vs benchmark of each pillar on its own (same industry vectors restricted to one pillar)."""
    out = {}
    for pil in fd.pillar_w:
        sub = E.FactorData(fd.name, fd.periods, fd.cells, fd.col, fd.pillars, {pil: 1.0}, fd.canonical,
                           fd.bench_name, subs=fd.subs, extra=fd.extra)
        df = E.evaluate(sub, within, st)
        s = E.summarize(df, fd.test)
        out[pil] = dict(excess=s["excess"], t=s["t"], q_win=s["q_win"])
    return out


def halves(df, fd):
    """Excess in the old training span (2018/03-2021/06) and the old test span (2021/09-2026/03)."""
    old_test = [p for p in fd.periods if p >= "2021/09/30"]
    old_train = [p for p in fd.test if p < "2021/09/30"]
    return {"2018/03-2021/06": E.summarize(df, old_train), "2021/09-2026/03": E.summarize(df, old_test)}


if __name__ == "__main__":
    t0 = time.time()
    out = {}
    for fd in [TT.quality(), TT.value(), *TT.momentum_reversion()]:
        st = E.train_stats(fd)
        within = E.within_weights(fd, st)["A"]
        df = E.evaluate(fd, within, st)
        dropped_glob = sorted(m for ms in fd.pillars.values() for m in ms if st["na_glob"][m] > E.NA_MAX)
        out[fd.name] = dict(periods=fd.periods, data=fd.train, test=fd.test, bench=fd.bench_name, df=df,
                            test_sum=E.summarize(df, fd.test), halves=halves(df, fd),
                            pillars=pillar_alone(fd, within, st), dropped=dropped_glob,
                            na_glob=st["na_glob"], na_ind=st["na_ind"],
                            weights={pil: (fd.pillar_w[pil] * w).round(4) for pil, w in within.items()})
        s = out[fd.name]["test_sum"]
        print(f"{fd.name}: datos {fd.train[0]}-{fd.train[-1]} ({len(fd.train)}), test {fd.test[0]}-{fd.test[-1]} "
              f"({len(fd.test)}) exceso {s['excess']*100:+.2f} t {s['t']:.2f} [{time.time()-t0:.0f}s]", flush=True)
    pickle.dump(out, open(os.path.join(HERE, "results.pkl"), "wb"))
