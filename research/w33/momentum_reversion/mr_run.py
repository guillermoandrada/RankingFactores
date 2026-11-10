"""Momentum and Reversion under the common protocol. Writes series.pkl and results tables (mr_results.pkl).

Pre-registered main variants (fixed before looking at results):
- Momentum: 5 pillars (precio incl. residual momentum, revisiones, trayectoria, sentimiento, volumen); the
  risk-adjusted pillar is out because volatility will get its own profile.
- Reversion: the 4 pillars of Reversion_04-11 with the new method.
"""
import os
import json
import pickle

import numpy as np
import pandas as pd

import mr_lib as L

REPO = r"C:\Users\usuario\PycharmProjects\RankingFactores"
CFG = json.load(open(os.path.join(REPO, "modules", "config", "ranking_profiles.json"), encoding="utf-8"))["profiles"]

PRECIO = ["12-1m Total Return", "6-1m Total Return", "12-6m Total Return"]
RESID = ["Residual 12-1m Return", "Residual 6-1m Return"]
VOLADJ = ["Risk-Adjusted 12-1m Return", "Risk-Adjusted 6-1m Return"]
REVIS = ["Estimated EPS Revision", "Estimated EPS Revision / Price", "Buy Rec % Change", "Sell Rec % Change",
         "Net Buy Rec % Change", "Best Analyst Rating Change"]
TRAY = ["Price / 52w High", "Price / 200d MA", "Price / 50d MA", "MACD 20/50 (SMA)", "MACD 50/200 (SMA)",
        "Information Discreteness"]
SENT = ["Buy Rec %", "Sell Rec %", "Net Buy Rec %", "Best Analyst Rating"]
VOLUM = ["Volume 20/60 Calendar Days", "Turnover 60 Calendar Days"]
REVERS = ["1m Return Reversal", "RSI 14 Reversal", "Price / 20d MA Reversal"]
REC = ["Buy Rec % Change", "Sell Rec % Change", "Net Buy Rec % Change", "Best Analyst Rating Change"]

MOM_B, REV_B = "12-1m Total Return", "1m Return Reversal"
VARIANTS = {
    "Momentum": dict(bench=MOM_B, fixed="Momentum_03-11", main="M2 principal: 5 pilares, precio con residual", runs={
        "M1 6 pilares con ajustado por volatilidad": {"Precio": PRECIO + RESID, "Ajustado por volatilidad": VOLADJ,
                                                      "Revisiones": REVIS, "Trayectoria": TRAY,
                                                      "Sentimiento": SENT, "Volumen": VOLUM},
        "M2 principal: 5 pilares, precio con residual": {"Precio": PRECIO + RESID, "Revisiones": REVIS,
                                                         "Trayectoria": TRAY, "Sentimiento": SENT, "Volumen": VOLUM},
        "M3 5 pilares, precio sin residual": {"Precio": PRECIO, "Revisiones": REVIS, "Trayectoria": TRAY,
                                              "Sentimiento": SENT, "Volumen": VOLUM},
        "M4 = M2 con pilares iguales": ({"Precio": PRECIO + RESID, "Revisiones": REVIS, "Trayectoria": TRAY,
                                         "Sentimiento": SENT, "Volumen": VOLUM}, dict(mode="equal")),
        "M5 = M2 con filtro de pilar (t IC >= 1)": ({"Precio": PRECIO + RESID, "Revisiones": REVIS,
                                                     "Trayectoria": TRAY, "Sentimiento": SENT, "Volumen": VOLUM},
                                                    dict(pillar_t=1.0)),
    }),
    "Reversion": dict(bench=REV_B, fixed="Reversion_04-11", main="R1 principal: 4 pilares", runs={
        "R1 principal: 4 pilares": {"Reversión": REVERS, "Revisiones": REC, "Sentimiento": SENT, "Volumen": VOLUM},
        "R2 sin pilar de revisiones": {"Reversión": REVERS, "Sentimiento": SENT, "Volumen": VOLUM},
        "R3 = R1 con pilares iguales": ({"Reversión": REVERS, "Revisiones": REC, "Sentimiento": SENT,
                                         "Volumen": VOLUM}, dict(mode="equal")),
        "R4 = R1 con filtro de pilar (t IC >= 1)": ({"Reversión": REVERS, "Revisiones": REC, "Sentimiento": SENT,
                                                     "Volumen": VOLUM}, dict(pillar_t=1.0)),
        "R5 = R2 con pilares iguales": ({"Reversión": REVERS, "Sentimiento": SENT, "Volumen": VOLUM},
                                        dict(mode="equal")),
    }),
}


def test_only(s):
    return s[s.index.get_level_values("period").isin(L.TEST)
             & s.index.get_level_values("industry").isin(L.INDUSTRIES)]


def main():
    out = {"INDUSTRIES": L.INDUSTRIES, "ALL_INDUSTRIES": L.ALL_INDUSTRIES, "TEST": L.TEST, "PERIODS": L.PERIODS}
    for factor, spec in VARIANTS.items():
        res = {}
        fixed_r, fixed_ic = L.weights_series(CFG[spec["fixed"]]["nodes"]["Scoring"]["inputs"])
        bench = L.metric_series(spec["bench"])
        bench_ic = L.IC[spec["bench"]]
        res[f"{spec['fixed']} (pesos actuales, YJ+Z)"] = dict(
            wf=L.summary(test_only(fixed_r), test_only(bench), test_only(fixed_ic), test_only(bench_ic)))
        for name, run in spec["runs"].items():
            pillars, opts = run if isinstance(run, tuple) else (run, {})
            prof = L.Profile(name, pillars, spec["bench"], **opts)
            wf, fits = prof.walk_forward()
            final = prof.fit(len(L.PERIODS))
            full_r, full_ic = L.weights_series(prof.metric_weights(final[0], final[1]))
            uni = lambda s: s[s.index.get_level_values("industry").isin(L.INDUSTRIES)]
            res[name] = dict(
                pillars=list(pillars), opts=opts, wf_frame=wf, fits=fits, final=final,
                wf=L.summary(wf.prof, wf.bench, wf.ic, wf.bench_ic),
                wf_rule=L.summary(wf.rule, wf.bench),
                full=L.summary(uni(full_r), uni(bench), uni(full_ic), uni(bench_ic)),
                final_weights=prof.metric_weights(final[0], final[1]))
            print(factor, name, {k: round(v, 4) for k, v in res[name]["wf"].items()}, flush=True)
            print("   rule", {k: round(v, 4) for k, v in res[name]["wf_rule"].items()}, flush=True)
        out[factor] = dict(spec=spec, res=res, bench=bench, bench_ic=bench_ic)
    pickle.dump(out, open(os.path.join(L.HERE, "series.pkl"), "wb"))


if __name__ == "__main__":
    main()
