"""datos_graficos_v4.json: perfiles definitivos (mejora/w33/final_profiles.py), test 2018/03-2026/03 (33 trim.),
anexo periodo completo 2017/03-2026/03 (37 trim.). Run from repo root."""
import os, sys, json, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); SCR = os.path.dirname(HERE)
W33 = os.path.join(SCR, "mejora", "w33"); sys.path.insert(0, W33)
import final_profiles as FP
from final_profiles import E
r1 = lambda x: None if x is None or pd.isna(x) else round(float(x) * 100, 1)
r2 = lambda x: None if x is None or pd.isna(x) else round(float(x), 2)
TEST0, SPLIT = "2018/03/30", "2021/09/30"


def risk(df, periods):
    d = df[df.p.isin(periods)].dropna(subset=["r", "bench"])
    pq, bq = d.groupby("p").r.mean(), d.groupby("p").bench.mean()
    dd = lambda s: float(((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min())
    return dict(sharpe=r2(pq.mean() * 4 / (pq.std() * 2)), sharpe_b=r2(bq.mean() * 4 / (bq.std() * 2)),
                vol_pct=r1(pq.std() * 2), vol_b_pct=r1(bq.std() * 2), dd_pct=r1(dd(pq)), dd_b_pct=r1(dd(bq)),
                corr=r2(pq.corr(bq))), pq, bq


def curve(pq, bq, start=True):
    lp = lb = 100.0
    pts = [dict(periodo="inicio", perfil=100.0, benchmark=100.0)] if start else []
    for p in pq.index:
        lp *= 1 + pq[p]; lb *= 1 + bq[p]
        pts.append(dict(periodo=p[:7], perfil=round(lp, 1), benchmark=round(lb, 1)))
    return pts


def block(fd, spec, pw, bench):
    df = FP.score_rows(fd, spec)
    P = list(fd.periods); T = [p for p in P if p >= TEST0]
    s = E.summarize(df, T); rk, pq, bq = risk(df, T)
    full = E.summarize(df, P); rkf, pqf, bqf = risk(df, P)
    h1 = E.summarize(df, [p for p in T if p < SPLIT]); h2 = E.summarize(df, [p for p in T if p >= SPLIT])
    exq = s["ex_q"]; sdq = exq.std()
    out = dict(benchmark=bench, ventana=f"{T[0][:7]}–{T[-1][:7]} ({len(T)} trim.)",
               kpis=dict(perfil_anual_pct=r1(s["r"]), benchmark_anual_pct=r1(s["bench"]), exceso_pp=r1(s["excess"]),
                         t=r2(s["t"]), t_ic=r2(s["t_ic"]), ic=round(float(s["ic"]), 4), ic_bench=round(float(s["bench_ic"]), 4),
                         trimestres_gana_pct=round(s["q_win"] * 100), ventanas_2a=s["roll"],
                         industrias=f"{s['ind_win']}/{s['n_ind']}", **rk,
                         acumulado_pct=r1((1 + pq).prod() - 1), acumulado_bench_pct=r1((1 + bq).prod() - 1),
                         exceso_2018_21=r1(h1["excess"]), exceso_2021_26=r1(h2["excess"]),
                         exceso_necesario_t2_pp=r1(2 * sdq * 4 / np.sqrt(len(exq))),
                         exceso_necesario_t3_pp=r1(3 * sdq * 4 / np.sqrt(len(exq)))),
               periodo_completo=dict(ventana=f"{P[0][:7]}–{P[-1][:7]} ({len(P)} trim.)", exceso_pp=r1(full["excess"]),
                                     t=r2(full["t"]), t_ic=r2(full["t_ic"]), perfil_anual_pct=r1(full["r"]),
                                     benchmark_anual_pct=r1(full["bench"]), ventanas_2a=full["roll"],
                                     industrias=f"{full['ind_win']}/{full['n_ind']}", **rkf))
    out["acumulado"] = curve(pq, bq)
    out["acumulado_completo"] = curve(pqf, bqf)
    out["exceso_trimestral_pp"] = [dict(periodo=p[:7], pp=r1(x)) for p, x in exq.items()]
    out["exceso_por_anio_pp"] = {k: r1(v * 4) for k, v in exq.groupby(exq.index.str[:4]).mean().items()}
    roll = exq.rolling(8).mean().dropna()
    idx = list(exq.index)
    out["ventanas_2a"] = [dict(desde=idx[idx.index(p) - 7][:7], hasta=p[:7], pp=r1(v * 4)) for p, v in roll.items()]
    dd_ = df[df.p.isin(T)].dropna(subset=["r", "bench"])
    pr, bb = E.ann_by_ind(dd_, "r"), E.ann_by_ind(dd_, "bench")
    out["industrias"] = [dict(industria=i, exceso_pp=r1(v), perfil_pct=r1(pr[i]), benchmark_pct=r1(bb[i]))
                         for i, v in s["by_ind"].sort_values(ascending=False).items()]
    contrib = {}
    for pil in pw:
        d2 = FP.score_rows(fd, spec, drop=pil)
        contrib[pil] = r1(s["excess"] - E.summarize(d2, T)["excess"])
    out["aportacion_pilar_pp"] = contrib
    return out, df


OUT = {"_meta": dict(version="v4", fecha="2026-11-09", test="2018/03–2026/03 (33 trimestres), sin entrenamiento",
                     anexo="periodo completo 2017/03–2026/03 (37 trimestres)", fuente="mejora/w33/final_profiles.py")}
fq, sq, wq = FP.quality(); OUT["quality"], dq = block(fq, sq, wq, "FF ROE")
print("Quality", OUT["quality"]["kpis"], flush=True)
fr, sr, wr = FP.reversion(); OUT["reversion"], dr = block(fr, sr, wr, "ST_Rev (1m)")
print("Reversion", OUT["reversion"]["kpis"], flush=True)
fv, sv, wv = FP.value(); OUT["value"], dv = block(fv, sv, wv, "FF B/M")
print("Value", OUT["value"]["kpis"], flush=True)

# Momentum entre industrias y dos capas (misma ventana)
MI = pickle.load(open(os.path.join(SCR, "mejora", "momentum_industrias", "series.pkl"), "rb"))
ls = MI["lin"]["ls"]["6-1m (benchmark)"]
T = [p for p in fr.periods if p >= TEST0]
ann = lambda s: (1 + s.dropna()).prod() ** (4 / s.notna().sum()) - 1
tst = lambda s: s.dropna().mean() / s.dropna().std() * np.sqrt(s.notna().sum())
s_t = ls.loc[[p for p in T if p in ls.index]]
OUT["mi_6_1m"] = dict(
    test=dict(ventana=f"{T[0][:7]}–{T[-1][:7]} ({len(s_t)} trim.)", anual_pct=r1(ann(s_t)), t=r2(tst(s_t)),
              trimestres_positivos_pct=round((s_t > 0).mean() * 100), vol_pct=r1(s_t.std() * 2),
              sharpe=r2(ann(s_t) / (s_t.std() * 2)),
              dd_pct=r1(((1 + s_t).cumprod() / (1 + s_t).cumprod().cummax() - 1).min())),
    periodo_completo=dict(ventana=f"{ls.index[0][:7]}–{ls.index[-1][:7]} ({len(ls)} trim.)", anual_pct=r1(ann(ls)),
                          t=r2(tst(ls)), trimestres_positivos_pct=round((ls > 0).mean() * 100)),
    por_anio_pct={k: r1(v) for k, v in s_t.groupby(s_t.index.str[:4]).apply(lambda x: (1 + x).prod() ** (4 / len(x)) - 1).items()})
lvl, pts = 100.0, [dict(periodo="inicio", perfil=100.0)]
for p, x in s_t.items():
    lvl *= 1 + x; pts.append(dict(periodo=p[:7], perfil=round(lvl, 1)))
OUT["mi_6_1m"]["acumulado"] = pts
d = dr.dropna(subset=["r", "bench"]); rev, strev = d.groupby("p").r.mean(), d.groupby("p").bench.mean()
two, twob = (ls / 2 + rev).loc[T].dropna(), (ls / 2 + strev).loc[T].dropna()
OUT["mi_dos_capas"] = dict(kpis=dict(anual_pct=r1(ann(two)), t=r2(tst(two)), sharpe=r2(ann(two) / (two.std() * 2)),
                                     trimestres_positivos_pct=round((two > 0).mean() * 100), bench_anual_pct=r1(ann(twob)),
                                     bench_t=r2(tst(twob)), diferencia_pp=r1(ann(two) - ann(twob)), t_diferencia=r2(tst(two - twob)),
                                     correlacion_capas=r2((ls / 2).loc[T].corr(rev.loc[T]))),
                           acumulado=curve(two, twob))
WR = pickle.load(open(os.path.join(W33, "results.pkl"), "rb"))["Momentum M2"]
m2 = WR["test_sum"]
OUT["anexo_m2"] = dict(exceso_pp=r1(m2["excess"]), t=r2(m2["t"]), t_ic=r2(m2["t_ic"]), ventanas_2a=m2["roll"],
                       industrias=f"{m2['ind_win']}/{m2['n_ind']}", exceso_2018_21=r1(WR["halves"]["2018/03-2021/06"]["excess"]),
                       exceso_2021_26=r1(WR["halves"]["2021/09-2026/03"]["excess"]),
                       industrias_detalle=[dict(industria=i, exceso_pp=r1(v)) for i, v in m2["by_ind"].sort_values(ascending=False).items()])
json.dump(OUT, open(os.path.join(HERE, "datos_graficos_v4.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pickle.dump(dict(q=dq, v=dv, r=dr), open(os.path.join(W33, "final_rows.pkl"), "wb"))
print("MI", OUT["mi_6_1m"]["test"], "\n2 capas", OUT["mi_dos_capas"]["kpis"])
print("M2", {k: v for k, v in OUT["anexo_m2"].items() if k != "industrias_detalle"})
for f in ("quality", "value", "reversion"):
    print(f, "completo", OUT[f]["periodo_completo"], "pilares", OUT[f]["aportacion_pilar_pp"])
