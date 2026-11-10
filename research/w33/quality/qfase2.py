"""Quality fase 2 (2026-11-06): saved profiles vs FF ROE, earnings-stability variants, tilted profiles and
theory industry profiles. Same panel/WF window as fase 1. Writes informe_fase2.md and fase2.pkl."""
import os
import sys
import json
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwf import (HERE, PERIODS, TEST, PILLARS, SUBSTITUTIONS, CELLS, INDUSTRIES, COL, NA_CUM, NA_MAX,  # noqa
                 NONFIN_NA, POOLED_IC, BY_P, walk_forward, fit_step, summarize, fixed_profile, long_short,
                 spearman, industry_sets)

pct = lambda x: "—" if pd.isna(x) else f"{x:+.1%}"  # noqa: E731
CFG = json.load(open(r"C:\Users\usuario\PycharmProjects\RankingFactores\modules\config\ranking_profiles.json",
                     encoding="utf-8"))
CFG = CFG.get("profiles", CFG)
# Quarters loaded on 2026-10-13/14 (BQL realignment): unseen by profiles designed before that date
NEW = [p for p in PERIODS if p <= "2019/12/30" or "2023/09/30" <= p <= "2024/03/30"]
GENERAL = ["Quality_08-09", "Quality_24-09", "Quality_28-09", "Quality_29-09", "Quality_07-10", "Quality_07-10_v2",
           "Quality_18-10"]
PRE_NEW = {"Quality_08-09", "Quality_24-09", "Quality_28-09", "Quality_29-09", "Quality_07-10"}
L = []


def row_cells(score_fn, inds=None):
    out = []
    for c in CELLS:
        if inds is not None and c["ind"] not in inds:
            continue
        s = score_fn(c)
        out.append(dict(p=c["p"], ind=c["ind"], r=long_short(s[:, None], c["fwd"])[0], ff=c["ff"],
                        ic=spearman(s, c["fwd"]), ff_ic=c["ff_ic"], inc=True))
    return pd.DataFrame(out)


def weights_vec(weights):
    vec = np.zeros(len(COL))
    missing = [m for m in weights if m not in COL]
    for m, w in weights.items():
        if m in COL:
            vec[COL[m]] = w
    return vec, missing


def ind_stats(df, ind, periods=TEST):
    d = df[(df.ind == ind) & df.p.isin(periods)].dropna(subset=["r", "ff"])
    if d.empty:
        return dict(r=np.nan, ff=np.nan, ex=np.nan, t=np.nan, q=np.nan, n=0)
    a = lambda s: (1 + s).prod() ** (4 / len(s)) - 1  # noqa: E731
    e = d.r - d.ff
    return dict(r=a(d.r), ff=a(d.ff), ex=a(d.r) - a(d.ff), t=e.mean() / e.std() * np.sqrt(len(e)), q=(e > 0).mean(),
                n=len(d))


def table(rows, head):
    L.extend(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])
    L.append("")


# ---------------------------------------------------------------- 0. Fase-1 main WF (reference)
MAIN_DF, MAIN_STEPS = walk_forward(PILLARS)
MAIN = summarize(MAIN_DF)
DEPLOY = fit_step(PILLARS, PERIODS)
L += ["# Quality fase 2: perfiles guardados, Estabilidad de Beneficios, pesos inclinados y perfiles por industria", "",
      "Mismo protocolo y ventana que la fase 1: WF de 28 trimestres (2019/03–2026/03), YJ → winsor → Z(0,1), "
      "frente a FF ROE dentro de cada industria, 20 industrias con ≥ 8 empresas. «Nuevos» = los 15 trimestres "
      "cargados el 13–14/10 (2017/03–2019/12 y 2023/09–2024/03), que no vio ningún perfil anterior a esa fecha.", "",
      f"Referencia fase 1 (principal WF): exceso {pct(MAIN['excess'])}, t {MAIN['t']:.2f}, "
      f"{MAIN['q_win']:.0%} trimestres, {MAIN['ind_win']}/20 industrias, ventanas 3a {MAIN['roll_win']}.", "",
      "## Conclusiones", "",
      "1. **Sí había un perfil guardado mejor en la ventana WF: Quality_24-09** (ND/EBITDA 80%, ROIC Change 10%, "
      "Total Assets Growth Change 10%): +2,7 pp/año sobre FF ROE, t 2,1, 16/20 industrias, 14/17 ventanas de 3 años. "
      "Pero es el mejor de 7, no es un perfil Quality (una métrica de deuda al 80%, rompe la regla de ≤ 25%) y en los "
      "15 trimestres que nunca vio pierde −1,5 pp (t −1,2). Quality_28-09/29-09 (idénticos) igual: +1,8 pp en WF y "
      "−4,0 pp (t −2,6) en los nuevos. Son sobreajuste.",
      "2. **El más honesto es Quality_07-10**: +1,4 pp en WF y +1,4 pp en los trimestres nuevos (t 0,9), con "
      "EPS Variability al 15%. Ningún perfil supera t ≈ 1 fuera de muestra real.",
      "3. **Estabilidad de Beneficios:** todas las variantes son, en la práctica, baja volatilidad (correlación "
      "+0,7–0,8 con el largo-corto de baja volatilidad de precio). Pierden en 2020, 2023 y 2025 por régimen y "
      "ganan en 2022. Cambiar la métrica no cambia el WF. Se mantiene el pilar (10%).",
      "4. **Más peso a Endeudamiento y Calidad de beneficios o a GP/A:** +1,4 a +1,8 pp con t 1,1–1,4, pero esos "
      "pesos se eligieron viendo los 36 trimestres; en los trimestres nuevos dan −1,5 a +0,7 pp. No hay mejora "
      "robusta frente a la fase 1.",
      "5. **Perfiles por industria por teoría** (Energy, HC Equipment, Consumer Services, Consumer Durables): no "
      "arreglan nada, siguen perdiendo contra FF ROE en las 4 (−3 a −8 pp). Lo que falla ahí es FF ROE ganando, no "
      "métricas mal elegidas; con los datos actuales no hay métrica sectorial que lo corrija (sección 5).",
      "6. **Recomendación:** mantener el general de la fase 1 en las 20 industrias (o Quality_07-10, equivalente y "
      "más honesto en los trimestres nuevos), presentar la ventaja como +1 pp/año frente a FF ROE no significativa, "
      "y no usar perfiles por industria salvo bancos y seguros por sustitución teórica.",
      "7. **Fragilidad:** quitar o añadir 5 empresas sin fundamentales del universo (de 17.092 observaciones) movía el "
      "exceso del principal de +1,0 a +1,2 pp. Diferencias de ±0,5 pp entre perfiles son ruido.", ""]

# ---------------------------------------------------------------- 1. Saved general profiles
L += ["## 1. Perfiles Quality guardados (pesos fijos, YJ+Z)", ""]
rows, saved = [], {}
for n in GENERAL:
    w = CFG[n]["nodes"]["Scoring"]["inputs"]
    vec, missing = weights_vec(w)
    df = row_cells(lambda c: c["z"] @ vec)
    saved[n] = df
    s, sn = summarize(df), summarize(df, NEW)
    full = summarize(df, PERIODS)
    rows.append([n, str(len(w)), pct(s["r"]), pct(s["excess"]), f"{s['t']:.2f}", f"{s['t_ic']:.2f}",
                 f"{s['q_win']:.0%}", f"{s['ind_win']}/{s['n_ind']}", s["roll_win"],
                 pct(sn["excess"]) + f" (t {sn['t']:.2f})" + (" ★" if n in PRE_NEW else ""),
                 pct(full["excess"]), ", ".join(missing) or "—"])
rows.append(["Principal fase 1 (WF anidado)", "—", pct(MAIN["r"]), pct(MAIN["excess"]), f"{MAIN['t']:.2f}",
             f"{MAIN['t_ic']:.2f}", f"{MAIN['q_win']:.0%}", f"{MAIN['ind_win']}/{MAIN['n_ind']}", MAIN["roll_win"],
             pct(summarize(MAIN_DF, [p for p in NEW if p in TEST])["excess"]) + " (solo WF)", "—", "—"])
table(rows, ["Perfil", "Nº métr.", "WF rent.", "WF exceso vs FF", "t rent.", "t IC", "% trim.", "Industrias",
             "Ventanas 3a", "Exceso trimestres nuevos", "Exceso 36q", "Métricas sin dato"])
best = max(GENERAL, key=lambda n: summarize(saved[n])["excess"])
L += [f"★ = perfil diseñado antes de cargar los trimestres nuevos: en esa columna es fuera de muestra de verdad.", "",
      f"**Lectura.** El mejor guardado en la ventana WF es **{best}**. Ojo: es el mejor de {len(GENERAL)} y casi todos "
      "se diseñaron mirando parte de estos trimestres, así que su exceso WF está inflado por selección; la comparación "
      "honesta es la columna de trimestres nuevos para los ★ y la t frente a la del principal.", ""]

# ---------------------------------------------------------------- 2. Earnings stability
STAB = ["EPS Variability", "EPS CV Squared", "EPS Var / Price", "ROA Change Sq", "EBIT Margin Change Sq"]
LOWVOL = row_cells(lambda c: c["z"][:, COL["Volatility 252 Calendar Days"]])
lv_q = LOWVOL.groupby("p").r.mean()
ff_q = LOWVOL.groupby("p").ff.mean()
L += ["## 2. Estabilidad de Beneficios", "",
      "Candidatas con base teórica (todas «menos es mejor»; signo comprobado en la DB: EPS Variability y Net Income "
      "Volatility tienen higher_is_better = 0; las ratios se construyen aquí con el mismo signo). NA: neutral "
      "(convención de la DB; NA ≈ 2%). **Net Income Volatility está en la DB pero vacía (100% NA en todos los "
      "trimestres)**, así que σ(ROA) ≈ NI Vol / Assets y NI CV² no se pueden probar; además su convención NA en la "
      "DB es replace_with_zero, que daría la mejor nota a quien no tiene dato: si se descarga, cambiar a neutral.", "",
      "- EPS Variability: desviación típica del EPS en $ (depende del precio por acción: penaliza acciones caras).",
      "- EPS CV²: (σ EPS / EPS)², escala libre; explota cuando el EPS ≈ 0.",
      "- EPS Var / Price: σ EPS / precio = volatilidad del earnings yield, escala libre y sin división por EPS ≈ 0.",
      "- ROA Change² y EBIT Margin Change²: variación anual al cuadrado (fase 1).",
      "- Earnings smoothness (Leuz: σ NI / σ CFO) y σ ROE/ROA a 5 años **no se pueden construir**: no hay serie de "
      "CFO ni de ROA histórico (5Y Average ROA está vacío). Hay que descargarlas (sección 5).", ""]
rows, stab_rows = [], {}
yrs = ["2020", "2022", "2023", "2025"]
for m in STAB:
    df = row_cells(lambda c: c["z"][:, COL[m]])
    stab_rows[m] = df
    s36 = summarize(df, PERIODS)
    ic = POOLED_IC[COL[m]] if COL[m] in POOLED_IC.columns else pd.Series(dtype=float)
    q = df.groupby("p").r.mean()
    by_y = q.groupby(q.index.str[:4]).mean() * 4
    rows.append([m, f"{NONFIN_NA[COL[m]].mean():.0%}", f"{ic.mean():+.3f} (t {ic.mean() / ic.std() * 6:.1f})",
                 pct(s36["r"]), pct(s36["excess"]), *[pct(by_y.get(y)) for y in yrs],
                 f"{q.corr(lv_q):+.2f}"])
lvy = lv_q.groupby(lv_q.index.str[:4]).mean() * 4
ffy = ff_q.groupby(ff_q.index.str[:4]).mean() * 4
rows.append(["(ref.) Baja volatilidad de precio 252d", "—", "—", pct(summarize(LOWVOL, PERIODS)["r"]), "—",
             *[pct(lvy.get(y)) for y in yrs], "1.00"])
rows.append(["(ref.) FF ROE", "—", "—", pct(summarize(LOWVOL, PERIODS)["ff"]), "—", *[pct(ffy.get(y)) for y in yrs],
             f"{ff_q.corr(lv_q):+.2f}"])
table(rows, ["Métrica sola", "NA no financ.", "IC agregado 36q", "Rent. 36q", "Exceso vs FF 36q", *yrs,
             "Corr. con baja vol."])
L += ["EPS Variability ($) es la única con IC claramente positivo, pero en parte mide el precio por acción "
      "(correlación dentro de industria de σ EPS con el precio +0,29); su versión sin escala (σ EPS / precio) tiene "
      "IC ≈ 0. En WF, cambiar la métrica del pilar no mueve el resultado (+0,1 a +1,1 pp). Se mantiene el pilar al "
      "10% con EPS CV² + ROA Change² (o EPS Variability si se prefiere la del perfil 07-10).", ""]

# WF with the stability pillar forced to each candidate (pillar kept, other pillars as fase 1)
rows, wf_stab = [], {}
for m in STAB:
    df, _ = walk_forward(dict(PILLARS, Estabilidad=[m]))
    s = summarize(df)
    wf_stab[m] = s
    rows.append([m, pct(s["excess"]), f"{s['t']:.2f}", f"{s['q_win']:.0%}", f"{s['ind_win']}/{s['n_ind']}",
                 s["roll_win"]])
table(rows, ["Estabilidad = (WF anidado, resto como fase 1)", "Exceso WF", "t", "% trim.", "Industrias",
             "Ventanas 3a"])
L += ["**Por qué resta en 2020, 2023 y 2025.** Las empresas de beneficio estable son, sobre todo, las de baja "
      "volatilidad y beta baja: la correlación trimestral del largo-corto de estabilidad con el de baja volatilidad de "
      "precio lo mide (última columna). En los tres años el mercado premió lo contrario: rebote de alta beta y "
      "cíclicas tras el COVID (2020 T2–T4), rally de 2023 concentrado en beneficios volátiles (tecnología, IA) y el "
      "«junk rally» de 2025 (RMW de Ken French −10%). Es un efecto régimen, no un fallo de la métrica: la estabilidad "
      "es un seguro que paga en caídas (2022) y cuesta en rebotes de riesgo.", ""]

# ---------------------------------------------------------------- 3. Tilted pillar weights
def spec_score(spec):
    """spec: {pillar: (weight, {metric: within-pillar weight})}; industry substitutions and NA rule applied."""
    def fn(c):
        na, sub = NA_CUM[(c["p"], c["ind"])], SUBSTITUTIONS.get(c["ind"], {})
        score = np.zeros(len(c["fwd"]))
        for pil, (w, mw) in spec.items():
            mw = {m: 1.0 for m in sub[pil]} if pil in sub else dict(mw)
            if c["ind"] == "Financial Services" and pil == "Endeudamiento":
                mw["Current Equity/Assets"] = np.mean(list(mw.values()))
            mw = {m: v for m, v in mw.items() if m in COL and na[COL[m]] <= NA_MAX}
            if mw:
                tot = sum(mw.values())
                score += w * sum(v / tot * c["z"][:, COL[m]] for m, v in mw.items())
        return score
    return fn


SEL = DEPLOY["sel"]
eq = lambda pil: {m: 1.0 for m in SEL[pil]}  # noqa: E731
rent_gpa = {m: (3.0 if "GP/" in m else 1.0) for m in SEL["Rentabilidad"]}  # GP/A = 50% of the pillar
TILTS = {
    "Fase 1 despliegue (25/15/30/10/10/10)": dict(zip(PILLARS, DEPLOY["w"])),
    "T1 Deuda 30 / Calidad benef. 25 / Rent. 15": dict(zip(PILLARS, [.15, .25, .30, .10, .10, .10])),
    "T2 Deuda 35 / Calidad benef. 25 / Rent. 10": dict(zip(PILLARS, [.10, .25, .35, .10, .10, .10])),
}
rows, tilt_res = [], {}
for name, w in TILTS.items():
    for gpa in (False, True):
        spec = {pil: (w[pil], rent_gpa if (gpa and pil == "Rentabilidad") else eq(pil)) for pil in PILLARS}
        df = row_cells(spec_score(spec))
        s, sn = summarize(df), summarize(df, NEW)
        key = name + (" + GP/A 50% de Rentabilidad" if gpa else "")
        tilt_res[key] = df
        rows.append([key, pct(s["excess"]), f"{s['t']:.2f}", f"{s['t_ic']:.2f}", f"{s['q_win']:.0%}",
                     f"{s['ind_win']}/{s['n_ind']}", s["roll_win"], pct(sn["excess"])])
L += ["## 3. Más peso en Endeudamiento y Calidad de beneficios, y GP/A dentro de Rentabilidad", "",
      "Pesos fijos dentro de la rejilla (10–35%, Change/Trend 10%), métricas = selección de despliegue de la fase 1. "
      "**Sesgo:** la inclinación se decidió viendo el diagnóstico por pilar de los 36 trimestres (fase 1), así que "
      "estas filas no son fuera de muestra puro aunque se midan en la ventana WF.", ""]
table(rows, ["Perfil", "Exceso WF", "t", "t IC", "% trim.", "Industrias", "Ventanas 3a", "Exceso trim. nuevos"])

# ---------------------------------------------------------------- 4. Industry profiles
W = dict(zip(PILLARS, DEPLOY["w"]))
gen = lambda pil: eq(pil)  # noqa: E731
THEORY = {
    "Energy": {
        "Rentabilidad": ["5Y Average ROIC", "Current ROIC", "Current ROA", "Current FCFF/Assets"],
        "Calidad de beneficios": ["Current CFO/Assets", "Current Accruals/Assets", "Current FCFF/Net Income"],
        "Endeudamiento": ["Current Net Debt / EBITDA", "Estimated Net Debt / EBITDA", "Current Net Debt/Assets",
                          "Current OCF/LT Debt"],
        "Márgenes": ["Current EBIT Margin", "Estimated EBIT Margin"],
        "Change/Trend": ["ROIC Change", "CFO/Assets Change", "EBIT Margin Change"]},
    "Health Care Equipment & Services": {
        "Rentabilidad": ["Current ROIC", "5Y Average ROIC", "Current ROA", "Estimated ROA"],
        "Márgenes": [],
        "Change/Trend": ["ROA Change", "ROIC Change", "CFO/Assets Change", "EBIT Margin Change",
                         "OCF/Net Income Change"]},
    "Consumer Services": {
        "Rentabilidad": ["Current ROA", "Estimated ROA", "Current ROIC", "5Y Average ROIC", "Current GP/Assets"],
        "Endeudamiento": ["Current Net Debt / EBITDA", "Estimated Net Debt / EBITDA", "Current Net Debt/OCF",
                          "Current OCF/LT Debt"],
        "Change/Trend": ["ROA Change", "ROIC Change", "CFO/Assets Change", "EBIT Margin Change", "GP/Assets Change"]},
    "Consumer Durables & Apparel": {
        "Rentabilidad": ["Current ROE", "Estimated ROE", "Current ROIC", "Current GP/Assets", "Estimated GP/Assets"],
        "Calidad de beneficios": ["Current Accruals/Assets", "Current CFO/Assets", "Estimated CFO/Assets"],
        "Márgenes": ["Current Gross Margin", "Estimated Gross Margin"]},
}
WHY = {
    "Energy": "Ciclo de materias primas: el margen bruto no compara E&P con refino; manda la rentabilidad del capital "
              "a lo largo del ciclo (5Y ROIC), la conversión a caja libre (FCFF) y la deuda frente a EBITDA.",
    "Health Care Equipment & Services": "Mezcla dispositivos (márgenes 25%+), aseguradoras de salud (≈5%) y "
                                        "distribuidores (≈1%, patrimonio negativo): márgenes, GP/A y ROE ordenan el "
                                        "modelo de negocio, no la calidad. Se usan ROIC/ROA y caja.",
    "Consumer Services": "Restaurantes, hoteles y franquicias con patrimonio negativo por recompras: ROE sin sentido; "
                         "deuda medida contra EBITDA/OCF (no contra activos, que el franquiciador apenas tiene).",
    "Consumer Durables & Apparel": "Marcas (poder de precio = margen bruto) y promotoras (inventario = devengos): más "
                                   "peso a margen bruto, GP/A y accruals.",
}
spec_for = {ind: {pil: (W[pil], {m: 1.0 for m in THEORY[ind].get(pil, SEL[pil])}) for pil in PILLARS}
            for ind in THEORY}
L += ["## 4. Perfiles por industria definidos por teoría", "",
      "Mismos pesos de pilar que el general (despliegue fase 1); solo cambian las métricas donde la teoría del sector "
      "lo pide. **Sesgo:** se eligen estas 4 industrias porque el general pierde ahí, así que la mejora media está "
      "sesgada a favor (regresión a la media).", ""]
rows, ind_res = [], {}
for ind, spec in spec_for.items():
    df = row_cells(spec_score(spec), {ind})
    ind_res[ind] = df
    a, g = ind_stats(df, ind), ind_stats(MAIN_DF, ind)
    q18 = ind_stats(saved["Quality_18-10"], ind)
    rows.append([ind, pct(a["r"]), pct(a["ff"]), pct(a["ex"]), f"{a['t']:.2f}", f"{a['q']:.0%}", pct(g["ex"]),
                 pct(q18["ex"])])
table(rows, ["Industria", "Perfil teoría WF", "FF ROE WF", "Exceso", "t", "% trim.", "Exceso general fase 1",
             "Exceso Quality_18-10"])
for ind in THEORY:
    L.append(f"- **{ind}.** {WHY[ind]} Métricas: " + "; ".join(
        f"{p}: {', '.join(THEORY[ind][p]) or 'no aplica'}" for p in THEORY[ind]) + ".")
L.append("")

# Saved industry profiles in their own industry
rows = []
for n, v in CFG.items():
    if not n.startswith("Quality") or "FF" in n:
        continue
    ind = next((i for i in INDUSTRIES if n.endswith("_" + i)), None)
    if ind is None:
        continue
    vec, missing = weights_vec(v["nodes"]["Scoring"]["inputs"])
    df = row_cells(lambda c: c["z"] @ vec, {ind})
    a, g = ind_stats(df, ind), ind_stats(MAIN_DF, ind)
    rows.append([n.replace("_" + ind, ""), ind, pct(a["ex"]), f"{a['t']:.2f}", pct(g["ex"]),
                 pct(ind_stats(df, ind, NEW)["ex"]), ", ".join(missing) or "—"])
L += ["### Perfiles por industria ya guardados, en su industria (pesos fijos, ventana WF)", "",
      "Casi todos se ajustaron por IC dentro de la industria con datos que incluyen parte de esta ventana: "
      "sesgo de sobreajuste (en la fase de octubre, sus WF fueron peores que el general).", ""]
table(rows, ["Perfil", "Industria", "Exceso vs FF WF", "t", "Exceso general fase 1", "Exceso trim. nuevos",
             "Métricas sin dato"])

# Combined: general + theory profiles in the 4 losing industries
comb = pd.concat([MAIN_DF[~MAIN_DF.ind.isin(THEORY)]] +
                 [ind_res[i][ind_res[i].p.isin(TEST)].assign(inc=True) for i in THEORY])
sc = summarize(comb)
L += [f"**General + 4 perfiles de teoría:** exceso WF {pct(sc['excess'])} (t {sc['t']:.2f}, {sc['q_win']:.0%} "
      f"trimestres, {sc['ind_win']}/{sc['n_ind']} industrias, ventanas 3a {sc['roll_win']}) frente a "
      f"{pct(MAIN['excess'])} (t {MAIN['t']:.2f}) del general solo. Con el sesgo de elección de industrias.", ""]

# ---------------------------------------------------------------- 5. Downloads
L += ["## 5. Métricas que habría que descargar", "",
      "- **Estabilidad (todas las industrias):** series de 20 trimestres de EPS, beneficio neto, CFO y ventas para "
      "σ(ROA) y σ(ROE) a 5 años, earnings smoothness (σ NI / σ CFO, Leuz) y estabilidad de márgenes; "
      "BDS «EPS volatility 5Y» / «Sales volatility» si existen en BQL.",
      "- **Bancos:** CET1 / Tier 1, morosidad (NPL/loans), provisiones/préstamos, coste del riesgo, loans/deposits.",
      "- **Seguros:** combined ratio, desarrollo de reservas, solvencia (RBC), ROE operativo.",
      "- **REITs:** FFO/AFFO, deuda neta/activos brutos, ocupación, cobertura de cargas fijas.",
      "- **Energy:** capex/OCF, reemplazo de reservas, coste de hallazgo y desarrollo, breakeven.",
      "- **Consumer Services:** deuda ajustada por arrendamientos / EBITDAR, cobertura de cargas fijas, ROIC ajustado "
      "por arrendamientos.",
      "- **Health Care Equipment & Services:** I+D/ventas, medical loss ratio (aseguradoras), rotación de inventario.",
      "- **Consumer Durables & Apparel:** rotación de inventario, cartera de pedidos (promotoras).", ""]
open(os.path.join(HERE, "informe_fase2.md"), "w", encoding="utf-8").write("\n".join(L))
pickle.dump(dict(saved=saved, stab=stab_rows, wf_stab=wf_stab, tilt=tilt_res, ind=ind_res, main=MAIN_DF),
            open(os.path.join(HERE, "fase2.pkl"), "wb"))
print("\n".join(L))
