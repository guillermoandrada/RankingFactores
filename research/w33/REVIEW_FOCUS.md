# Foco de la revisión (research/w33)

Estos scripts son copias del trabajo de investigación hecho en un scratchpad. Las rutas absolutas que aparecen dentro (`OLD`, `S`, `M`) apuntan a ese scratchpad. Los paneles `.pkl` no se incluyen porque son datos.

## Mapa de ficheros
- `w33/final_profiles.py`: perfiles definitivos de Quality, Value y Reversión. Contiene `score_rows` y los `spec` por industria.
- `w33/value_ff_lib2.py`: panel de Value en yields (1/múltiplo) y tratamiento de NA.
- `tt/engine.py`: `long_short`, IC, `summarize` y t pareada.
- `tt/run.py`: carga los paneles de cada factor.
- `quality/qcore.py`, `value/vlib.py` y `momentum_reversion/mr_lib.py`:
  - celdas industria×trimestre;
  - Yeo-Johnson → winsor 5/95 → z;
  - retorno forward a 3m;
  - benchmark FF.
- `quality/qpanel.py`, `value/build_panel.py` y `momentum_reversion/mr_panel.py`: construcción de paneles desde la DB y de los retornos forward.
- `momentum_industrias/`: 6-1m entre industrias y estrategia de dos capas.
- `presentacion/build_datos_v4.py`: los KPI de la presentación.
- `robustez/`:
  - top 30 vs bottom 30;
  - legacy rebalance de la app, sin restricción y con restricción por industria a peso de capitalización del trimestre;
  - comprobación ponderada por capitalización.

## Qué revisar
Revisa la corrección del backtest, no el estilo. Prioridad:

1. **Sesgo de anticipación.** Ningún score de un trimestre debe usar datos posteriores a ese trimestre. Comprobar:
   - lambdas de Yeo-Johnson;
   - winsor 5/95 y z;
   - filtro NA del 40% por celda;
   - pesos de industria por Market Cap.

   Además, el retorno forward a 3m (TR de Bloomberg) debe empezar después de la fecha del score.
2. **`final_profiles.py` (`score_rows` y `spec`):**
   - el reparto de pesos cuando una métrica sale por NA y cuando un pilar no aplica (financieras y cíclicas en Value, `SUBSTITUTIONS` en Quality);
   - la dirección de los yields;
   - el mapeo `replace_with_high` → `replace_with_low`;
   - los signos de Change/Trend.
3. **Benchmarks FF ROE, FF B/M y ST_Rev.** Deben construirse igual que el perfil: mismo universo, mismo `long_short` y NA tratados igual. La comparación tiene que ser justa.
4. **`robustez/`:**
   - escala 0-10 de la app (`minmax10`);
   - uso de `build_legacy_rebalance_target`: umbral 5, cuantil 0,5 por industria y 3% por nombre;
   - pesos de industria del mismo trimestre;
   - retorno forward NaN tratado como 0;
   - anualización y t pareada.
5. **`build_datos_v4.py`:** que los KPI se calculen correctamente y sobre la ventana 2018/03–2026/03. Los KPI son:
   - exceso, t y t del IC;
   - ventanas de 2 años;
   - industrias;
   - Sharpe y drawdown;
   - mitades.

Para cada hallazgo, indica el impacto esperado sobre el exceso frente a FF: signo y orden de magnitud.
