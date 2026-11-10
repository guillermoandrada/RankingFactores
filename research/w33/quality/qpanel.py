"""Raw (no NA fill) quality panel per (period, industry) + DB NA conventions + 3m forward TR.
Usage: python qpanel.py <chunk> <n_chunks>  (cwd = repo root)."""
import os
import sys
import pickle
import sqlite3

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Users\usuario\PycharmProjects\RankingFactores"
sys.path.insert(0, REPO)
from api.dependencies import get_db, get_derived_store  # noqa: E402
from modules.domain.analytics.metric_loader import fetch_metric_matrix  # noqa: E402
from modules.shared.tickers import canonical_ticker  # noqa: E402

METRICS = [
    # Rentabilidad
    "Current ROE", "Current ROA", "Current ROIC", "Estimated ROE", "Estimated ROA", "Estimated ROIC",
    "5Y Average ROIC", "5Y Average ROA", "Current GP/Assets", "Estimated GP/Assets",
    # Calidad de beneficios
    "Current Accruals/Assets", "Current CFO/Assets", "Estimated CFO/Assets", "Current OCF/Net Income",
    "Estimated OCF/Net Income",
    # Endeudamiento
    "Current Net Debt / EBITDA", "Estimated Net Debt / EBITDA", "Current Net Debt/Assets",
    "Estimated Net Debt/Assets", "Current Net Debt/OCF", "Current OCF/LT Debt", "Current LT Debt/Assets",
    "Current EBIT/Interest",
    # Estabilidad (deviations built later from level pairs)
    "EPS CV Squared", "Current Operating Margin", "5Y Average Operating Margin", "Current Net Margin",
    "5Y Average Net Margin",
    # Márgenes
    "Current EBIT Margin", "Estimated EBIT Margin", "Current Gross Margin", "Estimated Gross Margin",
    # Change/Trend
    "ROA Trend", "ROA Change", "CFO/Assets Change", "CFO/Assets Trend", "OCF/Net Income Change", "ROE Change",
    "ROE Trend", "ROIC Change", "ROIC Trend", "EBIT Margin Change", "EBIT Margin Trend", "GP/Assets Change",
    "GP/Assets Trend", "Gross Margin Change", "OCF/LT Debt Change",
    # Financials
    "Current Net Interest Margin", "Past Efficiency Ratio", "Current Equity/Assets", "Estimated Equity/Assets",
    "Net Interest Margin Change",
]
# Fase 2: every metric of every saved Quality profile + stability candidates + diagnostics
import json as _json
_cfg = _json.load(open(os.path.join(REPO, "modules", "config", "ranking_profiles.json"), encoding="utf-8"))
_cfg = _cfg.get("profiles", _cfg)
PROFILE_METRICS = sorted({m for n, v in _cfg.items() if n.startswith("Quality") and "FF" not in n
                          for m in v["nodes"]["Scoring"]["inputs"]})
STABILITY_BASE = ["EPS Variability", "Net Income Volatility", "Current Net Income", "Current Total Assets",
                  "Current EPS", "Last Price", "Volatility 252 Calendar Days", "Current Total Assets Growth"]
METRICS = list(dict.fromkeys(METRICS + PROFILE_METRICS + STABILITY_BASE))
INDEX = "SPX Index"
PERIODS = [f"{y}/{m}" for y in range(2017, 2027) for m in ("03/30", "06/30", "09/30", "12/30")]
PERIODS = [p for p in PERIODS if "2017/03/30" <= p <= "2026/03/30"]

if __name__ == "__main__":
    chunk, n_chunks = int(sys.argv[1]), int(sys.argv[2])
    out_dir = os.path.join(HERE, os.environ.get("QPANEL", "qpanel")); os.makedirs(out_dir, exist_ok=True)
    db, store = get_db(), get_derived_store()
    con = sqlite3.connect(os.path.join(REPO, "financial_data.db"))
    industries = [i for i in pd.read_sql("select industry_name from industries", con).industry_name
                  if i and i.lower() != "nan"]
    na_conv = dict(pd.read_sql('select metric_name, "n/a treatment" na from metrics', con).values)
    na_conv.update({k: v.get("na_handling") for k, v in store.list_formulas().items()})
    tr = pd.read_sql("select ticker, start_date, horizon_months, total_return from total_returns", con)
    tr["ticker"] = tr.ticker.map(canonical_ticker)
    tr = tr.drop_duplicates(["ticker", "start_date", "horizon_months"])
    for period in PERIODS[chunk::n_chunks]:
        path = os.path.join(out_dir, period.replace("/", "-") + ".pkl")
        if os.path.exists(path):
            continue
        frames, direction = {}, {}
        for ind in industries:
            try:
                df, d = fetch_metric_matrix(engine=db.engine, period=period, metric_names=METRICS,
                                            derived_store=store, index_name=INDEX, industry_name=ind,
                                            apply_na_handling=False)
            except ValueError:
                continue
            direction.update(d)
            df = df.reset_index()
            df["ticker"] = df.ticker.map(canonical_ticker)
            frames[ind] = df[["security_id", "ticker"] + [m for m in METRICS if m in df.columns]].copy()
        fwd = tr[(tr.start_date == period.replace("/", "-")) & (tr.horizon_months == 3)].set_index(
            "ticker").total_return
        pickle.dump({"period": period, "frames": frames, "direction": direction, "fwd": fwd,
                     "na": {m: na_conv.get(m) for m in METRICS}}, open(path, "wb"))
        print("done", period, len(frames), flush=True)
