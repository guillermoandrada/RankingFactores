"""Raw (no NA handling) Value panel per (period, industry) + NA map + direction + forward TR.
Usage: python build_panel.py <chunk> <n_chunks> (cwd = repo root)."""
import os
import sys
import pickle
import sqlite3

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Users\usuario\PycharmProjects\RankingFactores"
sys.path.insert(0, REPO)
from api.dependencies import get_db, get_derived_store  # noqa: E402
from modules.domain.analytics.metric_loader import fetch_metric_matrix, _get_base_na_handling  # noqa: E402
from modules.shared.tickers import canonical_ticker  # noqa: E402

METRICS = ["Current P/E Ratio", "Estimated P/E Ratio", "Current EV / EBITDA", "Estimated EV / EBITDA",
           "Current EV / EBITDAR", "Current P/B Ratio", "Estimated P/B Ratio", "Current Price to Tangible Book",
           "Current P/S Ratio", "Estimated P/S Ratio", "Current FCF Yield", "Estimated FCF Yield",
           "Current FCFF Yield", "Estimated FCFF Yield", "Current Dividend Yield", "Market Cap",
           "P/E Change", "P/E Trend", "P/B Change", "P/B Trend", "P/S Change", "P/S Trend",
           "EV/EBITDA Change", "EV/EBITDA Trend", "FCF Yield Change", "FCF Yield Trend", "P/TBV Change",
           "Current Book to Market", "Current P/GP", "Estimated P/GP", "Current ROE", "Estimated ROE",
           "Current Operating Cash Flow", "Current Net Interest Margin"]
PERIODS = [f"{y}/{m}" for y in range(2017, 2027) for m in ("03/30", "06/30", "09/30", "12/30")]
PERIODS = [p for p in PERIODS if "2017/03/30" <= p <= "2026/03/30"]

if __name__ == "__main__":
    chunk, n_chunks = int(sys.argv[1]), int(sys.argv[2])
    out_dir = os.path.join(HERE, os.environ.get("RPANEL", "rpanel")); os.makedirs(out_dir, exist_ok=True)
    db, store = get_db(), get_derived_store()
    con = sqlite3.connect(os.path.join(REPO, "financial_data.db"))
    industries = [i for i in pd.read_sql("select industry_name from industries", con).industry_name
                  if i and i.lower() != "nan"]
    tr = pd.read_sql("select ticker, start_date, horizon_months, total_return from total_returns", con)
    tr["ticker"] = tr.ticker.map(canonical_ticker)
    tr = tr.drop_duplicates(["ticker", "start_date", "horizon_months"])
    formulas = store.list_formulas()
    for period in PERIODS[chunk::n_chunks]:
        path = os.path.join(out_dir, period.replace("/", "-") + ".pkl")
        if os.path.exists(path):
            continue
        frames, direction = {}, {}
        for ind in industries:
            try:
                df, d = fetch_metric_matrix(engine=db.engine, period=period, metric_names=METRICS,
                                            derived_store=store, index_name="SPX Index", industry_name=ind,
                                            apply_na_handling=False)
            except ValueError:
                continue
            direction.update(d)
            df = df.reset_index()
            df["ticker"] = df.ticker.map(canonical_ticker)
            frames[ind] = df[["security_id", "ticker"] + [m for m in METRICS if m in df.columns]]
        start = period.replace("/", "-")
        fwd = {h: tr[(tr.start_date == start) & (tr.horizon_months == h)].set_index("ticker").total_return
               for h in (3, 12)}
        pickle.dump({"period": period, "frames": frames, "direction": direction, "fwd": fwd}, open(path, "wb"))
        print("done", period, len(frames), flush=True)
    na = _get_base_na_handling(db.engine, METRICS)
    na.update({k: v.get("na_handling") for k, v in formulas.items() if k in METRICS})
    pickle.dump(na, open(os.path.join(out_dir, "..", os.environ.get("RPANEL", "rpanel") + "_na_map.pkl"), "wb"))
