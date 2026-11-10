"""Extra Value metrics per (period, industry) for normalized earnings. Usage: build_extra.py <chunk> <n> (cwd repo root)."""
import os, sys, pickle, sqlite3
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Users\usuario\PycharmProjects\RankingFactores"
sys.path.insert(0, REPO)
from api.dependencies import get_db, get_derived_store  # noqa
from modules.domain.analytics.metric_loader import fetch_metric_matrix  # noqa
METRICS = ["Current EPS", "Past EPS", "Estimated EPS", "Last Price", "Past EV / EBITDA"]
PERIODS = [f"{y}/{m}" for y in range(2017, 2027) for m in ("03/30", "06/30", "09/30", "12/30")]
PERIODS = [p for p in PERIODS if "2017/03/30" <= p <= "2026/03/30"]
if __name__ == "__main__":
    chunk, n = int(sys.argv[1]), int(sys.argv[2])
    out = os.path.join(HERE, "rpanel_v33x"); os.makedirs(out, exist_ok=True)
    db, store = get_db(), get_derived_store()
    con = sqlite3.connect(os.path.join(REPO, "financial_data.db"))
    inds = [i for i in pd.read_sql("select industry_name from industries", con).industry_name if i and i.lower() != "nan"]
    for p in PERIODS[chunk::n]:
        frames = {}
        for ind in inds:
            try:
                df, _ = fetch_metric_matrix(engine=db.engine, period=p, metric_names=METRICS, derived_store=store,
                                            index_name="SPX Index", industry_name=ind, apply_na_handling=False)
            except ValueError:
                continue
            frames[ind] = df.reset_index()[["security_id"] + [m for m in METRICS if m in df.columns]]
        pickle.dump({"period": p, "frames": frames}, open(os.path.join(out, p.replace("/", "-") + ".pkl"), "wb"))
        print("done", p, len(frames), flush=True)
