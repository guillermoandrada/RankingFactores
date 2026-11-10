"""Combined Momentum + Reversion panel (raw values, app NA fills, industry, fwd TR) plus residual momentum.

Sources: the Momentum_03-11 / Reversion_04-11 panels of session 35a691c6 (raw = fetch_metric_matrix SPX-wide,
apply_na_handling=False) and the yfinance daily prices in the DB. Writes mr_panel.pkl.

Residual momentum (Blitz, Huij & Martens 2011, without the volatility scaling): daily stock returns regressed
without intercept on the equal-weighted SPX return and the equal-weighted industry return over the last 12 months;
residual return summed over the formation window (12-1m and 6-1m).
Run from the repo root.
"""
import os
import sys
import glob
import pickle
import sqlite3

import numpy as np
import pandas as pd

REPO = r"C:\Users\usuario\PycharmProjects\RankingFactores"
SRC = (r"C:\Users\usuario\AppData\Local\Temp\claude\C--Users-usuario-PycharmProjects-RankingFactores"
       r"\35a691c6-8f31-454e-8c45-92d497426d38\scratchpad")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from api.dependencies import get_db  # noqa: E402
from modules.domain.analytics.metric_loader import _get_base_na_handling  # noqa: E402
from api.dependencies import get_derived_store  # noqa: E402
from modules.shared.tickers import canonical_ticker  # noqa: E402

KEEP = ["12-1m Total Return", "6-1m Total Return", "12-6m Total Return", "Risk-Adjusted 12-1m Return",
        "Risk-Adjusted 6-1m Return", "Estimated EPS Revision", "Estimated EPS Revision / Price", "Buy Rec % Change",
        "Sell Rec % Change", "Net Buy Rec % Change", "Best Analyst Rating Change", "Price / 52w High",
        "Price / 200d MA", "Price / 50d MA", "MACD 20/50 (SMA)", "MACD 50/200 (SMA)", "Information Discreteness",
        "Buy Rec %", "Sell Rec %", "Net Buy Rec %", "Best Analyst Rating", "Volume 20/60 Calendar Days",
        "Turnover 60 Calendar Days", "1m Return Reversal", "RSI 14 Reversal", "Price / 20d MA Reversal"]
RESIDUAL = ["Residual 12-1m Return", "Residual 6-1m Return"]


def na_fill_map():
    db, formulas = get_db(), get_derived_store().list_formulas()
    base = _get_base_na_handling(db.engine, KEEP)
    out = {}
    for m in KEEP:
        val = (formulas.get(m) or {}).get("na_handling") or base.get(m)
        out[m] = val
    return out


def load_prices():
    con = sqlite3.connect(os.path.join(REPO, "financial_data.db"))
    px = pd.read_sql("select ticker, price_date, close_price from price_data", con)
    px["ticker"] = px.ticker.map(canonical_ticker)
    px = px.drop_duplicates(["ticker", "price_date"]).pivot(index="price_date", columns="ticker", values="close_price")
    px.index = pd.to_datetime(px.index)
    return px.sort_index().pct_change(fill_method=None)


def residual_momentum(rets, members: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """members: ticker_c, industry. Returns residual 12-1m and 6-1m per ticker."""
    win = rets.loc[(rets.index > date - pd.DateOffset(months=12)) & (rets.index <= date)]
    cols = [t for t in members.ticker_c.unique() if t in win.columns]
    win = win[cols]
    mkt = win.mean(axis=1)
    ind_of = members.drop_duplicates("ticker_c").set_index("ticker_c").industry
    ind_ret = {i: win[[t for t in cols if ind_of.get(t) == i]].mean(axis=1) for i in ind_of.dropna().unique()}
    skip = date - pd.DateOffset(months=1)
    six = date - pd.DateOffset(months=6)
    rows = {}
    for t in cols:
        y = win[t]
        x = pd.concat([mkt, ind_ret.get(ind_of.get(t), mkt)], axis=1)
        ok = y.notna() & x.notna().all(axis=1)
        if ok.sum() < 150:
            continue
        beta, *_ = np.linalg.lstsq(x[ok].to_numpy(), y[ok].to_numpy(), rcond=None)
        res = (y - x @ beta)[ok]
        rows[t] = {"Residual 12-1m Return": res[res.index <= skip].sum(),
                   "Residual 6-1m Return": res[(res.index > six) & (res.index <= skip)].sum()}
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    fills = na_fill_map()
    rets = load_prices()
    panel = {}
    for f in sorted(glob.glob(os.path.join(SRC, "panel", "*.pkl"))):
        a = pickle.load(open(f, "rb"))
        b = pickle.load(open(os.path.join(SRC, "rev_panel", os.path.basename(f)), "rb"))
        ra, rb = a["raw"].set_index("security_id"), b["raw"].set_index("security_id")
        df = ra[["ticker_c", "industry"] + [m for m in KEEP if m in ra.columns]]
        df = df.join(rb[[m for m in KEEP if m in rb.columns and m not in ra.columns]], how="left")
        df["industry"] = df.industry.fillna(rb.industry.reindex(df.index))
        df = df[df.industry.notna()].copy()
        raw_na = df[KEEP].isna()
        for m in KEEP:
            df[m] = pd.to_numeric(df[m], errors="coerce")
            if fills[m] == "replace_with_zero":
                df[m] = df[m].fillna(0.0)
        date = pd.Timestamp(a["period"].replace("/", "-"))
        res = residual_momentum(rets, df, date)
        df = df.join(res, on="ticker_c")
        raw_na = raw_na.join(df[RESIDUAL].isna())
        direction = {**b["direction"], **a["direction"], **{m: True for m in RESIDUAL}}
        df["fwd3"] = df.ticker_c.map(a["fwd"][3])
        df["fwd12"] = df.ticker_c.map(a["fwd"][12])
        panel[a["period"]] = {"df": df, "raw_na": raw_na, "direction": direction}
        print(a["period"], len(df), round(df[RESIDUAL[0]].isna().mean(), 3), flush=True)
    pickle.dump({"panel": panel, "fills": fills}, open(os.path.join(OUT, "mr_panel.pkl"), "wb"))


if __name__ == "__main__":
    main()
