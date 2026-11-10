def risk(df, periods):
    d = df[df.p.isin(periods)].dropna(subset=["r", "bench"])
    pq, bq = d.groupby("p").r.mean(), d.groupby("p").bench.mean()
    ex = pq - bq
    dd = lambda s: float(((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min())
    return dict(sharpe=pq.mean() * 4 / (pq.std() * 2), sharpe_b=bq.mean() * 4 / (bq.std() * 2),
                vol=pq.std() * 2, vol_b=bq.std() * 2, dd=dd(pq), dd_b=dd(bq),
                ir=ex.mean() * 4 / (ex.std() * 2), corr=pq.corr(bq))
