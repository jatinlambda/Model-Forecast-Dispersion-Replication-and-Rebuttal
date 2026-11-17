#!/usr/bin/env python3
"""Step 4 (shorting constraints) and long-horizon tests.

Question: once the ensemble's mean forecast is controlled for, does MFD (or the
asymmetry A) still predict returns
  (a) among hard-to-short stocks (Miller's mechanism needs costly shorting), or
  (b) at longer horizons (overpricing should correct over several months)?

Shorting-constraint proxies (Compustat on WRDS, linked to JKP by gvkey+iid):
  IOR  = 13F institutional shares / shares outstanding  (comp.io_qaggregate,
         2004+), lagged 3 months for the 45-day 13F filing delay. Low IOR =
         few lendable shares = hard to short.
  SIR  = short interest / shares outstanding (comp.sec_shortint, 2006+), latest
         report at least 10 days before month end. High SIR = heavy shorting
         demand = costly to borrow.
Shares outstanding = JKP market_equity ($m) / price.

Each month stocks are split into terciles of each proxy; Fama-MacBeth (paper
conventions, see test_miller.py) is run within each tercile.

Usage (project root, mfd env): python scripts/test_miller_step4.py
"""
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tm", str(Path(__file__).with_name("test_miller.py")))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)
CACHE = ROOT / "data/raw/io_shortint.parquet"


def connect():
    for _ in range(5):
        try:
            return psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds",
                                    sslmode="require", connect_timeout=60)
        except psycopg2.OperationalError:
            time.sleep(15)
    raise RuntimeError("could not connect to WRDS")


def fetch_constraints():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    con = connect()
    cur = con.cursor()
    cur.execute("select gvkey, iid, datadate, iotlshr from comp.io_qaggregate")
    io = pd.DataFrame(cur.fetchall(), columns=["gvkey", "iid", "datadate", "iotlshr"])
    cur.execute("select gvkey, iid, datadate, shortintadj from comp.sec_shortint "
                "where datadate >= '2004-01-01'")
    si = pd.DataFrame(cur.fetchall(), columns=["gvkey", "iid", "datadate", "shortintadj"])
    con.close()
    io["kind"], si["kind"] = "io", "si"
    io = io.rename(columns={"iotlshr": "value"})
    si = si.rename(columns={"shortintadj": "value"})
    out = pd.concat([io, si], ignore_index=True)
    out["datadate"] = pd.to_datetime(out["datadate"])
    out["value"] = out["value"].astype(float)
    out.to_parquet(CACHE, index=False)
    return out


def attach_constraints(panel):
    ids = pd.read_parquet(ROOT / "data/raw/usa.parquet", columns=["date", "permno", "gvkey", "iid", "prc"],
                          filters=[("date", ">=", 20040101)])
    ids["month"] = pd.to_datetime(ids.pop("date").astype(str)).dt.to_period("M")
    ids["permno"] = ids["permno"].astype(int)
    p = panel[panel.month >= pd.Period("2004-01", "M")].merge(ids, on=["month", "permno"], how="left")
    p["shares"] = p["market_equity"] * 1e6 / p["prc"].abs()
    p["mend"] = p["month"].dt.to_timestamp("M")
    p["gvkey"] = p["gvkey"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    p["iid"] = p["iid"].astype(str)

    c = fetch_constraints()
    for kind, lag_days, name in [("io", 92, "IOR"), ("si", 10, "SIR")]:
        x = c[c.kind == kind][["gvkey", "iid", "datadate", "value"]].copy()
        x["avail"] = x["datadate"] + pd.Timedelta(days=lag_days)
        x = x.sort_values("avail")
        left = p[["gvkey", "iid", "mend"]].reset_index().sort_values("mend")
        m = pd.merge_asof(left, x, left_on="mend", right_on="avail", by=["gvkey", "iid"],
                          direction="backward", tolerance=pd.Timedelta(days=200 if kind == "io" else 60))
        p.loc[m["index"].to_numpy(), name] = m["value"].to_numpy()
        p[name] = p[name] / p["shares"]
    p["IOR"] = p["IOR"].clip(upper=1.0)
    return p


def tercile_fm(p, proxy, xsets, labels, ctrl):
    d = p.dropna(subset=[proxy]).copy()
    d["terc"] = d.groupby("month")[proxy].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) + 1)
    rows = []
    for t in (1, 2, 3):
        sub = d[d.terc == t]
        for name, x in xsets:
            r, m = tm.fama_macbeth(sub, x + ctrl, transform="z")
            for v in x:
                rows.append({"proxy": proxy, "tercile": labels[t - 1], "spec": name, "var": v,
                             "coef": r.loc[v, "coef"], "t": r.loc[v, "t"], "months": m["months"], "avg_N": m["avg_N"]})
    return pd.DataFrame(rows)


def horizon_fm(panel, ctrl, horizons=range(1, 13)):
    """Signal at month t, return in month t+k: lag signals by k-1 months, use ret_exc_lead1m."""
    sig_cols = ["std", "mean", "A", "W"] + ctrl
    sig = panel[["month", "permno"] + sig_cols]
    rets = panel[["month", "permno", "ret_exc_lead1m"]]
    rows = []
    for k in horizons:
        s = sig.copy()
        s["month"] = s["month"] + (k - 1)
        d = s.merge(rets, on=["month", "permno"], how="inner")
        d = d[d.month <= tm.PAPER_END]
        for name, x in [("MFD + controls", ["std"]), ("MFD + mean + controls", ["std", "mean"]),
                        ("W + A + mean + controls", ["W", "A", "mean"])]:
            r, m = tm.fama_macbeth(d, x + ctrl, transform="z")
            for v in x:
                rows.append({"k": k, "spec": name, "var": v, "coef": r.loc[v, "coef"], "t": r.loc[v, "t"],
                             "months": m["months"]})
    return pd.DataFrame(rows)


def main():
    out = ROOT / "results/miller"
    out.mkdir(parents=True, exist_ok=True)
    ctrl = list(tm.CONTROLS.values())
    panel = tm.load_panel(ROOT / "data/mfd_full/mfd_quantiles.pq")

    # ---------- shorting constraints
    p = attach_constraints(panel)
    for proxy in ("IOR", "SIR"):
        cov = p.groupby("month")[proxy].apply(lambda s: s.notna().mean())
        print(f"{proxy}: coverage {cov.mean():.0%} of stock-months from {p.loc[p[proxy].notna(), 'month'].min()}; "
              f"median {p[proxy].median():.3f}")
    xsets = [("MFD + controls", ["std"]), ("MFD + mean + controls", ["std", "mean"]),
             ("W + A + mean + controls", ["W", "A", "mean"])]
    proxies = {"IOR": ["low IOR (hard to short)", "mid IOR", "high IOR (easy)"],
               "SIR": ["low SIR (easy)", "mid SIR", "high SIR (hard to short)"]}
    parts = []
    for proxy, labels in proxies.items():
        # comp.io_qaggregate only covers ~900 securities/quarter before 2025, so IOR
        # is unusable as history; skip any proxy with too little coverage.
        if p.groupby("month")[proxy].apply(lambda s: s.notna().mean()).mean() < 0.5:
            print(f"skipping {proxy}: coverage below 50% of stock-months")
            continue
        parts.append(tercile_fm(p, proxy, xsets, labels, ctrl))
    res = pd.concat(parts)
    res.to_csv(out / "step4_constraints.csv", index=False)
    print("\n=== Step 4: Fama-MacBeth within shorting-constraint terciles, 2004/06..2025-11, "
          "z-scored, NW 6 lags (% per month per 1 SD) ===")
    for (proxy, terc), g in res.groupby(["proxy", "tercile"], sort=False):
        print(f"-- {proxy}: {terc}  [{g.months.iloc[0]} mo, N~{g.avg_N.iloc[0]:,}]")
        for spec, gg in g.groupby("spec", sort=False):
            print(f"   {spec:26s} " + "  ".join(f"{r.var} {r.coef:+.3f} (t {r.t:+.2f})" for r in gg.itertuples()))

    # ---------- horizons
    h = horizon_fm(panel, ctrl)
    h.to_csv(out / "horizons.csv", index=False)
    print("\n=== Horizons: return in month t+k on signals at t, 1976..2022, z-scored, NW 6 lags ===")
    piv = h.assign(cell=h.apply(lambda r: f"{r.coef:+.3f} ({r.t:+.1f})", axis=1))
    piv = piv.pivot_table(index="k", columns=["spec", "var"], values="cell", aggfunc="first")
    piv = piv[[("MFD + controls", "std"), ("MFD + mean + controls", "std"), ("MFD + mean + controls", "mean"),
               ("W + A + mean + controls", "A")]]
    piv.columns = ["MFD alone", "MFD | mean", "mean | MFD", "A | W, mean"]
    print(piv.to_string())


if __name__ == "__main__":
    main()
