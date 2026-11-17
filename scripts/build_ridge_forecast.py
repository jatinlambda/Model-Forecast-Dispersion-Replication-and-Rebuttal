#!/usr/bin/env python3
"""Single ridge expected-return forecast, trained independently of the MFD forests.

Used as an outside control: if a separately trained linear model's forecast also
absorbs MFD's predictive power, the absorption cannot be a mechanical artefact of
taking the mean and the dispersion from the same 100 forests.

Same setup as the MFD forests (programs/01_run_mfd/construct_mfd.py):
  * features: all 153 rank-normalised JKP characteristics (data/derived/JKP_ranknorm.pq)
  * target: next-month excess return, winsorised 1%/1% within each month
  * rolling 120-month window, refit every 12 months, blocks Aug..Jul from Aug 1976
  * only stock-months with a next-month return (as in MFD's set_data)
  * ridge penalty alpha = 0.01, the paper's ridge investors (config 17)
One model on all characteristics -- no investors, no feature subsets.

Output: data/ridge/ridge_forecast.pq  (date, permno, ridge)
Does not touch data/mfd_full/.

Usage (project root, mfd env): python scripts/build_ridge_forecast.py [--alpha 0.01]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--fit-frequency", type=int, default=12)
    ap.add_argument("--first", default="1976-08")
    ap.add_argument("--out", default=str(ROOT / "data/ridge/ridge_forecast.pq"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    X = pd.read_parquet(ROOT / "data/derived/JKP_ranknorm.pq")
    r = pd.read_parquet(ROOT / "data/derived/RET.pq").rename(columns={"ret_exc_lead1m": "ret"}).dropna()
    d = X.join(r, how="inner")
    del X, r
    d = d.sort_index()
    feats = [c for c in d.columns if c != "ret"]
    dates = list(d.index.get_level_values(0).unique())
    dt = pd.to_datetime(pd.Series(dates).astype(str))
    first = int(np.argmax(dt >= pd.Timestamp(args.first)))
    first = max(first, args.window)
    print(f"{len(d):,} stock-months, {len(feats)} features, forecasts from {dt[first].date()} "
          f"({len(dates) - first} months), loaded in {time.time() - t0:.0f}s")

    preds = []
    for b in range(first, len(dates), args.fit_frequency):
        train = dates[b - args.window:b]
        tr = d.loc[train]
        y = tr.groupby(level=0)["ret"].transform(lambda x: np.asarray(winsorize(x.to_numpy(), [0.01, 0.01])))
        model = Ridge(alpha=args.alpha, fit_intercept=True).fit(tr[feats].to_numpy(), y.to_numpy())
        block = dates[b:b + args.fit_frequency]
        te = d.loc[block]
        preds.append(pd.DataFrame({"date": te.index.get_level_values(0), "permno": te.index.get_level_values(1),
                                   "ridge": model.predict(te[feats].to_numpy())}))
        print(f"  block {dt[b].date()}: trained on {len(tr):,} obs, predicted {len(te):,}")
    p = pd.concat(preds, ignore_index=True)
    p["date"] = pd.to_datetime(p["date"].astype(str)).dt.to_period("M").dt.to_timestamp("M")
    p["permno"] = p["permno"].astype(int)
    p.to_parquet(out, index=False)
    print(f"Wrote {out}: {len(p):,} forecasts, {p.date.min().date()}..{p.date.max().date()} "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
