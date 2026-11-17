#!/usr/bin/env python3
"""Economic profile of MFD: where in the forecast distribution does disagreement live,
which stocks are high-MFD, and how much of MFD is a smooth function of the mean forecast.

Outputs (results/economic/):
  by_mean_bin.csv    20 monthly mean-forecast bins: avg MFD pct rank, avg next-month return,
                     avg optimist forecast (q80) pct rank vs. all stocks' mean forecasts
  profile_by_mfd.csv avg percentile rank of controls by MFD quintile
  mfd_by_mean_decile.csv  % of each mean-forecast decile in each MFD quintile
  mfd_by_mean_quintile.csv / by_mean_quintile.csv  same by mean quintile; return and q80 rank by mean quintile
  r2.csv             avg monthly R^2 of MFD rank on mean-rank polynomials / controls
  summary.txt

Sample 1976-08..2022-11 (paper period). Usage: python scripts/economic_profile.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from test_miller import CONTROLS, PAPER_END, load_panel  # noqa: E402

OUT = ROOT / "results/economic"
OUT.mkdir(parents=True, exist_ok=True)


def pct(s):
    return s.rank(pct=True)


def main():
    d = load_panel(ROOT / "data/mfd_full/mfd_quantiles.pq")
    d = d[d["month"] <= pd.Period(PAPER_END, "M")].copy()
    d = d.dropna(subset=["mean", "std", "ret_exc_lead1m"])
    g = d.groupby("month")
    d["mean_pct"] = g["mean"].transform(pct)
    d["mfd_pct"] = g["std"].transform(pct)
    d["mean_bin"] = np.minimum((d["mean_pct"] * 20).astype(int), 19) + 1
    d["mfd_q"] = np.minimum((d["mfd_pct"] * 5).astype(int), 4) + 1

    # optimist forecast (q80) placed in the distribution of all stocks' mean forecasts that month
    def q80_rank(m):
        ref = np.sort(m["mean"].values)
        return pd.Series(np.searchsorted(ref, m["q80"].values) / len(ref), index=m.index)
    d["q80_vs_mean"] = g.apply(q80_rank).reset_index(level=0, drop=True)
    d["ret"] = d["ret_exc_lead1m"] * 100

    # 1. by mean bin: monthly averages, then time-series average
    mb = (d.groupby(["month", "mean_bin"])[["mfd_pct", "ret", "q80_vs_mean"]].mean()
          .groupby("mean_bin").mean())
    mb["share_top_mfd_q"] = (d.assign(top=d["mfd_q"] == 5).groupby(["month", "mean_bin"])["top"].mean()
                             .groupby("mean_bin").mean())
    mb.to_csv(OUT / "by_mean_bin.csv")

    # 1b. heatmap input: share of each mean-forecast decile in each MFD quintile (columns sum to 100%)
    d["mean_dec"] = np.minimum((d["mean_pct"] * 10).astype(int), 9) + 1
    ct = (d.groupby(["month", "mean_dec"])["mfd_q"].value_counts(normalize=True).unstack(fill_value=0)
          .groupby("mean_dec").mean().T * 100)
    ct.index.name = "mfd_quintile"
    ct.to_csv(OUT / "mfd_by_mean_decile.csv")
    d["mean_q"] = np.minimum((d["mean_pct"] * 5).astype(int), 4) + 1
    cq = (d.groupby(["month", "mean_q"])["mfd_q"].value_counts(normalize=True).unstack(fill_value=0)
          .groupby("mean_q").mean().T * 100)
    cq.index.name = "mfd_quintile"
    cq.to_csv(OUT / "mfd_by_mean_quintile.csv")
    (d.groupby(["month", "mean_q"])[["ret", "q80_vs_mean"]].mean().groupby("mean_q").mean()
     .to_csv(OUT / "by_mean_quintile.csv"))

    # 2. characteristics by MFD quintile (percentile ranks within month)
    prof = {}
    for k, c in CONTROLS.items():
        if c in d:
            d[f"r_{k}"] = d.groupby("month")[c].transform(pct)
            prof[k] = (d.groupby(["month", "mfd_q"])[f"r_{k}"].mean().groupby("mfd_q").mean())
    prof = pd.DataFrame(prof)
    prof["mean_forecast"] = d.groupby(["month", "mfd_q"])["mean_pct"].mean().groupby("mfd_q").mean()
    prof.to_csv(OUT / "profile_by_mfd.csv")

    # 3. how much of MFD (rank) is a function of the mean (rank)?
    ctrl_r = [f"r_{k}" for k in CONTROLS if f"r_{k}" in d]
    specs = {
        "mean (linear)": lambda m: [m.mean_pct],
        "mean (cubic)": lambda m: [m.mean_pct, m.mean_pct**2, m.mean_pct**3],
        "mean (20 bins)": lambda m: [pd.get_dummies(m.mean_bin, drop_first=True).astype(float).values],
        "12 controls (linear)": lambda m: [m[ctrl_r].values],
        "12 controls + mean (20 bins)": lambda m: [m[ctrl_r].values,
                                                  pd.get_dummies(m.mean_bin, drop_first=True).astype(float).values],
    }
    rows = []
    for name, f in specs.items():
        r2s = []
        for _, m in d.dropna(subset=ctrl_r).groupby("month"):
            X = np.column_stack([np.ones(len(m))] + [np.asarray(x, float).reshape(len(m), -1) for x in f(m)])
            y = m["mfd_pct"].values
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2s.append(1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        rows.append((name, np.mean(r2s)))
    r2 = pd.DataFrame(rows, columns=["explain MFD rank with", "avg_monthly_R2"])
    r2.to_csv(OUT / "r2.csv", index=False)

    # bottom-bin facts
    bot = d[d.mean_bin <= 2]
    frac_opt_below_median = (bot["q80_vs_mean"] < 0.5).groupby(bot["month"]).mean().mean()

    with open(OUT / "summary.txt", "w") as fh:
        p = lambda *a: (print(*a), print(*a, file=fh))
        p(f"Sample {d.month.min()}..{d.month.max()}, {len(d):,} stock-months\n")
        p("=== By mean-forecast bin (1 = lowest 5%) ===")
        p(mb.round(3).to_string())
        p("\n=== Avg percentile rank of characteristics by MFD quintile ===")
        p(prof.round(2).to_string())
        p("\n=== Avg monthly R^2 of MFD rank ===")
        p(r2.round(3).to_string(index=False))
        p(f"\nBottom 10% of mean forecast: share of stocks whose 80th-pct (optimist) forecast is below the "
          f"median stock's mean forecast: {frac_opt_below_median:.1%}")


if __name__ == "__main__":
    main()
