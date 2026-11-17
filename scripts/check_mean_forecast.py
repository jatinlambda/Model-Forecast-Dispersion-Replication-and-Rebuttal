#!/usr/bin/env python3
"""Robustness for Step 2: is MFD subsumed by the ensemble mean forecast?

1. Fama-MacBeth on the AUTHORS published mfd.csv (std, mean) with our returns/controls.
2. Dependent double sorts: MFD quintiles within mean-forecast quintiles (and the reverse),
   VW and EW, with FF6 alphas (Newey-West, 6 lags).

Usage (project root, mfd env): python scripts/check_mean_forecast.py
"""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("tm", str(Path(__file__).with_name("test_miller.py"))); tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
from pathlib import Path
fac = tm.ff_factors()
panel = tm.load_panel(Path("data/mfd_full/mfd_quantiles.pq"))
panel = panel[panel.month <= tm.PAPER_END]
ctrl = list(tm.CONTROLS.values())
# authors' MFD + mean, with our returns and controls
a = pd.read_csv("data/pkg_extracted/data/mfd/mfd.csv", usecols=["permno","date","mean","std"], engine="pyarrow")
a["month"] = pd.to_datetime(a.date).dt.to_period("M"); a["permno"] = a.permno.astype(int)
a = a.drop(columns="date").merge(panel[["month","permno","ret_exc_lead1m","me"] + ctrl], on=["month","permno"])
print("=== Fama-MacBeth on the AUTHORS' mfd.csv (std, mean), 1976-08..2022-11, z-scored, NW 6 lags ===")
for name, x in [("MFD + controls", ["std"]), ("mean + controls", ["mean"]), ("MFD + mean + controls", ["std","mean"])]:
    r, m = tm.fama_macbeth(a, x + ctrl, transform="z")
    print(f"  {name:24s}", "  ".join(f"{c} {r.loc[c,'coef']:+.3f} (t {r.loc[c,'t']:+.2f})" for c in x), f"[{m['months']} mo, N~{m['avg_N']}]")
print("  corr(MFD, mean), avg monthly Spearman:", round(a.groupby("month")[["std","mean"]].corr(method="spearman").xs("std", level=1)["mean"].mean(), 3))

def double_sort(df, first, second, label, n1=5, n2=5):
    df = df.dropna(subset=[first, second]).copy()
    df["p1"] = df.groupby("month")[first].transform(lambda s: pd.qcut(s.rank(method="first"), n1, labels=False) + 1)
    df["p2"] = df.groupby(["month","p1"])[second].transform(lambda s: pd.qcut(s.rank(method="first"), n2, labels=False) + 1)
    f = fac.copy(); f.index = f.index - 1
    print(f"=== {label}: {second} H-L (Q{n2}-Q1) within {first} quintiles, % per month ===")
    for w in ("vw", "ew"):
        if w == "ew":
            r = df.groupby(["month","p1","p2"]).ret_exc_lead1m.mean()
        else:
            g = df.assign(rw=df.ret_exc_lead1m*df.me)
            r = g.groupby(["month","p1","p2"]).rw.sum() / g.groupby(["month","p1","p2"]).me.sum()
        r = (r*100).unstack("p2"); hl = (r[n2] - r[1]).unstack("p1")
        cells = []
        for q in range(1, n1+1):
            m, t = tm.nw_mean(hl[q]); al, ta = tm.nw_alpha(hl[q], f*100)
            cells.append(f"Q{q} {m:+.2f}({t:+.1f})")
        avg = hl.mean(axis=1); m, t = tm.nw_mean(avg); al, ta = tm.nw_alpha(avg, f*100)
        print(f"  [{w.upper()}] " + "  ".join(cells) + f" | avg {m:+.2f} (t {t:+.2f}), FF6 alpha {al:+.2f} (t {ta:+.2f})")

double_sort(panel, "mean", "std", "OURS")
double_sort(a, "mean", "std", "AUTHORS' mfd.csv")
double_sort(panel, "std", "mean", "OURS (reverse: mean within MFD)")
