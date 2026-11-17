#!/usr/bin/env python3
"""Does MFD still predict returns inside the lowest mean-forecast quintile once the mean
is held fixed more finely?

The 5x5 double sort (mean first, then MFD) leaves one significant cell: in mean
quintile 1, MFD Q5-Q1 earns -1.12% EW (FF6 alpha -0.79, t -4.3). Quintile 1 is wide,
and within it high-MFD stocks have much lower mean forecasts. Here:

  A  Finer sorts inside quintile 1: split Q1 into K mean sub-bins (K = 1, 2, 4, 5), sort
     MFD into quintiles within each sub-bin, average Q5-Q1 across sub-bins.
     Also reports the mean-forecast gap between MFD Q5 and Q1 (bp/month).
  B  Fama-MacBeth inside quintile 1: return on MFD, + mean, + mean^2, with the paper's
     12 controls (z-scored within the Q1 cross-section each month).

Sample 1976-08..2022-11. Output: results/within_q1/summary.txt (+ csv).
Usage: python scripts/test_within_q1.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tm", str(Path(__file__).with_name("test_miller.py")))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)
OUT = ROOT / "results/within_q1"


def qrank(s, n):
    """Same dense-percentile convention as double_sorts_mean_mfd.py."""
    pct = s.rank(method="dense", pct=True)
    pct = pct.where(pct != pct.min(), 1e-8)
    return np.ceil(pct * n).clip(lower=1).astype(int)


def finer_sort(q1, K, fac):
    d = q1.copy()
    d["sub"] = d.groupby("month")["mean"].transform(lambda s: qrank(s, K)) if K > 1 else 1
    d["mq"] = d.groupby(["month", "sub"])["std"].transform(lambda s: qrank(s, 5))
    out = {}
    for w in ("ew", "vw"):
        if w == "ew":
            r = d.groupby(["month", "sub", "mq"])["ret_exc_lead1m"].mean()
        else:
            g = d.assign(rw=d.ret_exc_lead1m * d.me)
            r = g.groupby(["month", "sub", "mq"])["rw"].sum() / g.groupby(["month", "sub", "mq"])["me"].sum()
        r = (r * 100).unstack("mq")
        hl = (r[5] - r[1]).groupby("month").mean()
        m, t = tm.nw_mean(hl)
        a, ta = tm.nw_alpha(hl, fac)
        out[w] = (m, t, a, ta)
    mg = d.groupby(["month", "sub", "mq"])["mean"].mean().unstack("mq")
    gap = ((mg[5] - mg[1]) * 1e4).groupby("month").mean().mean()
    lvl = d.groupby(["month", "mq"])["mean"].mean().unstack("mq").mean() * 1e4
    return out, gap, lvl[1], lvl[5], len(hl)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = tm.load_panel(ROOT / "data/mfd_full/mfd_quantiles.pq")
    p = p[p.month <= tm.PAPER_END].dropna(subset=["mean", "std", "ret_exc_lead1m", "me"])
    p["mq_all"] = p.groupby("month")["mean"].transform(lambda s: qrank(s, 5))
    q1 = p[p.mq_all == 1].copy()
    fac = tm.ff_factors() * 100
    fac.index = fac.index - 1  # factor month t+1 aligned with formation month t

    rows, lines = [], []
    say = lambda s: (print(s), lines.append(s))
    say(f"Lowest mean-forecast quintile, {q1.month.min()}..{q1.month.max()}, avg {len(q1)/q1.month.nunique():.0f} stocks/month\n")
    say("=== A. MFD Q5-Q1 inside mean quintile 1, holding the mean fixed in K finer sub-bins ===")
    say("K  bin width   mean gap Q5-Q1 (bp)   EW spread (t)   EW FF6 alpha (t)   VW spread (t)   VW FF6 alpha (t)")
    for K in (1, 2, 4, 5):
        out, gap, l1, l5, n = finer_sort(q1, K, fac)
        e, v = out["ew"], out["vw"]
        say(f"{K}  {20/K:>4.0f}%       {gap:+7.1f}            {e[0]:+.2f} ({e[1]:+.2f})   {e[2]:+.2f} ({e[3]:+.2f})      "
            f"{v[0]:+.2f} ({v[1]:+.2f})   {v[2]:+.2f} ({v[3]:+.2f})")
        rows.append(dict(K=K, bin_pct=20 / K, mean_gap_bp=gap, ew_ret=e[0], ew_t=e[1], ew_a=e[2], ew_ta=e[3],
                         vw_ret=v[0], vw_t=v[1], vw_a=v[2], vw_ta=v[3], months=n))
    pd.DataFrame(rows).to_csv(OUT / "finer_sorts.csv", index=False)

    say("\n=== B. Fama-MacBeth inside mean quintile 1 (z-scored within Q1, paper's 12 controls, NW 6) ===")
    q1["mean_z"] = q1.groupby("month")["mean"].transform(lambda s: (s - s.mean()) / s.std())
    q1["mean2"] = q1["mean_z"] ** 2
    q1["mean3"] = q1["mean_z"] ** 3
    ctrl = list(dict.fromkeys(tm.CONTROLS.values()))
    fm = []
    for name, xs in [("MFD + controls", ["std"]), ("MFD + mean + controls", ["std", "mean"]),
                     ("MFD + mean + mean^2 + controls", ["std", "mean", "mean2"]),
                     ("MFD + mean + mean^2 + mean^3 + controls", ["std", "mean", "mean2", "mean3"])]:
        res, meta = tm.fama_macbeth(q1.dropna(subset=ctrl), xs + ctrl)
        parts = "  ".join(f"{x} {res.loc[x,'coef']:+.3f} (t {res.loc[x,'t']:+.2f})" for x in xs)
        say(f"{name:42s} {parts}")
        for x in xs:
            fm.append(dict(spec=name, var=x, coef=res.loc[x, "coef"], t=res.loc[x, "t"]))
    pd.DataFrame(fm).to_csv(OUT / "fama_macbeth_q1.csv", index=False)
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
