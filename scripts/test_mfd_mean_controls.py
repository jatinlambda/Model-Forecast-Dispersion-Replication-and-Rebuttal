#!/usr/bin/env python3
"""Is the loss of MFD's significance after adding the mean forecast specific to
the paper's 12 controls?

For a ladder of control sets, monthly Fama-MacBeth (paper conventions from
test_miller.py: pooled winsorize 0.5/99.5%, per-month z-score, NW 6 lags) of
next-month excess returns on
  (a) MFD + controls
  (b) MFD + mean + controls
  (c) MFD + mean + mean^2 + controls
Sample 1976-08..2022-11 (the paper's period).

Control sets
  none        no controls
  FF4         size, B/M, 12-1 momentum, 1-month reversal
  paper12     the paper's 12 (baseline)
  paper12+px  paper 12 + 10 more price/risk/liquidity characteristics
  price49     all 49 price-based characteristics the forests see
  fund104     all 104 accounting-based characteristics the forests see
  all153      all 153 characteristics the forests see (rank-normalized inputs)
  all153+ind  all 153 + Fama-French 49 industry dummies

Usage (project root, mfd env):
  python scripts/test_mfd_mean_controls.py            # each control set on its own available sample
  python scripts/test_mfd_mean_controls.py --common   # all sets on the paper-12 sample (paper12 + extras non-missing)
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tm", str(Path(__file__).with_name("test_miller.py")))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)
ts_spec = importlib.util.spec_from_file_location("ts", str(Path(__file__).with_name("test_structured.py")))
ts = importlib.util.module_from_spec(ts_spec)
ts_spec.loader.exec_module(ts)

OUT = ROOT / "results/mfd_mean_controls"
EXTRA_PX = ["rvol_21d", "ivol_capm_21d", "rmax1_21d", "rskew_21d", "iskew_capm_21d", "zero_trades_252d",
            "dolvol_126d", "bidaskhl_21d", "prc", "beta_dimson_21d"]


def main():
    global OUT
    OUT.mkdir(parents=True, exist_ok=True)
    panel = tm.load_panel(ROOT / "data/mfd_full/mfd_quantiles.pq")
    panel = panel[panel.month <= tm.PAPER_END]
    paper12 = list(tm.CONTROLS.values())

    rn = pd.read_parquet(ROOT / "data/derived/JKP_ranknorm.pq").reset_index()
    rn["month"] = pd.to_datetime(rn.pop("date").astype(str)).dt.to_period("M")
    rn["permno"] = rn["permno"].astype(int)
    feats = [c for c in rn.columns if c not in ("month", "permno")]
    rn = rn.rename(columns={f: "rn_" + f for f in feats})
    is_px = {f: f.startswith(ts.PRICE_PREFIX) or f in ts.PRICE_EXACT for f in feats}
    price = ["rn_" + f for f in feats if is_px[f]]
    fund = ["rn_" + f for f in feats if not is_px[f]]

    u = pd.read_parquet(ROOT / "data/raw/usa.parquet", columns=["date", "permno", "ff49"] + EXTRA_PX)
    u["month"] = pd.to_datetime(u.pop("date").astype(str)).dt.to_period("M")
    u["permno"] = u["permno"].astype(int)
    p = panel.merge(rn, on=["month", "permno"], how="left").merge(u, on=["month", "permno"], how="left")
    ind = pd.get_dummies(p["ff49"].fillna(-1).astype(int), prefix="ind", dtype=float)
    ind = ind.iloc[:, 1:]  # drop one industry (intercept in regression)
    p = pd.concat([p, ind], axis=1)
    p["mean2"] = p["mean"] ** 2
    common = "--common" in sys.argv
    if common:
        p = p.dropna(subset=paper12 + EXTRA_PX).copy()
        OUT = OUT / "common_sample"
        OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(price)} price-based, {len(fund)} accounting-based characteristics; "
          f"{ind.shape[1] + 1} industries; {len(p):,} stock-months, "
          f"{p[price + fund].notna().all(axis=1).mean():.1%} with characteristics")

    sets = [
        ("none", []),
        ("FF4", ["market_equity", "be_me", "ret_12_1", "ret_1_0"]),
        ("paper12", paper12),
        ("paper12+px", paper12 + EXTRA_PX),
        ("price49", price),
        ("fund104", fund),
        ("all153", price + fund),
        ("all153+ind", price + fund + list(ind.columns)),
    ]
    rows, lines = [], []
    hdr = (f"{'controls':12s} {'#':>4s} | {'MFD alone':>16s} | {'MFD | mean':>16s} {'mean | MFD':>16s} | "
           f"{'MFD | mean,mean2':>16s} {'mean | MFD,mean2':>16s} | months  N")
    print(hdr); lines.append(hdr)
    for name, ctrl in sets:
        z = [c for c in ctrl if not c.startswith("ind_")]
        d = p.dropna(subset=["std", "mean"] + z)
        cells = []
        for spec_name, x in [("MFD", ["std"]), ("MFD+mean", ["std", "mean"]),
                             ("MFD+mean+mean2", ["std", "mean", "mean2"])]:
            dums = [c for c in ctrl if c.startswith("ind_")]
            r, meta = fm_with_dummies(d, x + z, dums) if dums else tm.fama_macbeth(d, x + z, transform="z")
            for v in x:
                rows.append({"controls": name, "n_controls": len(ctrl), "spec": spec_name, "var": v,
                             "coef": r.loc[v, "coef"], "t": r.loc[v, "t"], "months": meta["months"],
                             "avg_N": meta["avg_N"]})
                if v in ("std", "mean"):
                    cells.append(f"{r.loc[v, 'coef']:+.3f} (t {r.loc[v, 't']:+5.2f})")
        line = (f"{name:12s} {len(ctrl):4d} | {cells[0]:>16s} | {cells[1]:>16s} {cells[2]:>16s} | "
                f"{cells[3]:>16s} {cells[4]:>16s} | {meta['months']}  {meta['avg_N']:,}")
        print(line, flush=True); lines.append(line)
    pd.DataFrame(rows).to_csv(OUT / "fama_macbeth_by_controls.csv", index=False)
    (OUT / "summary.txt").write_text(
        "Fama-MacBeth 1976-08..2022-11, z-scored regressors, % per month per 1 SD, NW 6 lags"
        + (" [common sample: paper-12 + extras non-missing]" if common else "") + "\n"
        + "\n".join(lines) + "\n")


def fm_with_dummies(d, zcols, dummies):
    """fama_macbeth with dummy columns passed through untransformed."""
    d = d[["month", "ret_exc_lead1m"] + zcols + dummies].dropna().copy()
    for c in zcols:
        lo, hi = d[c].quantile([0.005, 0.995])
        d[c] = d[c].clip(lo, hi)
        g = d.groupby("month")[c]
        d[c] = (d[c] - g.transform("mean")) / g.transform("std")
    d["y"] = d["ret_exc_lead1m"] * 100
    cols = zcols + dummies
    slopes, ns = [], []
    for m, g in d.groupby("month"):
        X = np.column_stack([np.ones(len(g)), np.nan_to_num(g[cols].to_numpy(float))])
        b, *_ = np.linalg.lstsq(X, g["y"].to_numpy(), rcond=None)
        slopes.append(pd.Series(b[1:len(zcols) + 1], index=zcols, name=m))
        ns.append(len(g))
    S = pd.DataFrame(slopes)
    res = pd.DataFrame({c: tm.nw_mean(S[c]) for c in S.columns}, index=["coef", "t"]).T
    return res, {"months": len(S), "avg_N": int(np.mean(ns))}


if __name__ == "__main__":
    main()
