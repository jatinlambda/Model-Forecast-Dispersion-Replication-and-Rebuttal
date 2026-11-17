#!/usr/bin/env python3
"""Where does MFD lose its predictive power? A ladder of controls.

Fama-MacBeth of next-month excess returns on MFD plus progressively richer
controls (paper conventions: pooled 0.5/99.5% winsorising, monthly z-scores,
% per month per 1 SD, Newey-West t with 6 lags). Run for our MFD and for the
authors' published mfd.csv, on one common sample: stock-months with all 12 of
the paper's raw controls (as in the paper's own regressions).

Rungs (each adds to the paper's 12 controls):
  paper      the 12 controls only
  MISP       Stambaugh-Yuan mispricing score (real data only to Dec 2016)
  vol block  12 extra volatility / liquidity / distress characteristics (ranked)
  153 chars  all 153 JKP characteristics, ranked, entered linearly
  ridge      one ridge forecast trained on all 153 characteristics, independent
             of the MFD forests (scripts/build_ridge_forecast.py)
  RF mean    the MFD forests' own mean forecast
  both       ridge + RF mean

Also: MFD decile spread within ridge-forecast quintiles (FF6 alphas), and
correlations between MFD, the RF mean and the ridge forecast.

Outputs go to results/controls/ (does not touch results/miller or data/mfd_full).
Usage (project root, mfd env): python scripts/test_controls.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tm", str(Path(__file__).with_name("test_miller.py")))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

VOL_BLOCK = ["rvol_21d", "ivol_ff3_21d", "rmax1_21d", "bidaskhl_21d", "zero_trades_252d", "dolvol_126d",
             "prc", "o_score", "z_score", "age", "iskew_capm_21d", "rskew_21d"]
MISP_END = pd.Period("2016-12", "M")


def build(ctrl):
    p = tm.load_panel(ROOT / "data/mfd_full/mfd_quantiles.pq")
    p = p[p.month <= tm.PAPER_END].dropna(subset=ctrl)
    p = p[["month", "permno", "ret_exc_lead1m", "me", "std", "mean"] + ctrl]

    ridge = pd.read_parquet(ROOT / "data/ridge/ridge_forecast.pq")
    ridge["month"] = ridge.pop("date").dt.to_period("M")
    p = p.merge(ridge, on=["month", "permno"], how="left")

    a = pd.read_csv(ROOT / "data/pkg_extracted/data/mfd/mfd.csv", usecols=["permno", "date", "mean", "std"],
                    engine="pyarrow").rename(columns={"std": "std_auth", "mean": "mean_auth"})
    a["month"] = pd.to_datetime(a.pop("date")).dt.to_period("M")
    a["permno"] = a["permno"].astype(int)
    p = p.merge(a, on=["month", "permno"], how="left")

    m = pd.read_csv(ROOT / "data/pkg_extracted/data/external/Misp_Score.txt", sep=";")
    m["month"] = pd.PeriodIndex(m.pop("yyyymm").astype(str), freq="M")
    m = m.rename(columns={"avg_score": "MISP"})
    m["permno"] = m["permno"].astype(int)
    p = p.merge(m[m.month <= MISP_END], on=["month", "permno"], how="left")

    rk = pd.read_parquet(ROOT / "data/derived/JKP_ranknorm.pq")
    rk.columns = ["rk_" + c for c in rk.columns]
    rk = rk.reset_index()
    rk["month"] = pd.to_datetime(rk.pop("date").astype(str)).dt.to_period("M")
    rk["permno"] = rk["permno"].astype(int)
    p = p.merge(rk, on=["month", "permno"], how="left")
    return p


def rungs(ctrl, mfd_col, mean_col):
    rk_all = [c for c in COLS if c.startswith("rk_")]
    return [
        ("paper 12 controls", [], "all"),
        ("paper 12, 1976-2016 sample", [], "misp"),
        ("+ mispricing score (1976-2016)", ["MISP"], "misp"),
        ("+ volatility/distress block (12)", ["rk_" + c for c in VOL_BLOCK], "all"),
        ("+ all 153 characteristics (linear)", rk_all, "all"),
        ("+ ridge forecast", ["ridge"], "all"),
        ("+ RF mean forecast", [mean_col], "all"),
        ("+ ridge + RF mean", ["ridge", mean_col], "all"),
    ]


def double_sort(df, first, second, factors, n1=5, n2=5):
    df = df.dropna(subset=[first, second]).copy()
    df["p1"] = df.groupby("month")[first].transform(lambda s: pd.qcut(s.rank(method="first"), n1, labels=False) + 1)
    df["p2"] = df.groupby(["month", "p1"])[second].transform(
        lambda s: pd.qcut(s.rank(method="first"), n2, labels=False) + 1)
    f = factors.copy()
    f.index = f.index - 1
    res = {}
    for w in ("vw", "ew"):
        if w == "ew":
            r = df.groupby(["month", "p1", "p2"]).ret_exc_lead1m.mean()
        else:
            g = df.assign(rw=df.ret_exc_lead1m * df.me)
            r = g.groupby(["month", "p1", "p2"]).rw.sum() / g.groupby(["month", "p1", "p2"]).me.sum()
        r = (r * 100).unstack("p2")
        hl = (r[n2] - r[1]).unstack("p1").mean(axis=1)
        m, t = tm.nw_mean(hl)
        al, ta = tm.nw_alpha(hl, f * 100)
        res[w] = (m, t, al, ta)
    return res


def main():
    out = ROOT / "results/controls"
    out.mkdir(parents=True, exist_ok=True)
    ctrl = list(tm.CONTROLS.values())
    p = build(ctrl)
    global COLS
    COLS = list(p.columns)
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say(f"Common sample: stock-months with all 12 paper controls, {p.month.min()}..{p.month.max()}, "
        f"{len(p):,} rows; ridge coverage {p.ridge.notna().mean():.1%}, authors' MFD {p.std_auth.notna().mean():.1%}")

    # correlations
    cc = p.groupby("month")[["std", "mean", "ridge", "std_auth", "mean_auth"]].corr(method="spearman") \
        .groupby(level=1).mean().loc[["std", "mean", "ridge"], ["std", "mean", "ridge", "std_auth", "mean_auth"]]
    say("\nAverage monthly Spearman correlations")
    say(cc.round(2).to_string())

    rows = []
    for label, mfd_col, mean_col in [("our MFD", "std", "mean"), ("authors' MFD", "std_auth", "mean_auth")]:
        d_all = p.dropna(subset=[mfd_col])
        d_misp = d_all[(d_all.month <= MISP_END)].dropna(subset=["MISP"])
        say(f"\n=== {label}: MFD slope as controls are added (% per month per 1 SD, NW t) ===")
        for name, extra, samp in rungs(ctrl, mfd_col, mean_col):
            d = d_misp if samp == "misp" else d_all
            r, m = tm.fama_macbeth(d, [mfd_col] + extra + ctrl, transform="z")
            added = ""
            if extra and len(extra) <= 2:
                added = "   " + "  ".join(f"{e} {r.loc[e, 'coef']:+.3f} (t {r.loc[e, 't']:+.2f})" for e in extra)
            say(f"  {name:38s} MFD {r.loc[mfd_col, 'coef']:+.3f} (t {r.loc[mfd_col, 't']:+.2f})"
                f"{added}   [{m['months']} mo, N~{m['avg_N']:,}]")
            rows.append({"mfd": label, "rung": name, "coef": r.loc[mfd_col, "coef"], "t": r.loc[mfd_col, "t"],
                         "months": m["months"], "avg_N": m["avg_N"]})
    pd.DataFrame(rows).to_csv(out / "control_ladder.csv", index=False)

    # ridge forecast on its own
    say("\n=== Ridge forecast as a predictor (1976-08..2022-11) ===")
    for name, x in [("ridge + 12 controls", ["ridge"]), ("ridge + RF mean + 12 controls", ["ridge", "mean"])]:
        r, m = tm.fama_macbeth(p, x + ctrl, transform="z")
        say(f"  {name:32s} " + "  ".join(f"{c} {r.loc[c, 'coef']:+.3f} (t {r.loc[c, 't']:+.2f})" for c in x))

    # double sorts within ridge quintiles
    factors = tm.ff_factors()
    say("\n=== MFD high-minus-low (Q5-Q1) within ridge-forecast quintiles, averaged over quintiles ===")
    for label, mfd_col in [("our MFD", "std"), ("authors' MFD", "std_auth")]:
        res = double_sort(p, "ridge", mfd_col, factors)
        for w, (m, t, al, ta) in res.items():
            say(f"  {label:13s} [{w.upper()}] spread {m:+.2f} (t {t:+.2f})   FF6 alpha {al:+.2f} (t {ta:+.2f})")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}/summary.txt and control_ladder.csv")


if __name__ == "__main__":
    main()
