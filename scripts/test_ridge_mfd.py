#!/usr/bin/env python3
"""MFD built from 100 ridge investors instead of random forests.

Each ridge investor sees the same random 76-of-153 characteristic subset as the
corresponding forest investor; ridge is deterministic, so all dispersion comes
from information-set differences (the cleanest version of the paper's concept).

Questions:
  1. Does ridge MFD predict returns (decile sort, Fama-MacBeth)?
  2. Is it absorbed by the ridge investors' own mean forecast?
  3. ... by an independent forecast (single ridge on all 153; the forests' mean)?
  4. Cross-check: is the FOREST MFD absorbed by the RIDGE investors' mean?
  5. Asymmetry A among ridge investors.

Inputs: data/mfd_ridge/mfd_quantiles.pq (construct_mfd.py --model-type ridge),
        data/mfd_full/mfd_quantiles.pq (forest MFD), data/ridge/ridge_forecast.pq.
Outputs: results/ridge_mfd/ (nothing else is written).
Usage (project root, mfd env): python scripts/test_ridge_mfd.py
"""
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tm", str(Path(__file__).with_name("test_miller.py")))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)


def double_sort(df, first, second, factors, n1=5, n2=5):
    """Dependent sort: `second` quintiles within `first` quintiles; Q5-Q1 spread of
    `second` averaged over `first` quintiles, with FF6 alpha (NW 6 lags)."""
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
        hl = ((r * 100).unstack("p2")[n2] - (r * 100).unstack("p2")[1]).unstack("p1").mean(axis=1)
        m, t = tm.nw_mean(hl)
        al, ta = tm.nw_alpha(hl, f * 100)
        res[w] = (m, t, al, ta)
    return res


def main():
    out = ROOT / "results/ridge_mfd"
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    ctrl = list(tm.CONTROLS.values())

    p = tm.load_panel(ROOT / "data/mfd_ridge/mfd_quantiles.pq")
    p = p.rename(columns={"std": "r_std", "mean": "r_mean"})
    rf = pd.read_parquet(ROOT / "data/mfd_full/mfd_quantiles.pq", columns=["permno", "date", "std", "mean"])
    rf["month"] = pd.to_datetime(rf.pop("date")).dt.to_period("M")
    rf = rf.rename(columns={"std": "rf_std", "mean": "rf_mean"})
    single = pd.read_parquet(ROOT / "data/ridge/ridge_forecast.pq")
    single["month"] = single.pop("date").dt.to_period("M")
    p = p.merge(rf, on=["month", "permno"], how="left").merge(single, on=["month", "permno"], how="left")
    ins, oos = p[p.month <= tm.PAPER_END], p[p.month > tm.PAPER_END]
    factors = tm.ff_factors()

    say(f"Ridge MFD panel: {p.month.min()}..{p.month.max()}, {len(p):,} stock-months")
    v = ["r_std", "r_mean", "rf_std", "rf_mean", "ridge"]
    cc = ins.groupby("month")[v].corr(method="spearman").groupby(level=1).mean().loc[v, v]
    say("\nAverage monthly Spearman correlations, 1976-2022 (r_ = ridge investors, rf_ = forest investors, "
        "ridge = single ridge on all 153)")
    say(cc.round(2).to_string())

    say("\n=== 1. Ridge MFD decile sorts, high-minus-low (% per month) ===")
    for name, df in [("ridge MFD, 1976-08..2022-11", ins), ("ridge MFD, OOS 2022-12..2025-11", oos)]:
        s = tm.decile_sort(df, "r_std", factors)
        for w in ("vw", "ew"):
            say(tm.fmt_sort(s[w], f"{name} [{w.upper()}]"))
            s[w].to_csv(out / f"sort_{'oos' if 'OOS' in name else 'insample'}_{w}.csv")

    common = ins.dropna(subset=ctrl)
    rows = []
    say("\n=== 2-4. Fama-MacBeth, 1976-2022, common sample (all 12 controls), z-scored, NW 6 lags ===")
    specs = [
        ("ridge MFD + 12 controls", ["r_std"]),
        ("+ ridge investors' own mean", ["r_std", "r_mean"]),
        ("+ single ridge forecast (all 153)", ["r_std", "ridge"]),
        ("+ forests' mean forecast", ["r_std", "rf_mean"]),
        ("+ own mean + forests' mean", ["r_std", "r_mean", "rf_mean"]),
        ("forest MFD + 12 controls", ["rf_std"]),
        ("forest MFD + ridge investors' mean", ["rf_std", "r_mean"]),
        ("forest MFD + forests' own mean", ["rf_std", "rf_mean"]),
        ("ridge MFD + forest MFD + both means", ["r_std", "rf_std", "r_mean", "rf_mean"]),
    ]
    for name, x in specs:
        r, m = tm.fama_macbeth(common, x + ctrl, transform="z")
        say(f"  {name:38s} " + "  ".join(f"{c} {r.loc[c, 'coef']:+.3f} (t {r.loc[c, 't']:+.2f})" for c in x)
            + f"   [{m['months']} mo, N~{m['avg_N']:,}]")
        for c in x:
            rows.append({"spec": name, "var": c, "coef": r.loc[c, "coef"], "t": r.loc[c, "t"]})

    say("\n=== 5. Asymmetry among ridge investors (W = Q80-Q20, A = (Q80-Q50)-(Q50-Q20)) ===")
    for name, x in [("W + A + controls", ["W", "A"]), ("W + A + own mean + controls", ["W", "A", "r_mean"]),
                    ("A + controls", ["A"])]:
        r, m = tm.fama_macbeth(common, x + ctrl, transform="z")
        say(f"  {name:38s} " + "  ".join(f"{c} {r.loc[c, 'coef']:+.3f} (t {r.loc[c, 't']:+.2f})" for c in x))
        for c in x:
            rows.append({"spec": "asym: " + name, "var": c, "coef": r.loc[c, "coef"], "t": r.loc[c, "t"]})
    pd.DataFrame(rows).to_csv(out / "fama_macbeth.csv", index=False)

    say("\n=== Double sorts: ridge-MFD H-L (Q5-Q1) within quintiles of a forecast, averaged ===")
    for first, label in [("r_mean", "ridge investors' mean"), ("ridge", "single ridge"), ("rf_mean", "forests' mean")]:
        res = double_sort(ins, first, "r_std", factors)
        for w, (mm, t, al, ta) in res.items():
            say(f"  within {label:22s} [{w.upper()}] spread {mm:+.2f} (t {t:+.2f})   FF6 alpha {al:+.2f} (t {ta:+.2f})")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}/summary.txt")


if __name__ == "__main__":
    main()
