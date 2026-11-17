#!/usr/bin/env python3
"""Compare our replicated MFD with the authors' published mfd.csv.

Matches stock-months on (permno, calendar month) and reports:
  * coverage: stocks per month in each file and how many overlap
  * levels: average MFD (std) and mean forecast in each file
  * pooled Pearson / Spearman correlation of std and of mean
  * average month-by-month cross-sectional Spearman (what matters for sorts)
  * decile agreement: share of stocks in the same MFD decile (and within one)

Exact equality is not expected: the forests' random draws differ from the
authors' run, and the WRDS JKP vintage may differ from theirs.

Usage:
  python scripts/compare_mfd.py runs/test_1976/data/mfd/mfd_quantiles.pq
  python scripts/compare_mfd.py OURS [--theirs data/pkg_extracted/data/mfd/mfd.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(ours_path, theirs_path):
    cols = ["permno", "date", "mean", "std"]
    p = Path(ours_path)
    if p.is_dir():  # folder of month=YYYYMM_disp.pq files from a run in progress
        files = sorted(p.glob("month=*_disp.pq"))
        ours = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    elif p.suffix == ".pq":
        ours = pd.read_parquet(p, columns=cols)
    else:
        ours = pd.read_csv(p, index_col=0, usecols=[0] + cols)
    theirs = pd.read_csv(theirs_path, usecols=cols, engine="pyarrow")
    for d in (ours, theirs):
        d["month"] = pd.to_datetime(d["date"]).dt.to_period("M")
        d["permno"] = d["permno"].astype(int)
    months = sorted(set(ours.month))
    theirs = theirs[theirs.month.isin(months)]
    return ours, theirs, months


def deciles(s):
    return pd.qcut(s.rank(method="first"), 10, labels=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("--theirs", default=str(ROOT / "data/pkg_extracted/data/mfd/mfd.csv"))
    args = ap.parse_args()

    ours, theirs, months = load(args.ours, args.theirs)
    m = ours.merge(theirs[["month", "permno", "mean", "std"]], on=["month", "permno"], suffixes=("", "_a"))

    cov = pd.DataFrame({
        "ours": ours.groupby("month").size(),
        "theirs": theirs.groupby("month").size(),
        "both": m.groupby("month").size(),
    }).fillna(0).astype(int)
    print(f"Months compared: {months[0]} .. {months[-1]} ({len(months)})\n")
    if len(months) <= 24:
        print("Coverage (stocks per month)")
        print(cov.to_string())
    else:  # summarise by calendar year, with the key correlations alongside
        yr = m.month.dt.year
        by_year = pd.DataFrame({
            "months": m.groupby(yr).month.nunique(),
            "ours/mo": cov.ours.groupby(cov.index.year).mean().round(0),
            "theirs/mo": cov.theirs.groupby(cov.index.year).mean().round(0),
            "overlap": (cov.both.groupby(cov.index.year).sum() / cov.theirs.groupby(cov.index.year).sum()),
            "MFD level ratio": m.groupby(yr)["std"].mean() / m.groupby(yr)["std_a"].mean(),
            "MFD Spearman": m.groupby(m.month).apply(
                lambda g: g["std"].corr(g["std_a"], method="spearman")).groupby(lambda p: p.year).mean(),
            "mean Spearman": m.groupby(m.month).apply(
                lambda g: g["mean"].corr(g["mean_a"], method="spearman")).groupby(lambda p: p.year).mean(),
        })
        print("By year (Spearman = average within-month rank correlation, ours vs authors)")
        print(by_year.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"  overlap = {cov.both.sum() / cov.theirs.sum():.1%} of authors' stock-months, "
          f"{cov.both.sum() / cov.ours.sum():.1%} of ours\n")

    print("Levels (pooled over matched stock-months)")
    lv = pd.DataFrame({
        "ours": [m["std"].mean(), m["std"].median(), m["mean"].mean()],
        "authors": [m["std_a"].mean(), m["std_a"].median(), m["mean_a"].mean()],
    }, index=["MFD (std) mean", "MFD (std) median", "mean forecast"])
    lv["ratio"] = lv.ours / lv.authors
    print(lv.to_string(float_format=lambda x: f"{x:.6f}"), "\n")

    rows = {}
    for v in ("std", "mean"):
        per_month = m.groupby("month").apply(lambda g: g[v].corr(g[v + "_a"], method="spearman"))
        rows[v] = {
            "pooled Pearson": m[v].corr(m[v + "_a"]),
            "pooled Spearman": m[v].corr(m[v + "_a"], method="spearman"),
            "avg monthly Spearman": per_month.mean(),
            "min monthly Spearman": per_month.min(),
        }
    print("Correlation, ours vs authors")
    print(pd.DataFrame(rows).rename(columns={"std": "MFD (std)", "mean": "mean forecast"})
          .to_string(float_format=lambda x: f"{x:.3f}"), "\n")

    m["dec"] = m.groupby("month")["std"].transform(deciles)
    m["dec_a"] = m.groupby("month")["std_a"].transform(deciles)
    same = (m.dec == m.dec_a).mean()
    near = ((m.dec - m.dec_a).abs() <= 1).mean()
    ext = m[m.dec_a.isin([0, 9])]
    print(f"MFD decile agreement: same decile {same:.1%}, within one decile {near:.1%}")
    print(f"  of the authors' top/bottom-decile stocks, we put {(ext.dec == ext.dec_a).mean():.1%} "
          f"in the same extreme decile")


if __name__ == "__main__":
    main()
