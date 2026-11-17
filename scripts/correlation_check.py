"""Test 2 (sanity check): how correlated are the optimist spread, the pessimist
spread and MFD?

Decision rule fixed in the proposal: if corr(optimist, pessimist) > 0.9, the
split is uninformative and that is the result.

Main statistic: time-series average of monthly cross-sectional correlations
(Pearson and Spearman). Pooled correlations are reported for reference; they mix
in time variation in the overall level of dispersion and so run higher.

Usage:
  python scripts/correlation_check.py [data/mfd_full/mfd_quantiles.pq] [out_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/mfd_full/mfd_quantiles.pq")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "results/correlation_check")
out.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(src)
df = df.rename(columns={"std": "mfd", "q50": "consensus"})
# Asymmetry share: fraction of the 20-80 width that sits above the median.
df["opt_share"] = df["opt"] / (df["opt"] + df["pes"])

PAIRS = [
    ("opt", "pes"),          # the decisive one
    ("opt", "mfd"),
    ("pes", "mfd"),
    ("opt90", "pes10"),      # robustness version
    ("opt90", "mfd"),
    ("pes10", "mfd"),
    ("consensus", "mfd"),
    ("consensus", "opt"),
    ("consensus", "pes"),
    ("opt_share", "mfd"),
]
cols = sorted({c for p in PAIRS for c in p})


def monthly_corrs(g, method):
    c = g[cols].corr(method=method)
    return pd.Series({f"{a}~{b}": c.loc[a, b] for a, b in PAIRS})


rows = []
by_month = {}
for method in ["pearson", "spearman"]:
    m = df.groupby("date").apply(lambda g: monthly_corrs(g, method))
    by_month[method] = m
    pooled = df[cols].corr(method=method) if method == "pearson" else df[cols].rank().corr()
    for a, b in PAIRS:
        s = m[f"{a}~{b}"]
        rows.append({
            "pair": f"{a} vs {b}", "method": method,
            "mean_monthly": s.mean(), "median_monthly": s.median(),
            "p5_monthly": s.quantile(0.05), "p95_monthly": s.quantile(0.95),
            "min_monthly": s.min(), "share_months_gt_0.9": (s > 0.9).mean(),
            "pooled": pooled.loc[a, b],
        })
summary = pd.DataFrame(rows)

# Partial correlation of opt and pes holding MFD fixed (monthly, Pearson):
# how much the two sides still co-move once overall width is removed.
def partial(g):
    X = np.column_stack([np.ones(len(g)), g["mfd"].values])
    r = {}
    for c in ["opt", "pes"]:
        beta, *_ = np.linalg.lstsq(X, g[c].values, rcond=None)
        r[c] = g[c].values - X @ beta
    return np.corrcoef(r["opt"], r["pes"])[0, 1]

pc = df.groupby("date").apply(partial)

# Stability by decade (Pearson opt~pes and Spearman opt~pes).
dec = (by_month["pearson"]["opt~pes"].groupby(by_month["pearson"].index.year // 10 * 10).mean()
       .to_frame("pearson_opt~pes"))
dec["spearman_opt~pes"] = by_month["spearman"]["opt~pes"].groupby(
    by_month["spearman"].index.year // 10 * 10).mean()
dec["n_months"] = by_month["pearson"]["opt~pes"].groupby(by_month["pearson"].index.year // 10 * 10).size()

summary.to_csv(out / "summary.csv", index=False)
by_month["pearson"].to_csv(out / "monthly_pearson.csv")
by_month["spearman"].to_csv(out / "monthly_spearman.csv")
dec.to_csv(out / "by_decade.csv")

pd.set_option("display.width", 200, "display.max_columns", 20)
print(f"Sample: {len(df):,} stock-months, {df['date'].nunique()} months, "
      f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}\n")
print(summary.round(3).to_string(index=False))
print(f"\nPartial corr(opt, pes | MFD), mean monthly: {pc.mean():.3f} "
      f"(p5 {pc.quantile(.05):.3f}, p95 {pc.quantile(.95):.3f})")
print("\nopt~pes by decade:\n", dec.round(3).to_string())
print("\nopt_share (opt / (opt+pes)) distribution:",
      df["opt_share"].describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_dict())

r = summary.query("pair == 'opt vs pes'").set_index("method")["mean_monthly"]
verdict = "SPLIT UNINFORMATIVE (>0.9)" if (r > 0.9).any() else "split has room to separate (<=0.9)"
print(f"\nDecision rule, corr(opt, pes) mean monthly: Pearson {r['pearson']:.3f}, "
      f"Spearman {r['spearman']:.3f} -> {verdict}")
