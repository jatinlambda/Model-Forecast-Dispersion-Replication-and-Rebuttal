#!/usr/bin/env python3
"""Steps 0-2 of the test plan: does the optimistic side of machine forecast
disagreement drive the MFD return effect (Miller 1977)?

Step 0  Replication gate: MFD decile sorts (EW and VW), high-minus-low return
        and FF6 alpha, on our MFD and on the authors' published mfd.csv.
Step 1  Can the data separate the two sides? Correlations of the optimist
        spread (Q80-Q50), pessimist spread (Q50-Q20), MFD, width and asymmetry.
Step 2  Key test: monthly Fama-MacBeth regressions of next-month excess returns
        on MFD or on width W = Q80-Q20 and asymmetry A = (Q80-Q50)-(Q50-Q20),
        with and without the mean forecast, plus the paper's 12 controls.
        Miller predicts a negative coefficient on A (equivalently b < c for
        optimist vs pessimist spread).

Conventions follow the authors' programs/02_analysis code:
  * deciles: dense percentile rank per month, ceil(rank * 10); H-L = D10 - D1
  * returns in % per month; VW weights = market equity at formation
  * alphas: FF5 + momentum, factor month aligned to the holding month,
    Newey-West (Bartlett) with 6 lags
  * Fama-MacBeth: regressors winsorized at 0.5/99.5% (pooled), then
    standardized within each month; slopes are % per month per 1 SD.
    Standard errors: Newey-West with 6 lags on the monthly slope series.

Usage (project root, mfd env):
  python scripts/test_miller.py                      # our full-sample MFD
  python scripts/test_miller.py --mfd data/mfd_full/mfd_quantiles.pq --out results/miller
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAXLAGS = 6
PAPER_END = pd.Period("2022-11", "M")   # last month in the authors' mfd.csv
CONTROLS = {  # paper's 12 controls (programs/02_analysis/helper.py mapper)
    "SUE": "niq_su", "AG": "at_gr1", "MOM": "ret_12_1", "ILLIQ": "ami_126d",
    "OP": "ope_be", "IVOL": "ivol_capm_252d", "BETA": "beta_60m", "SIZE": "market_equity",
    "BM": "be_me", "MAX": "rmax5_21d", "TURN": "turnover_126d", "STR": "ret_1_0",
}


# ----------------------------------------------------------------------------- data

def ff_factors() -> pd.DataFrame:
    """FF5 + momentum, monthly, in decimals, indexed by Period[M]."""
    def read(name):
        url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
        raw = urllib.request.urlopen(url, timeout=60).read()
        z = zipfile.ZipFile(io.BytesIO(raw))
        text = z.read(z.namelist()[0]).decode("latin-1").splitlines()
        rows, header = [], None
        for line in text:
            parts = [p.strip() for p in line.split(",")]
            if header is None and len(parts) > 1 and parts[0] == "" and parts[1]:
                header = parts[1:]
                continue
            if header is not None:
                if len(parts[0]) == 6 and parts[0].isdigit():
                    rows.append([parts[0]] + parts[1:len(header) + 1])
                elif rows:  # monthly block ended (annual block follows)
                    break
        df = pd.DataFrame(rows, columns=["ym"] + header)
        df.index = pd.PeriodIndex(df.pop("ym"), freq="M")
        return df.astype(float) / 100
    ff = read("F-F_Research_Data_5_Factors_2x3")
    mom = read("F-F_Momentum_Factor")
    mom.columns = ["MOM"]
    return ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].join(mom, how="inner")


def load_panel(mfd_path: Path) -> pd.DataFrame:
    """Our MFD distribution summary + returns, market equity and the 12 controls."""
    d = pd.read_parquet(mfd_path)
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M")
    d["permno"] = d["permno"].astype(int)
    d["W"] = d["q80"] - d["q20"]
    d["A"] = d["opt"] - d["pes"]
    d["W90"] = d["q90"] - d["q10"]
    d["A90"] = d["opt90"] - d["pes10"]

    cols = ["date", "permno", "ret_exc_lead1m"] + sorted(set(CONTROLS.values()))
    u = pd.read_parquet(ROOT / "data/raw/usa.parquet", columns=cols)
    u["month"] = pd.to_datetime(u.pop("date").astype(str)).dt.to_period("M")
    u["permno"] = u["permno"].astype(int)
    u["me"] = u["market_equity"]
    d = d.drop(columns=["date"]).merge(u, on=["month", "permno"], how="inner")
    return d.dropna(subset=["ret_exc_lead1m"])


def load_authors(panel: pd.DataFrame) -> pd.DataFrame:
    a = pd.read_csv(ROOT / "data/pkg_extracted/data/mfd/mfd.csv",
                    usecols=["permno", "date", "mean", "std"], engine="pyarrow")
    a["month"] = pd.to_datetime(a["date"]).dt.to_period("M")
    a["permno"] = a["permno"].astype(int)
    r = panel[["month", "permno", "ret_exc_lead1m", "me"]]
    return a.drop(columns="date").merge(r, on=["month", "permno"], how="inner")


# ----------------------------------------------------------------------------- stats

def nw_mean(x: pd.Series, lags: int = MAXLAGS):
    """Mean and Newey-West t-stat of a time series."""
    x = pd.Series(x).dropna().to_numpy()
    n = len(x)
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean(), x.mean() / np.sqrt(s / n)


def nw_alpha(y: pd.Series, X: pd.DataFrame, lags: int = MAXLAGS):
    """OLS of y on [1, X]; alpha and its Newey-West t-stat."""
    df = pd.concat([y.rename("y"), X], axis=1, join="inner").dropna()
    Y = df["y"].to_numpy()
    Z = np.column_stack([np.ones(len(df)), df.drop(columns="y").to_numpy()])
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    u = Y - Z @ beta
    n = len(Y)
    Zu = Z * u[:, None]
    S = Zu.T @ Zu
    for L in range(1, lags + 1):
        G = Zu[L:].T @ Zu[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    ZZi = np.linalg.inv(Z.T @ Z)
    V = ZZi @ S @ ZZi
    return beta[0], beta[0] / np.sqrt(V[0, 0])


def decile_sort(df, col, factors, pfs=10, by="month"):
    """Authors' univariate sort: EW and VW decile returns, H-L, FF6 alpha."""
    df = df.dropna(subset=[col]).copy()
    pct = df.groupby(by)[col].rank(method="dense", pct=True)
    pmin = pct.groupby(df[by]).transform("min")
    pct = pct.where(pct != pmin, 1e-8)
    df["port"] = np.ceil(pct * pfs).clip(lower=1).astype(int)
    out = {}
    for w in ("ew", "vw"):
        if w == "ew":
            r = df.groupby([by, "port"])["ret_exc_lead1m"].mean()
        else:
            g = df.assign(rw=df.ret_exc_lead1m * df.me)
            r = g.groupby([by, "port"])["rw"].sum() / g.groupby([by, "port"])["me"].sum()
        r = (r * 100).unstack()
        r["H-L"] = r[pfs] - r[1]
        fac = factors.copy()
        fac.index = fac.index - 1  # factor month t+1 aligned with formation month t
        rows = {}
        for p in list(range(1, pfs + 1)) + ["H-L"]:
            m, t = nw_mean(r[p])
            a, ta = nw_alpha(r[p], fac * 100)
            rows[p] = {"ret": m, "t": t, "alpha_FF6": a, "t_alpha": ta}
        out[w] = pd.DataFrame(rows).T
    return out


def fama_macbeth(df, xcols, ycol="ret_exc_lead1m", transform="z", diff=None):
    """Monthly cross-sectional OLS; returns avg slopes, NW t-stats, months, avg N.

    transform: 'z'    pooled winsorize 0.5/99.5% then per-month z-score (paper)
               'raw'  pooled winsorize only (slopes in raw units)
               'rank' per-month rank mapped to [-1, 1]
    diff: optional (a, b) to also test the monthly difference of two slopes.
    """
    d = df[["month", ycol] + xcols].dropna().copy()
    for c in xcols:
        if transform in ("z", "raw"):
            lo, hi = d[c].quantile([0.005, 0.995])
            d[c] = d[c].clip(lo, hi)
        if transform == "z":
            g = d.groupby("month")[c]
            d[c] = (d[c] - g.transform("mean")) / g.transform("std")
        elif transform == "rank":
            g = d.groupby("month")[c]
            d[c] = 2 * g.rank(pct=True) - 1
    d["y"] = d[ycol] * 100
    slopes, ns = [], []
    for m, g in d.groupby("month"):
        if len(g) < len(xcols) + 10:
            continue
        X = np.column_stack([np.ones(len(g)), g[xcols].to_numpy()])
        b, *_ = np.linalg.lstsq(X, g["y"].to_numpy(), rcond=None)
        slopes.append(pd.Series(b, index=["const"] + xcols, name=m))
        ns.append(len(g))
    S = pd.DataFrame(slopes)
    res = pd.DataFrame({c: nw_mean(S[c]) for c in S.columns}, index=["coef", "t"]).T
    meta = {"months": len(S), "avg_N": int(np.mean(ns))}
    if diff:
        meta["diff"] = nw_mean(S[diff[0]] - S[diff[1]])
    return res, meta


# ----------------------------------------------------------------------------- report

def fmt_sort(tab, label):
    hl = tab.loc["H-L"]
    return (f"{label:44s} D1 {tab.loc[1,'ret']:5.2f}  D10 {tab.loc[10,'ret']:5.2f}  "
            f"H-L {hl['ret']:6.2f} (t {hl['t']:5.2f})  FF6 alpha {hl['alpha_FF6']:6.2f} (t {hl['t_alpha']:5.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfd", default=str(ROOT / "data/mfd_full/mfd_quantiles.pq"))
    ap.add_argument("--out", default=str(ROOT / "results/miller"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    factors = ff_factors()
    panel = load_panel(Path(args.mfd))
    authors = load_authors(panel)
    in_paper = panel[panel.month <= PAPER_END]
    oos = panel[panel.month > PAPER_END]
    say(f"Sample: {panel.month.min()}..{panel.month.max()}, {len(panel):,} stock-months; "
        f"factors to {factors.index.max()}")

    # ---------------- Step 0
    say("\n=== Step 0: MFD decile sorts, high-minus-low (% per month; paper: VW H-L -1.14, t -4.33; "
        "EW -1.32, t -5.61; VW FF6 alpha -0.63, t -3.51) ===")
    sorts = {
        "ours, 1976-08..2022-11": decile_sort(in_paper, "std", factors),
        "authors' mfd.csv, 1976-08..2022-11": decile_sort(authors, "std", factors),
        "ours, out of sample 2022-12..2025-11": decile_sort(oos, "std", factors),
    }
    for name, s in sorts.items():
        for w in ("vw", "ew"):
            say(fmt_sort(s[w], f"{name} [{w.upper()}]"))
            s[w].to_csv(out / f"step0_{name.split(',')[0].replace(' ', '_').replace(chr(39), '')}_{w}"
                        f"{'_oos' if 'out of' in name else ''}.csv")

    # ---------------- Step 1
    say("\n=== Step 1: can the two sides be separated? (average monthly cross-sectional correlation) ===")
    v = ["opt", "pes", "std", "W", "A", "mean", "q50"]
    for method in ("pearson", "spearman"):
        cm = in_paper.groupby("month")[v].corr(method=method).groupby(level=1).mean().loc[v, v]
        say(f"-- {method}")
        say(cm.round(2).to_string())
        cm.to_csv(out / f"step1_corr_{method}.csv")
    share = (in_paper.opt > in_paper.pes).mean()
    say(f"share of stock-months with optimist spread > pessimist spread: {share:.1%}; "
        f"median A/W: {(in_paper.A / in_paper.W).median():.3f}")

    # ---------------- Step 2
    ctrl = list(CONTROLS.values())
    specs = [
        ("(1) MFD + controls [paper: -0.23, t -5.36]", ["std"], None),
        ("(2) opt + pes + controls", ["opt", "pes"], ("opt", "pes")),
        ("(3) opt + pes + mean + controls", ["opt", "pes", "mean"], ("opt", "pes")),
        ("(4) W + A + controls", ["W", "A"], None),
        ("(5) W + A + mean + controls", ["W", "A", "mean"], None),
        ("(6) W + A + median + controls", ["W", "A", "q50"], None),
        ("(7) W90 + A90 + mean + controls (90/10)", ["W90", "A90", "mean"], None),
    ]
    rows = []
    for sample_name, sample in [("1976-08..2022-11", in_paper), ("2022-12..2025-11 (OOS)", oos)]:
        for transform in ("z", "raw", "rank"):
            if sample_name.endswith("(OOS)") and transform != "z":
                continue
            say(f"\n=== Step 2: Fama-MacBeth, {sample_name}, regressors '{transform}' "
                f"(% per month{' per 1 SD' if transform == 'z' else ''}; NW t, 6 lags) ===")
            for name, x, diff in specs:
                res, meta = fama_macbeth(sample, x + ctrl, transform=transform, diff=diff)
                key = "  ".join(f"{c} {res.loc[c, 'coef']:+.3f} (t {res.loc[c, 't']:+.2f})" for c in x)
                extra = f"  | b-c {meta['diff'][0]:+.3f} (t {meta['diff'][1]:+.2f})" if diff else ""
                say(f"{name:42s} {key}{extra}   [{meta['months']} mo, N~{meta['avg_N']:,}]")
                for c in x:
                    rows.append({"sample": sample_name, "transform": transform, "spec": name, "var": c,
                                 "coef": res.loc[c, "coef"], "t": res.loc[c, "t"],
                                 "months": meta["months"], "avg_N": meta["avg_N"]})
                if diff:
                    rows.append({"sample": sample_name, "transform": transform, "spec": name, "var": "b-c",
                                 "coef": meta["diff"][0], "t": meta["diff"][1],
                                 "months": meta["months"], "avg_N": meta["avg_N"]})
    pd.DataFrame(rows).to_csv(out / "step2_fama_macbeth.csv", index=False)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}/summary.txt and CSVs")


if __name__ == "__main__":
    main()
