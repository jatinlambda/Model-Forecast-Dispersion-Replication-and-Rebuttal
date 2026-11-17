#!/usr/bin/env python3
"""Idea 1: split machine forecast disagreement into a structured part (investors
disagree because they see different characteristics) and a noise part (forest
randomness from row/feature subsampling), and test which one is priced once the
mean forecast is controlled for.

Each investor k sees a fixed subset of 76 of the 153 characteristics (same in
every month; data/mfd_full/investor_features.csv). For stock i in month t, the
100 forecasts f_ik are regressed across investors on an investor design X_k:

  main   X = inclusion dummies for the K=10 characteristics whose inclusion best
             explains cross-investor forecast variation in month t-1 (ranked on
             forecasts only, never on returns; lagged so month-t selection does
             not inflate month-t fit). K=5 and K=20 as robustness.
  theme  X = counts of each investor's features in the 7 HXZ theme groups
             (one dropped); no selection at all.

  total var  V   = SS_tot / 99
  noise var  N^2 = SS_res / (99 - p)          (unbiased under pure noise)
  struct var S^2 = max(V - N^2, 0)            (df-adjusted fitted variance)

Hong-Stein gap G: per stock, OLS slope of f_ik on investor k's share of
price-based characteristics, times sd of that share across investors
(forecast difference per 1 SD more price-based). G > 0: price-based investors
more bullish than fundamental ones.

Usage (project root, mfd env): python scripts/test_structured.py
"""
import importlib.util
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tm", str(Path(__file__).with_name("test_miller.py")))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

FDIR = ROOT / "data/mfd_full"
OUT_PQ = FDIR / "mfd_structured.pq"
OUT = ROOT / "results/structured"
KS = (5, 10, 20)

PRICE_PREFIX = ("ret_", "resff3_", "seas_", "beta", "ivol_", "iskew_", "rskew_", "rvol_", "rmax", "coskew_",
                "corr_", "ami_", "dolvol", "turnover_", "zero_trades_", "bidaskhl_", "prc")
PRICE_EXACT = {"market_equity"}


def investor_design():
    inv = pd.read_csv(FDIR / "investor_features.csv")
    feats = sorted(set(inv.values.ravel()))
    D = np.zeros((inv.shape[1], len(feats)))
    col = {f: j for j, f in enumerate(feats)}
    for k in range(inv.shape[1]):
        D[k, [col[f] for f in inv.iloc[:, k]]] = 1
    is_price = np.array([f.startswith(PRICE_PREFIX) or f in PRICE_EXACT for f in feats])
    det = pd.read_excel(ROOT / "data/raw/Factor Details.xlsx").dropna(subset=["abr_jkp"]).set_index("abr_jkp")
    grp = det.loc[feats, "group"].str.upper().to_numpy()
    groups = sorted(set(grp))
    T = np.column_stack([D[:, grp == g].sum(1) for g in groups])[:, 1:]  # drop one (counts sum to 76)
    return feats, D, is_price, T, groups


def month_files():
    fs = sorted(FDIR.glob("month=*_forecasts.pq"))
    return [(f.name[6:12], f) for f in fs]


def load_F(path):
    f = pd.read_parquet(path)
    F = f[[f"inv_{k}" for k in range(100)]].to_numpy(float)
    return f[["date", "permno"]], F


def screen(args):
    """Average across stocks of single-dummy R^2 for every characteristic."""
    ym, path = args
    _, F = load_F(path)
    Fc = F - F.mean(1, keepdims=True)
    Dc = D_ - D_.mean(0)
    num = (Fc @ Dc) ** 2
    den = (Fc ** 2).sum(1, keepdims=True) * (Dc ** 2).sum(0)
    return ym, np.nanmean(num / den, axis=0)


def decompose(Fc, X):
    Xc = X - X.mean(0)
    n, p = Fc.shape[1], Xc.shape[1]
    beta = np.linalg.lstsq(Xc, Fc.T, rcond=None)[0]
    res = Fc - (Xc @ beta).T
    V = (Fc ** 2).sum(1) / (n - 1)
    N2 = (res ** 2).sum(1) / (n - 1 - p)
    return np.sqrt(np.maximum(V - N2, 0)), np.sqrt(N2), np.maximum(V - N2, 0) / V


def measures(args):
    ym, path, topk = args
    ids, F = load_F(path)
    Fc = F - F.mean(1, keepdims=True)
    out = ids.copy()
    for K, idx in topk.items():
        out[f"S{K}"], out[f"N{K}"], out[f"sh{K}"] = decompose(Fc, D_[:, idx])
    out["S_th"], out["N_th"], out["sh_th"] = decompose(Fc, T_)
    pc = PI_ - PI_.mean()
    out["G"] = (Fc @ pc) / (pc @ pc) * PI_.std(ddof=1)
    return out


def init(D, T, PI):
    global D_, T_, PI_
    D_, T_, PI_ = D, T, PI


def build():
    feats, D, is_price, T, groups = investor_design()
    PI = D[:, is_price].sum(1) / D.sum(1)
    print(f"{is_price.sum()} of {len(feats)} characteristics price-based; investor price share "
          f"mean {PI.mean():.3f}, sd {PI.std(ddof=1):.3f}, range {PI.min():.3f}-{PI.max():.3f}")
    print(f"themes {groups} (first dropped)")
    files = month_files()
    with Pool(32, initializer=init, initargs=(D, T, PI)) as pool:
        scr = dict(pool.map(screen, files))
        yms = [ym for ym, _ in files]
        jobs = []
        for i, (ym, path) in enumerate(files[1:], 1):
            r = scr[yms[i - 1]]
            order = np.argsort(-r)
            jobs.append((ym, path, {K: order[:K] for K in KS}))
        parts = pool.map(measures, jobs)
    s = pd.DataFrame(scr, index=feats).T
    top = s.mean().sort_values(ascending=False)
    print("characteristics whose inclusion moves forecasts most (avg single-dummy R^2 across stocks, all months):")
    print("  " + ", ".join(f"{f} {v:.3f}" for f, v in top.head(12).items()))
    s.to_csv(OUT / "screening_r2_by_month.csv")
    m = pd.concat(parts, ignore_index=True)
    m.to_parquet(OUT_PQ, index=False)
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(OUT_PQ) if OUT_PQ.exists() else build()
    m["month"] = pd.to_datetime(m["date"]).dt.to_period("M")
    m["permno"] = m["permno"].astype(int)
    panel = tm.load_panel(FDIR / "mfd_quantiles.pq")
    p = panel.merge(m.drop(columns="date"), on=["month", "permno"], how="inner")
    p["mean2"] = p["mean"] ** 2
    p["absG"] = p["G"].abs()
    ctrl = list(tm.CONTROLS.values())
    ins = p[p.month <= tm.PAPER_END]
    oos = p[p.month > tm.PAPER_END]

    lines = []
    def say(s=""):
        print(s); lines.append(s)

    say(f"Sample {ins.month.min()}..{ins.month.max()}, {len(ins):,} stock-months (in-sample)")
    say("\n=== Descriptives (pooled means of monthly cross-sectional stats) ===")
    for tag in ("10", "5", "20", "_th"):
        sh = ins.groupby("month")[f"sh{tag}"].mean().mean()
        say(f"structured share of MFD variance, X={'top-'+tag if tag[0] != '_' else 'themes'}: "
            f"{sh:.1%} (avg across stocks, months)")
    cols = ["std", "mean", "S10", "N10", "S_th", "N_th", "G", "absG"]
    corr = ins.groupby("month")[cols].corr(method="spearman").groupby(level=1).mean().loc[cols, cols]
    say("average monthly Spearman correlations:\n" + corr.round(2).to_string())
    corr.to_csv(OUT / "correlations.csv")

    specs = [
        ("(1) MFD + controls", ["std"], None),
        ("(2) MFD + mean + controls", ["std", "mean"], None),
        ("(3) S + N + controls", ["S10", "N10"], ("S10", "N10")),
        ("(4) S + N + mean + controls", ["S10", "N10", "mean"], ("S10", "N10")),
        ("(5) S + N + mean + mean^2 + controls", ["S10", "N10", "mean", "mean2"], ("S10", "N10")),
        ("(6) K=5:  S + N + mean + controls", ["S5", "N5", "mean"], ("S5", "N5")),
        ("(7) K=20: S + N + mean + controls", ["S20", "N20", "mean"], ("S20", "N20")),
        ("(8) themes: S + N + mean + controls", ["S_th", "N_th", "mean"], ("S_th", "N_th")),
        ("(9) G + |G| + controls", ["G", "absG"], None),
        ("(10) G + |G| + mean + controls", ["G", "absG", "mean"], None),
        ("(11) G + |G| + mean + mean^2 + controls", ["G", "absG", "mean", "mean2"], None),
    ]
    samples = [("1976-08..2022-11", ins), ("1976-08..1999-12", ins[ins.month <= pd.Period("1999-12", "M")]),
               ("2000-01..2022-11", ins[ins.month >= pd.Period("2000-01", "M")]),
               ("2022-12..2025-11 (OOS)", oos)]
    rows = []
    for sname, d in samples:
        say(f"\n=== Fama-MacBeth {sname}: z-scored regressors, % per month per 1 SD, NW 6 lags ===")
        use = specs if sname.startswith("1976-08..2022") else [s for s in specs if s[0][:4] in ("(1) ", "(2) ", "(4) ", "(8) ", "(10)")]
        for name, x, diff in use:
            r, meta = tm.fama_macbeth(d, x + ctrl, transform="z", diff=diff)
            cells = [f"{v} {r.loc[v, 'coef']:+.3f} (t {r.loc[v, 't']:+.2f})" for v in x]
            for v in x:
                rows.append({"sample": sname, "spec": name, "var": v, "coef": r.loc[v, "coef"], "t": r.loc[v, "t"],
                             "months": meta["months"], "avg_N": meta["avg_N"]})
            if diff:
                c, t = meta["diff"]
                cells.append(f"| S-N {c:+.3f} (t {t:+.2f})")
                rows.append({"sample": sname, "spec": name, "var": "S-N", "coef": c, "t": t,
                             "months": meta["months"], "avg_N": meta["avg_N"]})
            say(f"{name:42s} " + "  ".join(cells) + f"   [{meta['months']} mo]")
    pd.DataFrame(rows).to_csv(OUT / "fama_macbeth.csv", index=False)
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
