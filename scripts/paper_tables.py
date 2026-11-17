#!/usr/bin/env python3
"""Paper-style tables for an MFD variant, without and with the mean forecast.

Table 3 style  univariate MFD decile sorts (EW and VW): decile excess returns,
               H-L, and alphas for the paper's five factor models
               (CAPM, FF6, HXZ q-factor, Stambaugh-Yuan, Daniel-Hirshleifer-Sun).
Table 3 + mean MFD deciles formed within mean-forecast quintiles (the paper's
               5x10 dependent bivariate sort, Table 5 style), averaged across
               quintiles, with the same five alphas.
Table 6 style  Fama-MacBeth, the paper's three specifications:
               (1) MFD BETA SIZE BM MOM  (2) + AG OP  (3) the 12 controls
               ... and the same three with the mean forecast added.

Conventions copied from programs/02_analysis (helper.py, analysis_portfolio_sorts.py,
analysis_fama_macbeth.py): dense percentile ranks, VW by market equity, factor
month aligned to the holding month, alphas with Newey-West (6 lags); Fama-MacBeth
via linearmodels FamaMacBeth (kernel Newey-West, automatic bandwidth), regressors
winsorised 0.5/99.5% pooled then z-scored within month, returns in %.
Factor files from the replication package (data/pkg_extracted/data/external).

Usage (project root, mfd env):
  python scripts/paper_tables.py --variant ridge     # data/mfd_ridge
  python scripts/paper_tables.py --variant forest    # data/mfd_full
Outputs: results/paper_tables/<variant>/
"""
import argparse
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels import FamaMacBeth

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data/pkg_extracted/data/external"
spec = importlib.util.spec_from_file_location("tm", str(Path(__file__).with_name("test_miller.py")))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

VARIANTS = {"ridge": ROOT / "data/mfd_ridge/mfd_quantiles.pq", "forest": ROOT / "data/mfd_full/mfd_quantiles.pq"}
FM_SPECS = [
    ("(1)", ["BETA", "SIZE", "BM", "MOM"]),
    ("(2)", ["BETA", "SIZE", "BM", "AG", "OP", "MOM"]),
    ("(3)", ["SUE", "AG", "MOM", "ILLIQ", "OP", "IVOL", "BETA", "SIZE", "BM", "MAX", "TURN", "STR"]),
]
PAPER_FM = {"(1)": -0.35, "(2)": -0.31, "(3)": -0.24}  # package results/tab6.tex


def factor_models():
    ff = tm.ff_factors()
    hxz = pd.read_csv(EXT / "q5_factors_monthly_2021.csv")
    hxz.index = pd.PeriodIndex([f"{y}-{m:02d}" for y, m in zip(hxz.year, hxz.month)], freq="M")
    hxz = hxz.drop(columns=["year", "month", "R_F"]) / 100
    sy = pd.read_csv(EXT / "stambaugh.csv")
    sy.index = pd.PeriodIndex(sy.pop("YYYYMM").astype(str), freq="M")
    sy = sy.drop(columns=["RF"])
    dhs = pd.read_csv(EXT / "DHS.csv")
    dhs.index = pd.PeriodIndex(dhs.pop("Date").astype(str), freq="M")
    dhs = (dhs / 100).join(ff[["Mkt-RF"]], how="inner")
    return {"CAPM": ff[["Mkt-RF"]], "FF6": ff, "HXZ": hxz, "SY": sy, "DHS": dhs}


def alphas(series, models):
    out = {}
    for name, f in models.items():
        f = f.copy()
        f.index = f.index - 1  # factor month t+1 aligned with formation month t
        a, t = tm.nw_alpha(series, f * 100)
        out[name] = (a, t)
    return out


def port_returns(df, port_cols, w):
    if w == "ew":
        r = df.groupby(["month"] + port_cols)["ret_exc_lead1m"].mean()
    else:
        g = df.assign(rw=df.ret_exc_lead1m * df.me)
        r = g.groupby(["month"] + port_cols)["rw"].sum() / g.groupby(["month"] + port_cols)["me"].sum()
    return r * 100


def dense_deciles(s, n):
    pct = s.rank(method="dense", pct=True)
    pct = pct.where(pct != pct.min(), 1e-8)
    return np.ceil(pct * n).clip(lower=1).astype(int)


def table3(df, col, models, cond=None, n1=5, n2=10):
    """Univariate (cond=None) or dependent bivariate sort (cond quintiles, col deciles within)."""
    df = df.dropna(subset=[col] + ([cond] if cond else [])).copy()
    if cond:
        df["p1"] = df.groupby("month")[cond].transform(lambda s: dense_deciles(s, n1))
        df["port"] = df.groupby(["month", "p1"])[col].transform(lambda s: dense_deciles(s, n2))
    else:
        df["port"] = df.groupby("month")[col].transform(lambda s: dense_deciles(s, n2))
    res = {}
    for w in ("ew", "vw"):
        if cond:
            r = port_returns(df, ["p1", "port"], w).unstack("port")
            r = r.groupby(level="month").mean()  # average across cond quintiles
        else:
            r = port_returns(df, ["port"], w).unstack("port")
        r["H-L"] = r[n2] - r[1]
        rows = {}
        for p in list(range(1, n2 + 1)) + ["H-L"]:
            m, t = tm.nw_mean(r[p])
            row = {"XRet": m, "t": t}
            for k, (a, ta) in alphas(r[p], models).items():
                row[k], row["t_" + k] = a, ta
            rows[p] = row
        res[w] = pd.DataFrame(rows).T
    return res


def fama_macbeth_paper(df, xcols):
    d = df[["permno", "month", "ret_exc_lead1m"] + xcols].dropna().copy()
    for c in xcols:
        lo, hi = d[c].quantile([0.005, 0.995])
        d[c] = d[c].clip(lo, hi)
        g = d.groupby("month")[c]
        d[c] = (d[c] - g.transform("mean")) / g.transform("std")
    d["date"] = d["month"].dt.to_timestamp("M")
    d = d.set_index(["permno", "date"]).sort_index()
    y = d["ret_exc_lead1m"] * 100
    X = d[xcols].assign(Const=1.0)
    res = FamaMacBeth(y, X).fit(cov_type="kernel", kernel="newey-west")
    return res


def stars(t):
    return "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.645 else ""


def fmt_t3(tab, label, say):
    say(f"\n{label}")
    cols = ["XRet", "CAPM", "FF6", "HXZ", "SY", "DHS"]
    say(f"{'':6s}" + "".join(f"{c:>17s}" for c in cols))
    for idx, row in tab.iterrows():
        name = {1: "Low", 10: "High"}.get(idx, str(idx))
        tname = lambda c: "t" if c == "XRet" else "t_" + c
        say(f"{name:6s}" + "".join(
            f"{row[c]:+.2f}{stars(row[tname(c)]):<3s}({row[tname(c)]:+.2f})".rjust(17) for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="ridge")
    ap.add_argument("--end", default="2022-11", help="last formation month (paper: 2022-11)")
    args = ap.parse_args()
    out = ROOT / "results/paper_tables" / args.variant
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    say = lambda s="": (print(s, flush=True), lines.append(s))

    p = tm.load_panel(VARIANTS[args.variant])
    p = p[p.month <= pd.Period(args.end, "M")]
    for k, v in tm.CONTROLS.items():
        p[k] = p[v]
    models = factor_models()
    say(f"MFD variant: {args.variant} ({VARIANTS[args.variant].relative_to(ROOT)}), "
        f"{p.month.min()}..{p.month.max()}, {len(p):,} stock-months")
    say("Factor windows: " + ", ".join(f"{k} to {v.index.max()}" for k, v in models.items()))

    # ---------------- Table 3 style
    for title, cond in [("TABLE 3 style: univariate MFD decile sorts", None),
                        ("TABLE 3 + mean: MFD deciles within mean-forecast quintiles (5x10 dependent sort, "
                         "averaged across quintiles)", "mean")]:
        say("\n" + "=" * 100 + f"\n{title}\n(% per month; t-stats in parentheses; *** 1%, ** 5%, * 10%)")
        t3 = table3(p, "std", models, cond=cond)
        for w, lab in [("ew", "Panel A: Equal-Weighted"), ("vw", "Panel B: Value-Weighted")]:
            fmt_t3(t3[w], lab, say)
            t3[w].to_csv(out / f"table3{'_mean' if cond else ''}_{w}.csv")

    # ---------------- Table 6 style
    say("\n" + "=" * 100 + "\nTABLE 6 style: Fama-MacBeth (% per month per 1 SD; NW t-stats; paper's estimator)")
    rows = []
    for with_mean in (False, True):
        for spec_name, ctrls in FM_SPECS:
            x = ["std"] + (["mean"] if with_mean else []) + ctrls
            res = fama_macbeth_paper(p, x)
            for v in ["std", "mean"] + ctrls + ["Const"]:
                if v in res.params:
                    rows.append({"with_mean": with_mean, "spec": spec_name, "var": v,
                                 "coef": res.params[v], "t": res.tstats[v],
                                 "N_obs": int(res.nobs), "months": int(res.time_info["total"])})
    T = pd.DataFrame(rows)
    T.to_csv(out / "table6.csv", index=False)
    for with_mean in (False, True):
        say(f"\n{'Panel B: with the mean forecast' if with_mean else 'Panel A: as in the paper'}")
        sub = T[T.with_mean == with_mean]
        say(f"{'':8s}" + "".join(f"{s:>18s}" for s, _ in FM_SPECS))
        order = ["std", "mean"] + FM_SPECS[-1][1] + ["Const"]
        for v in order:
            cells = []
            for s, _ in FM_SPECS:
                r = sub[(sub.spec == s) & (sub["var"] == v)]
                cells.append("" if r.empty else f"{r.coef.iloc[0]:+.2f}{stars(r.t.iloc[0]):<3s}({r.t.iloc[0]:+.2f})")
            if any(cells):
                say(f"{'MFD' if v == 'std' else 'MEAN' if v == 'mean' else v:8s}" + "".join(f"{c:>18s}" for c in cells))
        n = sub.groupby("spec").N_obs.first()
        say(f"{'Obs':8s}" + "".join(f"{n[s]:>18,}" for s, _ in FM_SPECS))
        if not with_mean:
            say(f"{'paper':8s}" + "".join(f"{'MFD ' + format(PAPER_FM[s], '+.2f'):>18s}" for s, _ in FM_SPECS))
    (out / "tables.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}/tables.txt and CSVs")


if __name__ == "__main__":
    main()
