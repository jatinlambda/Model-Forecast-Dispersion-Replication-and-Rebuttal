"""Fraction of machine investors predicting an up move, in Fama-MacBeth
regressions with the mean forecast and the paper's 12 controls.

FRAC_UP_{i,t}  = share of the K forecasts of next-month excess return that are > 0
FRAC_REL_{i,t} = share of the K forecasts above that month's cross-sectional
                 median of the mean forecast (robustness: in bearish months nearly
                 every forecast can be < 0, leaving FRAC_UP with little variation)

Set-up follows programs/02_analysis/analysis_fama_macbeth.py: pooled
winsorization at 0.5/99.5 for regressors, monthly cross-sectional standardization,
returns in %, Newey-West (6 lags, as in the proposal).

Usage:
  python scripts/frac_up_fmb.py [mfd_dir] [out_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels import FamaMacBeth

mfd_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/mfd_full")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "results/frac_up")
out.mkdir(parents=True, exist_ok=True)

# %% 1. FRAC_UP from per-investor forecasts
cache = out / "frac_up.pq"
if cache.exists():
    fu = pd.read_parquet(cache)
else:
    parts = []
    for f in sorted(mfd_dir.glob("month=*_forecasts.pq")):
        d = pd.read_parquet(f)
        P = d.filter(like="inv_").to_numpy(dtype=np.float64)
        mean = P.mean(axis=1)
        thr = np.median(mean)
        parts.append(pd.DataFrame({
            "date": d["date"].values, "permno": d["permno"].values,
            "frac_up": (P > 0).mean(axis=1),
            "frac_rel": (P > thr).mean(axis=1),
            "n_inv": P.shape[1],
        }))
    fu = pd.concat(parts, ignore_index=True)
    fu.to_parquet(cache, index=False)
fu["date"] = pd.to_datetime(fu["date"]).dt.to_period("M").dt.to_timestamp("M")

q = pd.read_parquet(mfd_dir / "mfd_quantiles.pq", columns=["date", "permno", "mean", "std"])
q["date"] = pd.to_datetime(q["date"]).dt.to_period("M").dt.to_timestamp("M")
df = fu.merge(q.rename(columns={"mean": "mfd_mean", "std": "mfd"}), on=["date", "permno"])

# %% 2. Returns and controls (same JKP columns as the paper's mapper)
CTRL = {"SUE": "niq_su", "AG": "at_gr1", "MOM": "ret_12_1", "ILLIQ": "ami_126d",
        "OP": "ope_be", "IVOL": "ivol_capm_252d", "BETA": "beta_60m",
        "SIZE": "market_equity", "BM": "be_me", "MAX": "rmax5_21d",
        "TURN": "turnover_126d", "STR": "ret_1_0"}
chars = pd.read_parquet("data/derived/JKP_chars.pq", columns=list(CTRL.values())).reset_index()
ret = pd.read_parquet("data/derived/RET.pq").reset_index()
for x in (chars, ret):
    x["date"] = pd.to_datetime(x["date"].astype(str), format="%Y%m%d").dt.to_period("M").dt.to_timestamp("M")
df = df.merge(ret, on=["date", "permno"]).merge(chars, on=["date", "permno"])
df = df.dropna(subset=["ret_exc_lead1m"])

# %% 3. Diagnostics on FRAC_UP
diag = df.groupby("date").agg(frac_up_mean=("frac_up", "mean"), frac_up_sd=("frac_up", "std"),
                              share_all_down=("frac_up", lambda s: (s == 0).mean()),
                              share_all_up=("frac_up", lambda s: (s == 1).mean()))
corr = df.groupby("date")[["frac_up", "frac_rel", "mfd_mean", "mfd"]].corr(method="spearman")
corr_avg = corr.groupby(level=1).mean().loc[["frac_up", "frac_rel", "mfd_mean", "mfd"]]

# %% 4. Winsorize + standardize, as in the paper
REGS = ["frac_up", "frac_rel", "mfd_mean", "mfd"] + list(CTRL.values())
for c in REGS:
    lo, hi = df[c].quantile([0.005, 0.995])
    df[c] = df[c].clip(lo, hi)
    df[c] = df.groupby("date")[c].transform(lambda x: (x - x.mean()) / x.std())
df["mean_sq"] = df["mfd_mean"] ** 2   # nonlinear-in-mean controls (frac_rel ~ mean: rho 0.99)
df["mean_cu"] = df["mfd_mean"] ** 3
df["y"] = df["ret_exc_lead1m"] * 100
df = df.set_index(["permno", "date"]).sort_index()

ctrls = list(CTRL.values())
SPECS = {
    "(1) FRAC_UP": ["frac_up"],
    "(2) + MEAN": ["frac_up", "mfd_mean"],
    "(3) + MEAN + 12 ctrls": ["frac_up", "mfd_mean"] + ctrls,
    "(4) + MEAN + MFD + 12 ctrls": ["frac_up", "mfd_mean", "mfd"] + ctrls,
    "(5a) MFD + 12 ctrls [paper Tab 6]": ["mfd"] + ctrls,
    "(5b) MEAN + 12 ctrls": ["mfd_mean"] + ctrls,
    "(5) MFD + MEAN + 12 ctrls [baseline]": ["mfd", "mfd_mean"] + ctrls,
    "(6) FRAC_REL + MEAN + 12 ctrls": ["frac_rel", "mfd_mean"] + ctrls,
    "(7) FRAC_REL + MEAN + MFD + 12 ctrls": ["frac_rel", "mfd_mean", "mfd"] + ctrls,
    "(8) = (7) + MEAN^2 + MEAN^3": ["frac_rel", "mfd_mean", "mean_sq", "mean_cu", "mfd"] + ctrls,
    "(9) = (4) + MEAN^2 + MEAN^3": ["frac_up", "mfd_mean", "mean_sq", "mean_cu", "mfd"] + ctrls,
}
SHOW = ["frac_up", "frac_rel", "mfd_mean", "mfd"]
rows = []
for name, xs in SPECS.items():
    d = df[xs + ["y"]].dropna()
    X = d[xs].assign(const=1.0)
    res = FamaMacBeth(d["y"], X).fit(cov_type="kernel", kernel="newey-west", bandwidth=6)
    r = {"spec": name, "n_obs": int(res.nobs), "n_months": d.index.get_level_values("date").nunique(),
         "avg_r2": res.rsquared}
    for v in SHOW:
        if v in xs:
            r[f"{v}_b"], r[f"{v}_t"] = res.params[v], res.tstats[v]
    rows.append(r)
tab = pd.DataFrame(rows)

tab.to_csv(out / "fmb.csv", index=False)
diag.to_csv(out / "frac_up_monthly_diag.csv")
corr_avg.to_csv(out / "spearman_avg.csv")

pd.set_option("display.width", 250, "display.max_columns", 30)
print(f"Sample: {len(df):,} stock-months, {df.index.get_level_values('date').nunique()} months, "
      f"{df.index.get_level_values('date').min():%Y-%m} to {df.index.get_level_values('date').max():%Y-%m}\n")
print("FRAC_UP raw distribution:", fu["frac_up"].describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_dict())
print(f"Months where >50% of stocks have FRAC_UP==0: {(diag.share_all_down > .5).sum()}; "
      f"==1: {(diag.share_all_up > .5).sum()}; median monthly cross-sectional sd of FRAC_UP: {diag.frac_up_sd.median():.3f}\n")
print("Avg monthly Spearman correlations:\n", corr_avg.round(3).to_string(), "\n")
print("Fama-MacBeth (y = next-month excess return in %, regressors standardized; coef = % per 1 sd):")
print(tab.round(3).to_string(index=False))
