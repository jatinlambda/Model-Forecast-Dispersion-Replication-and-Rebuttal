"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Analyses: bivariate portfolio sorts
"""
import pandas as pd
import numpy as np

from pathlib import Path

def get_project_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2]
    else:
        # interactive / notebook / IDE execution
        return Path.cwd().resolve().parents[1]

ROOT = get_project_root()

DATA_RAW = ROOT / "data" / "raw"
DATA_DERIVED = ROOT / "data" / "derived"
DATA_EXTERNAL = ROOT / "data" / "external"
DATA_MFD = ROOT / "data" / "mfd"  
DATA_RESULTS = "results/"

from helper import (do_portfolio_sort, 
                    average_char_decile, 
                    bivariate_sort, 
                    bivariate_sort_to_latex, 
                    bivariate_sort_to_latex_no_avg, 
                    get_factors, 
                    savepath, 
                    pfs, 
                    bp_min_n, 
                    bps, 
                    get_green, 
                    get_misp, 
                    get_norm_chars, 
                    get_ior)

num_format="%.2f"

# %% Load Data
mfd = pd.read_parquet(DATA_MFD / "main_data.pq")

ap_factors = get_factors()
mfd = mfd.sort_values(["date","permno"])

# %% AFD
pfs_bivariate = (5,pfs)
bps_bivariate = (bps, bps)
bp_min_n_bivariate = (5,bp_min_n)

green = get_green()
mfd_afd = mfd.merge(green, on = ["date","permno"], how ="left")
mfd_afd = mfd_afd[mfd_afd.date > "1982-05-31"]
mfd_afd = mfd_afd[mfd_afd.date <= "2022-03-31"]
mfd_afd = do_portfolio_sort(mfd_afd.copy(), "mfd", "mfd_port", 10, bps, bp_min_n)
mfd_afd = mfd_afd.drop(columns = ["cdf","min_cdf","bp_stock","bp_n"])

mfd_afd = mfd_afd.dropna(subset = "afd")
panelA = average_char_decile(mfd_afd, "afd", "AFD", agg_type = lambda x: np.nanmean(x))

panelB, starsB = bivariate_sort(mfd_afd.copy(), "ew", 
                                           ("afd","mfd"), ap_factors,
                                           pfs_bivariate, bps_bivariate, 
                                           bp_min_n_bivariate, typ = "dependent",
                                           include_alphas=["FF6"],
                                           abbrev = "AFD")

bivariate_sort_to_latex(panelA, panelB, None, starsB, None, DATA_RESULTS + "tab7", num_format="%.2f")

# %% Now add quintile portfolio information as we will do 3x5 sorts
mfd = do_portfolio_sort(mfd.copy(), "mfd", "mfd_port", 5, bps, bp_min_n)
mfd = mfd.drop(columns = ["cdf","min_cdf","bp_stock","bp_n"])
pfs_bivariate = (3,5)
bps_bivariate = (bps, bps)
bp_min_n_bivariate = (3,5)

# %% MISPricing measure of SYY
misp = get_misp()
mfd_misp = mfd.merge(misp, on = ["date","permno"], how = "left")

mfd_misp = mfd_misp.dropna(subset = "MISP")
panelA = average_char_decile(mfd_misp, "MISP", "MISP", agg_type = lambda x: np.nanmean(x))
panelB, starsB = bivariate_sort(mfd_misp.copy(), "ew", 
                                           ("MISP","mfd"), ap_factors,
                                           pfs_bivariate, bps_bivariate, 
                                           bp_min_n_bivariate, typ = "dependent",
                                           include_alphas=["FF6"],
                                           abbrev = "MISP")

bivariate_sort_to_latex(panelA, panelB, None, starsB, None, DATA_RESULTS +  "tab13", num_format="%.2f")

# %% OOS RMSE
rmse = mfd[["permno","date","mfd_mean","ret_exc_lead1m"]].copy()
rmse["errr"] = (rmse["mfd_mean"]-rmse["ret_exc_lead1m"])**2

rmse = rmse.sort_values(["permno","date"])
rmse = rmse.set_index("date")
rmse = rmse.groupby("permno").errr.rolling(24, min_periods = 12).mean()
rmse = rmse.groupby("permno").shift(1)

mfd_rmse = mfd.merge(rmse.reset_index(), on = ["date","permno"])
mfd_rmse = mfd_rmse.dropna(subset = "errr")
panelB, starsB = bivariate_sort(mfd_rmse.copy(), "ew", 
                                           ("errr","mfd"), ap_factors,
                                           pfs_bivariate, bps_bivariate, 
                                           bp_min_n_bivariate, typ = "dependent",
                                           include_alphas=["FF6"],
                                           abbrev = "RMSE OOS")

bivariate_sort_to_latex_no_avg(panelB, starsB, pfs_bivariate[1], DATA_RESULTS + "ia_tab_f1", num_format="%.2f")

# %% Markit Short-selling data
markit = pd.read_parquet(DATA_RAW  / "markit_shortfee.pq")
markit = markit.rename(columns = {"ym": "date"})

mfd_markit = pd.merge(mfd, markit, on = ["date","permno"], how = "left")
mfd_markit = mfd_markit[mfd_markit.date >= "2004-01-01"]
mfd_markit = mfd_markit[mfd_markit.date <= markit.date.max()]
mfd_markit_ifee = mfd_markit.dropna(subset = "IndicativeFee").copy()
mfd_markit_ifee.IndicativeFee *=100

panelA = average_char_decile(mfd_markit_ifee.copy(), "IndicativeFee", "BORROWFEE", agg_type = lambda x: np.nanmean(x))
panelB, starsB = bivariate_sort(mfd_markit_ifee.copy(), "ew", 
                                            ("IndicativeFee","mfd"), ap_factors,
                                            pfs_bivariate, bps_bivariate, 
                                            bp_min_n_bivariate, typ = "dependent",
                                            include_alphas=["FF6"],
                                            abbrev = "BORROWFEE")

bivariate_sort_to_latex(panelA, panelB, None, starsB, None, DATA_RESULTS +  "bivariate_IFEE", num_format="%.2f")

# %% Limits to arb index
norm_chars = get_norm_chars("unnorm")
arb = pd.merge(mfd, norm_chars, on = ["permno","date"])
arb = arb.set_index(["date","permno"])

char = ["-market_equity", "ivol_capm_252d", "-age"]

contraints = []
for c in char:
    if c[0] == "-":
        sign = -1
        c = c[1:]
    else:
        sign = 1
    tmp = arb[[c]].copy()
    tmp[c] = sign * tmp[c]
    tmp = tmp.sort_index()
    tmp["rank_"+c] = tmp.groupby("date")[c].transform(lambda x: pd.qcut(x, 10, labels = False, duplicates="drop")) + 1
    tmp = tmp[["rank_"+c]]
    contraints.append(tmp)
contraints = pd.concat(contraints, axis = 1)
contraints["arb"] = contraints.sum(axis = 1)
arb = pd.merge(contraints, arb, left_index = True, right_index = True)
arb = arb.reset_index()

panelA = average_char_decile(arb, "arb", "ARB", agg_type = lambda x: np.nanmean(x))
panelB, starsB = bivariate_sort(arb.copy(), "ew", 
                                           ("arb","mfd"), ap_factors,
                                           pfs_bivariate, bps_bivariate, 
                                           bp_min_n_bivariate, typ = "dependent",
                                           include_alphas=["FF6"],
                                           abbrev = "ARB")

bivariate_sort_to_latex(panelA, panelB, None, starsB, None, DATA_RESULTS +  "tab16", num_format="%.2f")

# %% Institutional Ownership
ior = get_ior()

mfd_ior = mfd.merge(ior, on = ["permno","date"], how = "left")

panelA = average_char_decile(mfd_ior, "ior", "INST", agg_type = lambda x: np.nanmean(x))
panelB, starsB = bivariate_sort(mfd_ior.copy(), "ew", 
                                           ("ior","mfd"), ap_factors,
                                           pfs_bivariate, bps_bivariate, 
                                           bp_min_n_bivariate, typ = "dependent",
                                           include_alphas=["FF6"],
                                           abbrev = "INST")

bivariate_sort_to_latex(panelA, panelB, None, starsB, None, DATA_RESULTS +  "bivariate_IOR", num_format="%.2f")