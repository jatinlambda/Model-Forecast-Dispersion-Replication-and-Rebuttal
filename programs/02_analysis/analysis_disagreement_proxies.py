"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Analyses: Comparison to disagreement proxies
"""

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

import pandas as pd
import numpy as np
import statsmodels.api as sm
from helper import get_norm_chars, get_green, sig_stars, get_short_interest
from linearmodels import FamaMacBeth
from pylatex import (
    Tabular,
    MultiColumn,
)

num_format="%.2f"

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

cm = 1 / 2.54
width = 15.92 * cm
height = 8 * cm

matplotlib.rcParams.update({"font.size": 8})
matplotlib.rcParams["axes.xmargin"] = 0.02
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.edgecolor"] = "0.15"
matplotlib.rcParams["axes.linewidth"] = 1.25

# %% load data
mfd = pd.read_parquet(DATA_MFD / "main_data.pq")

norm_chars = get_norm_chars("unnorm")
proxy_cols = ["rvol_21d", "ivol_capm_252d"]
norm_chars = norm_chars[["date", "permno"] + proxy_cols]
mfd = pd.merge(mfd, norm_chars, on = ["permno","date"])

# %% other disagreement proxies

# analyst forecast dispersion
green = get_green()
mfd = mfd.merge(green, on = ["date","permno"], how ="left")

# Marina Niessner sentiment
social = pd.read_csv(DATA_EXTERNAL / "CooksonNiessner20203_Disagreement.txt", delimiter= "\t")
social.date = pd.to_datetime(social.date, format = "%d%b%Y")
social.date = social.date.dt.to_period("M")
social = social.groupby(["date","permno"]).mean()
social = social.reset_index()
social.date = social.date.dt.to_timestamp("M")
mfd = mfd.merge(social, on = ["date", "permno"], how = "left")
proxy_cols.extend(["dis_all", "dis_within_group", "dis_across_group"])

# monthly turnover
monthly_to = pd.read_pickle(DATA_DERIVED / "monthly_turnover.pkl.gz")[["date","permno","turnover_21d"]]
mfd = mfd.merge(monthly_to, on = ["date", "permno"], how = "left")
proxy_cols.append("turnover_21d")

# SUV monthly or 3-month with average of daily in the last month
suv_all = pd.read_parquet(DATA_DERIVED / "suv.pq")
mfd = mfd.merge(suv_all, on = ["date","permno"], how = "left")
proxy_cols.append("suv_m")

# option disagreement
option_dis = pd.read_parquet(DATA_RAW / "option_disagreement_single_stocks.pq")
option_dis["date"] = option_dis.date.dt.to_period("M").dt.to_timestamp("M")
option_dis = option_dis.groupby(["date","permno"]).median().reset_index()
mfd = mfd.merge(option_dis, on = ["date", "permno"], how = "left")
proxy_cols.append("option_disagreement")

# expected idiosyncratic skewness
exp_iskew  = pd.read_csv(DATA_EXTERNAL / "predskew.txt", header = None, sep = " ")
exp_iskew.columns = ["date", "permno", "exp_iskew", "exp_skew"]
exp_iskew.date = pd.to_datetime(exp_iskew.date, format = "%Y%m").dt.to_period("M").dt.to_timestamp("M")
mfd = mfd.merge(exp_iskew, on = ["date","permno"], how = "left")
proxy_cols.append("exp_iskew")

# ratio of newly issued forecasts to outstanding forecasts
numest = pd.read_parquet(DATA_RAW / "numest_ratio.pq")
mfd = mfd.merge(numest, on = ["date","permno"], how = "left")
proxy_cols.append("ratio_analyst_issue")

# RSI for disagreement index
rsi = get_short_interest()
mfd = mfd.merge(rsi, on = ["date","permno"], how = "left")

mfd_xs = mfd[mfd.date > "1982-12-31"].copy()

xs_res = []
for proxy in proxy_cols:
        
    tmp_mfd = mfd_xs[["mfd", "date"] + [proxy]].dropna().copy()
    tmp_afd = mfd_xs[["afd", "date"] + [proxy]].dropna().copy()
    
    tmp_mfd = tmp_mfd.groupby("date").apply(lambda x: x.corr(method = "spearman").loc["mfd",proxy]).dropna()
    tmp_afd = tmp_afd.groupby("date").apply(lambda x: x.corr(method = "spearman").loc["afd",proxy]).dropna()
    
    tmp = tmp_mfd.to_frame(name="MFD").merge(tmp_afd.to_frame(name = "AFD"), left_index = True, right_index = True)
    #tmp.plot()
    tmp["Difference"] = tmp.MFD - tmp.AFD
    
    res = pd.Series()
    for col in tmp.columns:
        t_res = sm.OLS(tmp[col], np.ones(tmp[col].shape)).fit(cov_type = "HAC", cov_kwds = {"maxlags": 6})
        res[col] = (t_res.params.iloc[0], t_res.tvalues.iloc[0])
    res = res.to_frame(name = proxy).T
    xs_res.append(res)
xs_res = pd.concat(xs_res)

proxy_col_renamer = {
    "ivol_capm_252d": "IVOL",
    "rvol_21d": "HV",
    "dis_all": "StockTwits Disagreement",
    "dis_within_group": "StockTwits Within Group Disagreement",
    "dis_across_group": "StockTwits Across Group Disagreement",
    "turnover_21d": "Last Month Turnover",
    "option_disagreement": "Option Disagreement",
    "exp_iskew": "Expected Idiosyncratic Volatility",
    "suv_m": "Standardized Unexplained Volume",
    "ratio_analyst_issue": "New Analyst Issues"} 

def do_table(data, fname, num_format="%.2f"):
    ncol = data.shape[1]*2 + 1
    tabular = Tabular("l" + "c" * (ncol - 1), booktabs=True)
    colgroups = data.columns
    colnames = ["XS-Corr.", "t-stat"]
    to_add = [""]
    for colg in colgroups:
        to_add.append(MultiColumn(2, data = colg, align = "c"))
    tabular.add_row(to_add)
    tabular.add_hline(2,3, cmidruleoption = "lr")
    tabular.add_hline(4,5, cmidruleoption = "lr")
    tabular.add_hline(6,7, cmidruleoption = "lr")

    to_add = [""]
    to_add.extend(3*colnames)
    tabular.add_row(to_add)
    tabular.add_hline()
    
    for i, row in data.iterrows():
        if i in proxy_col_renamer.keys():
            i = proxy_col_renamer[i]
        
        to_add = [i]
        for c_g in colgroups:
            r1 = row[c_g][0]
            r2 = row[c_g][1]
            r1 = num_format % r1 + sig_stars(r2)
            r2 = "({})".format(num_format % r2)
            to_add.append(r1)
            to_add.append(r2)
        tabular.add_row(to_add)
    
    tabular.generate_tex(fname)
    print("table writen to {}".format(fname))

do_table(xs_res, DATA_RESULTS + "tab8")

# %% Disagreement index 
d_index_narrow = ["rvol_21d", "ivol_capm_21d","rsi","exp_iskew"]
proxy_cols = ["ivol_capm_21d"]
norm_chars = get_norm_chars("unnorm")[["date", "permno"] + proxy_cols]
mfd = pd.merge(mfd, norm_chars, on = ["permno","date"])

# MFD sample
mfd_dindex = mfd.copy()
mfd_dindex = mfd_dindex.sort_values(["date","permno"])
mfd_dindex = mfd_dindex[mfd_dindex.date >= "1978-01-31"]
mfd_dindex = mfd_dindex[mfd_dindex.date <= "2021-12-31"]

for i in d_index_narrow:
    mfd_dindex[f"{i}_rank"] = mfd_dindex.groupby("date")[i].transform(lambda x: pd.qcut(x, 10, labels = False, duplicates="drop"))+1
mfd_dindex["d_index"] = mfd_dindex[[c+"_rank" for c in d_index_narrow]].mean(axis = 1)

mfd_dindex["mfd_rank"] = mfd_dindex.groupby("date")["mfd"].transform(lambda x: pd.qcut(x, 10, labels = False, duplicates="drop"))+1

# AFD sample
mfd_dindex_afd = mfd.dropna(subset = "afd").copy()
mfd_dindex_afd = mfd_dindex_afd.sort_values(["date","permno"])
mfd_dindex_afd = mfd_dindex_afd[mfd_dindex_afd.date > "1982-12-31"]
mfd_dindex_afd = mfd_dindex_afd[mfd_dindex_afd.date <= "2021-12-31"]

for i in d_index_narrow:
    mfd_dindex_afd[f"{i}_rank"] = mfd_dindex_afd.groupby("date")[i].transform(lambda x: pd.qcut(x, 10, labels = False, duplicates="drop"))+1
mfd_dindex_afd["d_index"] = mfd_dindex_afd[[c+"_rank" for c in d_index_narrow]].mean(axis = 1)

# %% Fama-MacBeth decomp

def wins(x, bds):
    q = x.quantile(bds)
    x.loc[x<=q.iloc[0]] = q.iloc[0]
    x.loc[x>=q.iloc[1]] = q.iloc[1]
    return x

flag_winsorize = True
winsorize_levels = [0.005, 0.995]

def decompose_effect_multi(dat, to_explain, candidates, ret_def):
    
    data = dat.dropna(subset = [to_explain, ret_def] + candidates)[[to_explain, ret_def] + candidates].copy()
    if flag_winsorize:
        cols_to_win = [to_explain] + candidates
        for c in cols_to_win:
            data[c] = wins(data[c], winsorize_levels)
            
    data["Const"] = 1

    data = data.sort_index()
    
    data_ret = data[ret_def] *100
    data_explain = data[[to_explain, "Const"]]
    data_candidates = data[ candidates + ["Const"]]
    
    mod = FamaMacBeth(data_ret, data_explain)
    res_1 = mod.fit(cov_type="kernel", kernel="newey-west")
    gamma = res_1.all_params[to_explain]
    
    T = res_1.time_info["total"]

    mod = FamaMacBeth(data_explain[to_explain], data_candidates)
    res_2 = mod.fit(cov_type="kernel", kernel="newey-west")
    
    data_all = data_ret.reset_index()
    data_all = data_all.merge(res_2.all_params.rename(columns = {x: x + "_param" for x in res_2.all_params.columns}), 
                              left_on = "date", right_index = True)
    data_all = pd.merge(data_all, data_candidates[candidates], left_on = ["permno","date"], right_index = True)
    data_all = pd.merge(data_all, data_explain[[to_explain]], left_on = ["permno","date"], right_index = True)
    
    data_all["candidate_part"] = np.sum(data_all[[c + "_param" for c in candidates]].values * data_all[candidates].values, axis = 1)
    data_all["resid_part"] = data_all[to_explain] - data_all["candidate_part"] 
    
    #data_all["u"] = data_all[to_explain] - data_all["Const_param"] - data_all[candidate+"_param"] * data_all[candidate]
    
    gamma_R = data_all.groupby("date").apply(lambda x: x[[ret_def,"resid_part"]].cov().iloc[0,1] / x[to_explain].var())
    gammas = pd.merge(gamma_R.to_frame(name = "gamma_R"), gamma.to_frame(name = "gamma"), left_index = True, right_index = True)

    for c in candidates:
        data_all[f"{c}_part"] = data_all[f"{c}_param"].values * data_all[f"{c}"].values
        gamma_c = data_all.groupby("date").apply(lambda x: x[[ret_def,f"{c}_part"]].cov().iloc[0,1] / x[to_explain].var())
        gammas = pd.merge(gammas, gamma_c.to_frame(name = f"gamma_{c}"), left_index = True, right_index = True)
    
    stage_2 = []

    for c in candidates:
        ratio = gammas[f"gamma_{c}"].mean() / gammas.gamma.mean()
        ratio_var = 1 / T * ratio**2 * (gammas[f"gamma_{c}"].var() / gammas[f"gamma_{c}"].mean() **2 +   gammas.gamma.var() / gammas.gamma.mean() **2  
                                                            - 2 * gammas[[f"gamma_{c}","gamma"]].cov().iloc[0,1] /    gammas[f"gamma_{c}"].mean() /  gammas.gamma.mean()  
                                                            )
        ratio_t = ratio / np.sqrt(ratio_var)
    
        stage_2.append(pd.DataFrame({
            "control": c,
            "ratio": ratio,
            "ratio_var": ratio_var,
            "ratio_t": ratio_t}, index = [0]))
        
    ratio_residual = gammas.gamma_R.mean() / gammas.gamma.mean()
    ratio_residual_var = 1 / T * ratio_residual**2 * (gammas.gamma_R.var() / gammas.gamma_R.mean() **2 +   gammas.gamma.var() / gammas.gamma.mean() **2  
                                                - 2 * gammas[["gamma_R","gamma"]].cov().iloc[0,1] /    gammas.gamma_R.mean() /  gammas.gamma.mean()      )
    
    ratio_residual_t = ratio_residual / np.sqrt(ratio_residual_var)

    stage_2.append(pd.DataFrame({
        "control": "residual",
        "ratio": ratio_residual,
        "ratio_var": ratio_residual_var,
        "ratio_t": ratio_residual_t}, index = [0]))

    stage_2 = pd.concat(stage_2)
    stage_2["to_explain"] = to_explain
    
    return res_1, stage_2

# copy data
mfd_dindex_mfd_sample = mfd_dindex.dropna(subset = ["d_index"]).copy()
mfd_dindex_afd = mfd_dindex_afd.dropna(subset = ["d_index","afd"]).copy()

# do decomposition on single stocks
stage_1_mfd_only, stage_2_mfd_only = decompose_effect_multi(mfd_dindex_mfd_sample.copy().set_index(["permno","date"]).sort_index(),
                                          to_explain = "d_index", candidates = ["mfd"],
                                          ret_def = "ret_exc_lead1m"
                                          )

stage_1_mfd, stage_2_mfd = decompose_effect_multi(mfd_dindex_afd.copy().set_index(["permno","date"]).sort_index(),
                                          "d_index", candidates = ["mfd"],
                                          ret_def = "ret_exc_lead1m"
                                          )

_, stage_2_afd = decompose_effect_multi(mfd_dindex_afd.copy().set_index(["permno","date"]).sort_index(),
                                          "d_index", candidates = ["afd"],
                                          ret_def = "ret_exc_lead1m"
                                          )

_, stage_2_afd_mfd = decompose_effect_multi(mfd_dindex_afd.copy().set_index(["permno","date"]).sort_index(),
                                          "d_index", candidates = ["mfd","afd"],
                                          ret_def = "ret_exc_lead1m"
                                          )

# copy data and do decomposition on 50 portfolios sorted on DIS 
mfd_dindex_mfd_pf = mfd_dindex_mfd_sample.copy()
mfd_dindex_mfd_pf["dis_rank"] = mfd_dindex_mfd_pf.groupby("date")["d_index"].transform(lambda x: pd.qcut(x, 50, labels = False, duplicates="drop"))+1
mfd_dindex_mfd_pf = mfd_dindex_mfd_pf.groupby(["date","dis_rank"])[["mfd","afd","ret_exc_lead1m","d_index"]].mean().reset_index().rename(columns = {"dis_rank":"permno"})


mfd_dindex_afd_pf = mfd_dindex_afd.dropna(subset = ["d_index","afd"]).copy()
mfd_dindex_afd_pf["dis_rank"] = mfd_dindex_afd_pf.groupby("date")["d_index"].transform(lambda x: pd.qcut(x, 50, labels = False, duplicates="drop"))+1
mfd_dindex_afd_pf = mfd_dindex_afd_pf.groupby(["date","dis_rank"])[["mfd","afd","ret_exc_lead1m","d_index"]].mean().reset_index().rename(columns = {"dis_rank":"permno"})


stage_1_mfd_only_pf, stage_2_mfd_only_pf = decompose_effect_multi(mfd_dindex_mfd_pf.copy().set_index(["permno","date"]).sort_index(),
                                          to_explain = "d_index", candidates = ["mfd"],
                                          ret_def = "ret_exc_lead1m"
                                          )

stage_1_mfd_pf, stage_2_mfd_pf = decompose_effect_multi(mfd_dindex_afd_pf.copy().set_index(["permno","date"]).sort_index(),
                                          "d_index", candidates = ["mfd"],
                                          ret_def = "ret_exc_lead1m"
                                          )

_, stage_2_afd_pf = decompose_effect_multi(mfd_dindex_afd_pf.copy().set_index(["permno","date"]).sort_index(),
                                          "d_index", candidates = ["afd"],
                                          ret_def = "ret_exc_lead1m"
                                          )

_, stage_2_afd_mfd_pf = decompose_effect_multi(mfd_dindex_afd_pf.copy().set_index(["permno","date"]).sort_index(),
                                          "d_index", candidates = ["mfd","afd"],
                                          ret_def = "ret_exc_lead1m"
                                          )

# do Table
fname = "tab9"
tabular = Tabular("l@{\hspace{2\\tabcolsep}}lc@{\hspace{0.5\\tabcolsep}}c@{\hspace{2\\tabcolsep}}lc@{\hspace{0.5\\tabcolsep}}c@{\hspace{2\\tabcolsep}}lc@{\hspace{0.5\\tabcolsep}}c@{\hspace{2\\tabcolsep}}lc@{\hspace{0.5\\tabcolsep}}c", booktabs=True)

# Panel A: Stocks
tabular.add_row([MultiColumn(13, data = "Panel A: Single Stocks", align = "l")])
tabular.add_hline()
colnames = ["", 
            MultiColumn(3, data = "DIS on Excess Returns", align = "c"),
            MultiColumn(3, data = "Decomposition wrt MFD", align = "c"),
            MultiColumn(3, data = "Decomposition wrt AFD", align = "c"),
            MultiColumn(3, data = "Multivariate Decomposition", align = "c")]
tabular.add_row(colnames)
tabular.add_hline(2,4, cmidruleoption = "r")
tabular.add_hline(5,7, cmidruleoption = "r")
tabular.add_hline(8,10, cmidruleoption = "r")
tabular.add_hline(11,13, cmidruleoption = "r")

# MFD sample
to_add = ["MFD Sample"]
to_add2 = [""]

# 1st stage
r1 = stage_1_mfd_only.params["Const"]
r2 = stage_1_mfd_only.tstats["Const"]
to_add.extend(["Const.",  num_format % r1 + sig_stars(r2), "({})".format(num_format % r2)])

r1 = stage_1_mfd_only.params["d_index"]
r2 = stage_1_mfd_only.tstats["d_index"]
to_add2.extend(["DIS",  num_format % r1 + sig_stars(r2), "({})".format(num_format % r2)])

# MFD Decomp
tmp = stage_2_mfd_only[stage_2_mfd_only.control == "mfd"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add.extend(["MFD",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

tmp = stage_2_mfd_only[stage_2_mfd_only.control == "residual"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add2.extend(["Residual",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

# finish up rest
to_add.extend(list(np.repeat("",6)))
to_add2.extend(list(np.repeat("",6)))

tabular.add_row(to_add)
tabular.add_row(to_add2)

# AFD sample
to_add = ["AFD Sample"]
to_add2 = [""]
to_add3 = list(np.repeat("",10))

# 1st stage
r1 = stage_1_mfd.params["Const"]
r2 = stage_1_mfd.tstats["Const"]
to_add.extend(["Const.",  num_format % r1 + sig_stars(r2), "({})".format(num_format % r2)])

r1 = stage_1_mfd.params["d_index"]
r2 = stage_1_mfd.tstats["d_index"]
to_add2.extend(["DIS",  num_format % r1 + sig_stars(r2), "({})".format(num_format % r2)])

# MFD Decomp
tmp = stage_2_mfd[stage_2_mfd.control == "mfd"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add.extend(["MFD",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

tmp = stage_2_mfd[stage_2_mfd.control == "residual"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add2.extend(["Residual",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

# AFD Decomp
tmp = stage_2_afd[stage_2_afd.control == "afd"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add.extend(["AFD",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

tmp = stage_2_afd[stage_2_afd.control == "residual"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add2.extend(["Residual",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

# Multivariate Decomp
tmp = stage_2_afd_mfd[stage_2_afd_mfd.control == "mfd"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add.extend(["MFD",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

tmp = stage_2_afd_mfd[stage_2_afd_mfd.control == "afd"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add2.extend(["AFD",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

tmp = stage_2_afd_mfd[stage_2_afd_mfd.control == "residual"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add3.extend(["Residual",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

tabular.add_row(to_add)
tabular.add_row(to_add2)
tabular.add_row(to_add3)

tabular.add_hline()
# Panel B: 50 Disagreement index sorted portfolios
tabular.add_row([MultiColumn(13, data = "Panel B: 50 Portfolios Sorted on DIS", align = "l")])
tabular.add_hline()
colnames = ["", 
            MultiColumn(3, data = "DIS on Excess Returns", align = "c"),
            MultiColumn(3, data = "Decomposition wrt MFD", align = "c"),
            MultiColumn(3, data = "Decomposition wrt AFD", align = "c"),
            MultiColumn(3, data = "Multivariate Decomposition", align = "c")]
tabular.add_row(colnames)
tabular.add_hline(2,4, cmidruleoption = "r")
tabular.add_hline(5,7, cmidruleoption = "r")
tabular.add_hline(8,10, cmidruleoption = "r")
tabular.add_hline(11,13, cmidruleoption = "r")

# MFD sample
to_add = ["MFD Sample"]
to_add2 = [""]

# 1st stage
r1 = stage_1_mfd_only_pf.params["Const"]
r2 = stage_1_mfd_only_pf.tstats["Const"]
to_add.extend(["Const.",  num_format % r1 + sig_stars(r2), "({})".format(num_format % r2)])

r1 = stage_1_mfd_only_pf.params["d_index"]
r2 = stage_1_mfd_only_pf.tstats["d_index"]
to_add2.extend(["DIS",  num_format % r1 + sig_stars(r2), "({})".format(num_format % r2)])

# MFD Decomp
tmp = stage_2_mfd_only_pf[stage_2_mfd_only_pf.control == "mfd"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add.extend(["MFD",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

tmp = stage_2_mfd_only_pf[stage_2_mfd_only_pf.control == "residual"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add2.extend(["Residual",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

# finish up rest
to_add.extend(list(np.repeat("",6)))
to_add2.extend(list(np.repeat("",6)))

tabular.add_row(to_add)
tabular.add_row(to_add2)

# AFD sample
to_add = ["AFD Sample"]
to_add2 = [""]
to_add3 = list(np.repeat("",10))

# 1st stage
r1 = stage_1_mfd_pf.params["Const"]
r2 = stage_1_mfd_pf.tstats["Const"]
to_add.extend(["Const.",  num_format % r1 + sig_stars(r2), "({})".format(num_format % r2)])

r1 = stage_1_mfd_pf.params["d_index"]
r2 = stage_1_mfd_pf.tstats["d_index"]
to_add2.extend(["DIS",  num_format % r1 + sig_stars(r2), "({})".format(num_format % r2)])

# MFD Decomp
tmp = stage_2_mfd_pf[stage_2_mfd_pf.control == "mfd"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add.extend(["MFD",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

tmp = stage_2_mfd_pf[stage_2_mfd_pf.control == "residual"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add2.extend(["Residual",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

# AFD Decomp
tmp = stage_2_afd_pf[stage_2_afd_pf.control == "afd"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add.extend(["AFD",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

tmp = stage_2_afd_pf[stage_2_afd_pf.control == "residual"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add2.extend(["Residual",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

# Multivariate Decomp
tmp = stage_2_afd_mfd_pf[stage_2_afd_mfd_pf.control == "mfd"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add.extend(["MFD",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

tmp = stage_2_afd_mfd_pf[stage_2_afd_mfd_pf.control == "afd"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add2.extend(["AFD",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

tmp = stage_2_afd_mfd_pf[stage_2_afd_mfd_pf.control == "residual"]
r1 = tmp.ratio.iloc[0]*100
r2 = tmp.ratio_t.iloc[0]
to_add3.extend(["Residual",  num_format % r1 + "%" + sig_stars(r2), "({})".format(num_format % r2)])

tabular.add_row(to_add)
tabular.add_row(to_add2)
tabular.add_row(to_add3)

tabular.generate_tex(DATA_RESULTS + fname)
print("table writen to {}".format(DATA_RESULTS + fname))

# %% Figure: Median Disagreement rank per MFD and AFD rank 

mfd_dindex_afd["mfd_rank"] = mfd_dindex_afd.groupby("date")["mfd"].transform(lambda x: pd.qcut(x, 10, labels = False, duplicates="drop"))+1
mfd_dindex_afd["afd_rank"] = mfd_dindex_afd.groupby("date")["afd"].transform(lambda x: pd.qcut(x, 10, labels = False, duplicates="drop"))+1
mfd_dindex_afd["dindex_rank"] = mfd_dindex_afd.groupby("date")["d_index"].transform(lambda x: pd.qcut(x, 10, labels = False, duplicates="drop"))+1

mfd_dindex_afd["yyyy"] = mfd_dindex_afd.date.dt.to_period("Y").apply(lambda x: x.year)

mfd_afd_dindex_rank_mfd = mfd_dindex_afd.groupby(["yyyy","date","mfd_rank"]).dindex_rank.median().groupby(["yyyy","mfd_rank"]).mean().to_frame(name = "d_index").reset_index()
mfd_afd_dindex_rank_mfd["d_type"] = "MFD"
mfd_afd_dindex_rank_afd = mfd_dindex_afd.groupby(["yyyy","date","afd_rank"]).dindex_rank.median().groupby(["yyyy","afd_rank"]).mean().to_frame(name = "d_index").reset_index()
mfd_afd_dindex_rank_afd["d_type"] = "AFD"

mfd_afd_plot = pd.concat([mfd_afd_dindex_rank_mfd.rename(columns = {"mfd_rank": "rank"}),
                          mfd_afd_dindex_rank_afd.rename(columns = {"afd_rank": "rank"})])

mfd_afd_plot = mfd_afd_plot[mfd_afd_plot["rank"].isin([1,5,10])]


fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=1000, sharex=True)
sns.lineplot(data=mfd_afd_plot, 
             x="yyyy", y="d_index", hue="rank", style="d_type",
             palette = "rocket_r")

ax.set_ylim(1, 10)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:4]+ handles[5:], labels=labels[1:4] + labels[5:])
fig.tight_layout()
ax.set(xlabel='', ylabel='Disagreement Index')
fig.savefig(DATA_RESULTS + "fig4.pdf")
