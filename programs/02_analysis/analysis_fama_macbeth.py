"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Analyses: Fama-MacBeth regressions
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
from helper import (fmb_to_table, 
                    get_norm_chars, 
                    mapper, 
                    mapper_rev, 
                    get_green, 
                    savepath)
from linearmodels import FamaMacBeth

flag_winsorize = True
flag_standardize = True

def wins(x, bds):
    q = x.quantile(bds)
    x.loc[x<=q.iloc[0]] = q.iloc[0]
    x.loc[x>=q.iloc[1]] = q.iloc[1]
    return x

# %%% Merge MFD with Characteristics Data
mfd = pd.read_parquet(DATA_MFD / "main_data.pq")

norm_chars = get_norm_chars("unnorm")
mfd = mfd[["date","permno","mfd","ret_exc_lead1m","mfd_mean"]]
mfd = pd.merge(mfd, norm_chars, on = ["permno","date"])

# green data for disp
green = get_green()

# same dates
mfd = mfd.merge(green, on = ["date","permno"], how ="left")

# get additional return data
add_returns = pd.read_parquet(DATA_RAW / "dgtw_ff48_returns.pq")
add_data = pd.read_csv(DATA_RAW / "usa.csv", usecols = ["permno","eom","ret","ret_exc"])
add_data.eom = pd.to_datetime(add_data.eom, format = "%Y%m%d")
add_data.eom = add_data.eom.dt.to_period("M").dt.to_timestamp("M")
add_data = add_data.rename(columns = {"eom":"date"})
add_data["rf"] = add_data.ret - add_data.ret_exc
add_data = add_data.groupby("date").rf.mean().to_frame(name = "rf").reset_index()
add_returns = add_returns.merge(add_data, on = "date")
add_returns["ff48_pf"] = add_returns.ff48_pf - add_returns.rf
add_returns["dgtw_vwret"] = add_returns.dgtw_vwret - add_returns.rf
add_returns.date = (add_returns.date.dt.to_period("M")-1).dt.to_timestamp("M")
mfd = mfd.merge(add_returns[["date","permno","ff48_pf", "dgtw_vwret"]], on = ["date","permno"], how = "left")
mfd["ret_exc_lead1m_ff48"] = mfd.ret_exc_lead1m - mfd.ff48_pf
mfd["ret_exc_lead1m_dgtw"] = mfd.ret_exc_lead1m - mfd.dgtw_vwret
mfd = mfd.drop(columns = ["ff48_pf","dgtw_vwret"])

mfd = mfd.set_index(["date","permno"])

# arrange for FMB
mfd = mfd.sort_index()

# Winsorize data
mfd_win = mfd.copy()
if flag_winsorize:
    cols_to_win = [x for x in mfd_win.columns if not(x in ["ret_exc_lead1m","ret_exc_lead1m_ff48","ret_exc_lead1m_dgtw"])]
    for c in cols_to_win:
        mfd_win[c] = wins(mfd_win[c], [0.005, 0.995])

if flag_standardize:
    cols_to_win = [x for x in mfd_win.columns if not(x in ["ret_exc_lead1m","ret_exc_lead1m_ff48","ret_exc_lead1m_dgtw"])]
    for c in cols_to_win:
        mfd_win[c] = mfd_win.groupby("date")[c].transform(lambda x: (x-x.mean())/x.std())
        
# conform to expected input for linearmodels FMB implementation using (entity, time) as MultiIndex
mfd_win = mfd_win.reset_index().sort_values(["date","permno"]).set_index(["permno","date"])
        
# %%% Run Fama-MacBeth regressions

controls = (["MFD", "BETA", "SIZE", "BM", "MOM"],
            ["MFD", "BETA", "SIZE", "BM", "AG", "OP", "MOM"],
            ["MFD", 'SUE', 'AG', 'MOM', 'ILLIQ', 'OP', 'IVOL', 'BETA',
             'SIZE', 'BM', 'MAX', 'TURN', 'STR'],
            )

# Future Excess Returns
results = []
for control in controls:
    needed_cols = [mapper[co] for co in control]
    tmp = mfd_win[needed_cols + ["ret_exc_lead1m"]].copy()
    y = tmp.ret_exc_lead1m*100
    x = tmp[needed_cols]
    x["Const"] = 1
    mod = FamaMacBeth(y, x)
    res = mod.fit(cov_type="kernel", kernel="newey-west")
    results.append(res)

add_ret_def = ["ret_exc_lead1m_ff48","ret_exc_lead1m_dgtw"]

for ret_def in add_ret_def:
    control = controls[-1]
    needed_cols = [mapper[co] for co in control]
    tmp = mfd_win[needed_cols + [ret_def]].copy()
    
    y = tmp[ret_def]*100
    x = tmp[needed_cols]
    x["Const"] = 1
    mod = FamaMacBeth(y, x)
    res = mod.fit(cov_type="kernel", kernel="newey-west")
    results.append(res)
    
fmb_to_table(results, DATA_RESULTS + "tab6", renamer = mapper_rev, 
             colnames = ["Excess Return"]*len(controls) + ["Industry-adj. Return", "DGTW-adj. Return"], 
             prec = 2)

# %% afd

mfd_afd = mfd[mfd.index.get_level_values("date") > "1982-12-31"].copy()
mfd_afd = mfd_afd.sort_index()

# Winsorize data
mfd_afd = mfd_afd.copy()
if flag_winsorize:
    cols_to_win = [x for x in mfd_afd.columns if not(x in ["ret_exc_lead1m","ret_exc_lead1m_ff48","ret_exc_lead1m_dgtw"])]
    for c in cols_to_win:
        mfd_afd[c] = wins(mfd_afd[c], [0.005, 0.995])

if flag_standardize:
    cols_to_win = [x for x in mfd_afd.columns if not(x in ["ret_exc_lead1m","ret_exc_lead1m_ff48","ret_exc_lead1m_dgtw"])]
    for c in cols_to_win:
        mfd_afd[c] = mfd_afd.groupby("date")[c].transform(lambda x: (x-x.mean())/x.std())
        
# conform to expected input for linearmodels FMB implementation using (entity, time) as MultiIndex
mfd_afd = mfd_afd.reset_index().set_index(["permno","date"]).sort_index()


controls = (["MFD", 'SUE', 'AG', 'MOM', 'ILLIQ', 'OP', 'IVOL', 'BETA',
 'SIZE', 'BM', 'MAX', 'TURN', 'STR', "AFD"],
            )

results = []
for control in controls:
    needed_cols = [mapper[co] for co in control]
    tmp = mfd_afd[needed_cols + ["ret_exc_lead1m"]].copy()
    y = tmp.ret_exc_lead1m*100
    x = tmp[needed_cols]
    x["Const"] = 1
    mod = FamaMacBeth(y, x)
    res = mod.fit(cov_type="kernel", kernel="newey-west")
    results.append(res)

add_ret_def = ["ret_exc_lead1m_ff48","ret_exc_lead1m_dgtw"]

for ret_def in add_ret_def:
    control = controls[-1]
    needed_cols = [mapper[co] for co in control]
    tmp = mfd_afd[needed_cols + [ret_def]].copy()
    
    y = tmp[ret_def]*100
    x = tmp[needed_cols]
    x["Const"] = 1
    mod = FamaMacBeth(y, x)
    res = mod.fit(cov_type="kernel", kernel="newey-west")
    results.append(res)
    
fmb_to_table(results, DATA_RESULTS + "ia_tab_i2", renamer = mapper_rev, 
             colnames = ["Excess Return"]*len(controls) + ["Industry-adj. Return", "DGTW-adj. Return"], 
             prec = 2)