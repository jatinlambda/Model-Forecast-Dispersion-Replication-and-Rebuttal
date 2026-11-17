"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Analyses: MFD pricing for international stocks
"""

from pathlib import Path
import os
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
from helper import fmb_to_table, mapper, mapper_rev
from scipy.stats.mstats import winsorize
import glob
from linearmodels import FamaMacBeth

flag_winsorize = True
flag_standardize = True

def wins(x, bds):
    q = x.quantile(bds)
    x.loc[x<=q.iloc[0]] = q.iloc[0]
    x.loc[x>=q.iloc[1]] = q.iloc[1]
    return x

# %% Load Data
folder = str(DATA_MFD / "international/") + os.path.sep
dataset = str(DATA_RAW / "international/") + os.path.sep 

name_mapper = {
    "g10": "G10 ex USA",
    "g7": "G7 ex USA",
    "emerging": "Emerging",
    "developed": "Developed ex USA",
    "europe": "Europe"}

# %% Load data
international_files = glob.glob(folder + "*.csv")
international_files = np.sort(international_files)
controls = (["MFD", "BETA", "SIZE", "BM", "MOM"],
            ["MFD", "BETA", "SIZE", "BM", "AG", "OP", "MOM"],
            )
controls = ["MFD", 'SUE', 'AG', 'MOM', 'ILLIQ', 'OP', 'IVOL', 'BETA',
 'SIZE', 'BM', 'MAX', 'TURN', 'STR']

fmb_regressions = []
col_names_fmb = []
for file in international_files:
    print(f"Running {file}")
    runid = os.path.basename(file).split(".")[0]
    name = name_mapper[runid]
    col_names_fmb.append(name)
    tmp = pd.read_csv(file,  index_col = 0)
    tmp["mfd"] = tmp["std"]
    tmp.date = pd.to_datetime(tmp.date, format = "%Y-%m-%d")
    tmp.date = tmp.date.dt.to_period("M").dt.to_timestamp("M")
    tmp = tmp[["date","permno","mfd"]]

    tmp = tmp.dropna(subset = "mfd")
    tmp = tmp.sort_values(["date","permno"]).set_index(["date","permno"])
    
    ret = pd.read_pickle(f'{dataset}{runid}_RET.pkl.zip')
    ret = ret.reset_index()
    ret.date = pd.to_datetime(ret.date, format = "%Y%m%d")
    ret.date = ret.date.dt.to_period("M").dt.to_timestamp("M")
    ret = ret.dropna(subset = "ret_exc_lead1m")
    
    mcap = pd.read_pickle(f'{dataset}{runid}_me.pkl.zip')
    mcap = mcap.rename(columns = {"market_equity":"me"})
    mcap = mcap.reset_index()
    mcap.date = pd.to_datetime(mcap.date, format = "%Y%m%d")
    mcap.date = mcap.date.dt.to_period("M").dt.to_timestamp("M")
    mcap = mcap.set_index(["date","permno"])

    tmp = tmp.merge(ret, on = ["date","permno"])
    tmp = tmp.merge(mcap, on = ["date","permno"])
    tmp.ret_exc_lead1m = tmp.groupby("date").ret_exc_lead1m.transform(lambda x: winsorize(x, [0.005, 0.005]))

    unnorm_char = pd.read_pickle(f'{dataset}{runid}.pkl.zip')
    tmp = tmp[["date","permno","mfd","ret_exc_lead1m"]]
    tmp = pd.merge(tmp, unnorm_char, on = ["permno","date"])
    tmp = tmp.set_index(["date","permno"]).sort_index()

    # Winsorize data
    needed_cols = [mapper[co] for co in controls]

    tmp_win = tmp.copy()
    if flag_winsorize:
        for c in needed_cols:
            tmp_win[c] = wins(tmp_win[c], [0.005, 0.995])

    if flag_standardize:
        for c in needed_cols:
            tmp_win[c] = tmp_win.groupby("date")[c].transform(lambda x: (x-x.mean())/x.std())
            
    # conform to expected input for linearmodels FMB implementation using (entity, time) as MultiIndex
    tmp_win = tmp_win.reset_index().set_index(["permno","date"]).sort_index()

    tmp_tmp = tmp_win[needed_cols + ["ret_exc_lead1m"]].copy()
    y = tmp_tmp.ret_exc_lead1m*100
    x = tmp_tmp[needed_cols]
    x["Const"] = 1
    mod = FamaMacBeth(y, x)
    res = mod.fit(cov_type="kernel", kernel="newey-west")
    fmb_regressions.append(res)
    
fmb_to_table(fmb_regressions, DATA_RESULTS + "ia_tab_g1", renamer = mapper_rev, 
             colnames = col_names_fmb, 
             prec = 2)