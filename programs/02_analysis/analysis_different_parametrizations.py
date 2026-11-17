"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Analyses: Different MFD parametrizations
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
from helper import (do_strategy_overview, 
                    sig_stars,
                    dollar,
                    pfs,
                    bp_min_n,
                    bps,
                    savepath,
                    get_factors)

from pylatex import (
    Tabular,
    MultiColumn,
    LongTable,
    Command
)
import glob
import os

def make_table_pretty(tab):
    aligner = ["Model", "Nbr. features", "Nbr. investors", "Random Features", "gamma", "Max. depth", "Nbr. trees", 
               "Subsample", "Max. features", "Min. sample split", "by node", "by level", "by tree", 
               "Min. child weight", "lambda", "eta", "alpha", "XRet", "tval", "FF6", "t_FF6"]
    
    col_renamer = {
        "m": "Nbr. features",
        "model_type": "Model",
        "n_investors": "Nbr. investors",
        "min_child_weight": "Min. child weight",
        "max_features": "Max. features",
        "min_samples_split": "Min. sample split",
        "colsample_bynode": "by node",
        "colsample_bylevel": "by level",
        "colsample_bytree": "by tree",
        "max_depth": "Max. depth",
        "n_estimators": "Nbr. trees",
        "subsample": "Subsample",
        "random_feature": "Random Features"
        }

    to_drop = ["window_length", "expanding", "fit_frequency", "truncate", "winsorize", "cv", "cv_frequency", "transformation"]
    tab["random_feature"] = tab.apply(lambda row: row["random_feature"].capitalize() if row["transformation"] else None, axis = 1 )
    tab.loc[tab.transformation.apply(lambda x: not x), "gamma"] = None
    tab = tab.drop(columns = to_drop)
    tab = tab.rename(columns = col_renamer)
    tab.Model = tab.Model.str.capitalize()
    tab = tab.replace(np.nan, None)
    tab = tab[aligner]
    tab = tab.sort_values("Model")
    tab = pd.concat([tab[tab.Model != "Ridge"],
                     tab[tab.Model == "Ridge"]])

    return tab   

def params_sensi_to_latex(data, fname, caption, label, num_format = "%.2f", names = [], force_long = False):
    
    def replacer(inp):
        if inp == "XRet":
            return "H-L"
        elif inp == "tval":
            return "t-stat"
        elif inp[:2] == "t_":
            return "t-stat"
        elif inp == "lambda":
            return dollar(r"\lambda")
        elif inp == "eta":
            return dollar(r"\eta")
        elif inp == "alpha":
            return dollar(r"\alpha")
        elif inp == "gamma":
            return dollar(r"\gamma")
        else:
            return inp
        
    def exp_replacer(inp):
        if inp is None:
            return None
        if np.round(inp, 5) == 0.01832:
            return dollar(r"e^{-4}")
        elif np.round(inp, 5) == 0.36788:
            return dollar(r"e^{-1}")
        elif np.round(inp, 5) == 0.13534:
            return dollar(r"e^{-2}")
        elif np.round(inp, 5) == 0.04979:
            return dollar(r"e^{-3}")
        else:
            return inp
    
    def do_panel(tabular, tmp, name):
        if name is not None:
            to_add =  [MultiColumn(ncol, data = name, align = "l")]
            tabular.add_row(to_add)
            tabular.add_hline()
            tabular.add_row(colnames1)
            tabular.add_row(colnames2)
            tabular.add_hline(0,ncol-1)
            
            for ix, row in tmp.iterrows():
                to_add = []
                for ct, (i, r) in enumerate(row.items()):
                    if not(isinstance(r, str)) and not(isinstance(r, int)):
                        if r is None:
                            r = ""
                        else:
                            if i == "gamma" or i== "lambda":
                                r = exp_replacer(r)
                            else:
                                r = num_format % r
                    if (i == "tval") or (i[:2] == "t_"):
                        r = "({})".format(r)
                    elif (i == "XRet") or (i == "FF6"):
                        r = r + sig_stars(row.iloc[ct+1])
                    to_add.append(r)
                tabular.add_row(to_add)
    
    if isinstance(data, list):
        ncol = data[0].shape[1]
        force_long = True
    else:
        ncol = data.shape[1] 
        names = [None]
        data = [data]
    
    colnames = [replacer(x) for x in data[0].columns]
    colnames1 = []
    colnames2 = []
    by_flag = False
    for c in colnames:
        if "." in c:
            c = c.split(".")
            colnames1.append(c[0])
            colnames2.append(c[1])
        elif "by " in c:
            if by_flag:
                colnames2.append(c)
            else:
                by_flag = True
                colnames1.append(MultiColumn(3, data = "Feature sample", align = "c"))
                colnames2.append(c)
        else:
            colnames1.append(c)
            colnames2.append("")
        
    if force_long:
        tabular = LongTable("c" * ncol, booktabs=True)
    else:
        tabular = Tabular("c" * ncol, booktabs=True)
        
    for cnt, (ttmp, nam) in enumerate(zip(data,names)):
        do_panel(tabular, ttmp, nam)
        if force_long:
            if (cnt>=0) and (cnt<len(data)-1):
                tabular.add_hline(0, ncol-1)
            if (cnt < (len(data)-1)):
                tabular.append(Command("pagebreak"))
                tabular.add_hline()
    tabular.generate_tex(fname)
    
    fname = fname + ".tex"
    
    f = open(fname)
    lines = f.readlines()
    f.close()
    
    # insert caption and label
    idx_tabular = np.argwhere(["\\toprule" in line for line in lines]).T[0][0]
    caption_label = "\\caption{" + caption + "} \\label{" + label + "} \\\\%\n"
    lines.insert(idx_tabular, caption_label)
    
    with open(fname, mode="w") as f:
        f.write("% generated by python\n")
    f.close()
    
    with open(fname, mode="a") as f:
        for line in lines:
            f.write(line)
    f.close()

    print("table writen to {}".format(fname)) 
    

# %% Load Data
mcap = pd.read_parquet(DATA_DERIVED / "MKTEQ.pq")
mcap = mcap.rename(columns = {"market_equity":"me"})
mcap = mcap.reset_index()
mcap.date = pd.to_datetime(mcap.date, format = "%Y%m%d")
mcap.date = mcap.date.dt.to_period("M").dt.to_timestamp("M")
mcap = mcap.set_index(["date","permno"])

ret = pd.read_parquet(DATA_DERIVED / "RET.pq")
ret = ret.reset_index()
ret.date = pd.to_datetime(ret.date, format = "%Y%m%d")
ret.date = ret.date.dt.to_period("M").dt.to_timestamp("M")

# %% Other hyperparameter
ap_factors = get_factors()
param_risk_factors = {"FF6": ap_factors["FF6"]}

folder =  DATA_MFD / "different_parametrizations" 

files = glob.glob(str(folder) + os.path.sep + "*.pq")
files = [x for x in files if not("config" in x)]
ew = []
vw = []

for file in files:
    print(file)
    
    mfd_alt = pd.read_parquet(file)
    runid = file.split(os.path.sep)[-1].replace(".pq","")
    config_file = os.path.sep.join(file.split(os.path.sep)[:-1]) + os.path.sep + "config_" +file.split(os.path.sep)[-1]
    
    
    mfd_alt = mfd_alt.dropna(subset = "mfd_alt")
    mfd_alt = mfd_alt.sort_values(["date","permno"]).set_index(["date","permno"])
    
    mfd_alt = mfd_alt.merge(ret, on = ["date","permno"])
    mfd_alt = mfd_alt.merge(mcap, on = ["date","permno"])

    config = pd.read_parquet(config_file)

    tmp_vw,_ = do_strategy_overview(mfd_alt.copy(), "vw", "mfd_alt", param_risk_factors, pfs, bps, bp_min_n)
    tmp_vw = tmp_vw.loc["H-L"].to_frame(name = runid).T
    tmp_ew,_ = do_strategy_overview(mfd_alt.copy(), "ew", "mfd_alt", param_risk_factors, pfs, bps, bp_min_n)
    tmp_ew = tmp_ew.loc["H-L"].to_frame(name = runid).T
    config["runid"] = runid
    config = config.set_index("runid")
    
    tmp_ew["runid"] = runid
    tmp_ew = tmp_ew.set_index("runid")

    tmp_vw["runid"] = runid
    tmp_vw = tmp_vw.set_index("runid")
        
    ew.append(config.merge(tmp_ew, left_index = True, right_index = True))
    vw.append(config.merge(tmp_vw, left_index = True, right_index = True))

ew = pd.concat(ew)
vw = pd.concat(vw)

ew.index = ew.index.astype(int)
ew = ew.sort_index()
vw.index = vw.index.astype(int)
vw = vw.sort_index()

# %% make table
ew = make_table_pretty(ew)
vw = make_table_pretty(vw)

params_sensi_to_latex([ew, vw], 
                      fname = DATA_RESULTS+ "ia_tab_e1", 
                      num_format = "%.3f",
                       names = ["Panel A: Equal-Weighted Portfolios",
                                "Panel B: Value-Weighted Portfolios"],
                       caption = "Univariate MFD-spread Portfolio Returns for Different Predictor Model Specifications",
                       label = "tab:pfsortsparam")
