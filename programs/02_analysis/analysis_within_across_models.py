"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Analyses: Disentangling within and across models/information set disagreement
"""

import pandas as pd
from helper import (sig_stars, 
                    do_strategy_overview, 
                    pfs, 
                    bp_min_n, 
                    bps, 
                    get_factors)

from pylatex import (
    Tabular,
    MultiColumn
)

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

# %% functions

def worker_single_combination(ids):
    
    f = str(DATA_MFD / "within_across") + os.path.sep
    
    mfd_different = pd.read_parquet(f+"_".join([str(x) for x in ids]) + ".pq")
    tmp = mfd.copy().merge(mfd_different, on = ["date","permno"])
    
    res = []
    strat_vw, _ = do_strategy_overview(tmp.copy(), "vw", "mfd_total", ap_factors, pfs, bps, bp_min_n)
    strat_vw = strat_vw[strat_vw.index == "H-L"]
    strat_vw["mfd_type"] = "mfd_total"
    res.append(strat_vw)
    
    strat_vw, _ = do_strategy_overview(tmp.copy(), "vw", "mfd_across_model", ap_factors, pfs, bps, bp_min_n)
    strat_vw = strat_vw[strat_vw.index == "H-L"]
    strat_vw["mfd_type"] = "mfd_across_model"
    res.append(strat_vw)
    
    strat_vw, _ = do_strategy_overview(tmp.copy(), "vw", "mfd_within_model", ap_factors, pfs, bps, bp_min_n)
    strat_vw = strat_vw[strat_vw.index == "H-L"]
    strat_vw["mfd_type"] = "mfd_within_model"
    res.append(strat_vw)
    res = pd.concat(res)
    
    res["weighting"] = "VW"
    
    res_ew = []
    strat_ew, _ = do_strategy_overview(tmp.copy(), "ew", "mfd_total", ap_factors, pfs, bps, bp_min_n)
    strat_ew = strat_ew[strat_ew.index == "H-L"]
    strat_ew["mfd_type"] = "mfd_total"
    res_ew.append(strat_ew)
    
    strat_ew, _ = do_strategy_overview(tmp.copy(), "ew", "mfd_across_model", ap_factors, pfs, bps, bp_min_n)
    strat_ew = strat_ew[strat_ew.index == "H-L"]
    strat_ew["mfd_type"] = "mfd_across_model"
    res_ew.append(strat_ew)
    
    strat_ew, _ = do_strategy_overview(tmp.copy(), "ew", "mfd_within_model", ap_factors, pfs, bps, bp_min_n)
    strat_ew = strat_ew[strat_ew.index == "H-L"]
    strat_ew["mfd_type"] = "mfd_within_model"
    res_ew.append(strat_ew)
    res_ew = pd.concat(res_ew)

    res_ew["weighting"] = "EW"
    
    res = pd.concat([res_ew, res])
    
    return res

# %% Model vs Information Set Disagreement
mfd = pd.read_parquet(DATA_MFD / "main_data.pq")

ap_factors = get_factors()
ap_factors = {"FF6": ap_factors["FF6"]}

num_format="%.2f"

diff_rf = [
    [4,5,9,13],
    [4,6,9,13],
    [5,10,12,15],
    [5,11,12,15],
    [5,9,12,14],
    [6,7,10,15]]

diff_mod = [
    [1,11,17,18],
    [1,3,17,18],
    [1,7,17,18],
    [2,16,17,18],
    [2,6,17,19],
    [2,8,17,19]
    ]

# work on rf only
rf_data = []
for ids in diff_rf:
    tmp = worker_single_combination(ids)
    tmp["Config IDs"] = ", ".join([str(x) for x in ids])
    rf_data.append(tmp)
rf_data = pd.concat(rf_data)
    
# work on mixing models
m_data = []
for ids in diff_mod:
    tmp = worker_single_combination(ids)
    tmp["Config IDs"] = ", ".join([str(x) for x in ids])
    m_data.append(tmp)
m_data = pd.concat(m_data)

m_data = m_data.set_index("Config IDs")

# do table
tabular = Tabular("l"+"c" * 12, booktabs=True)
columns = ["Config IDs", 
           MultiColumn(6, align = "c", data = "Equal-Weighted"),
           MultiColumn(6, align = "c", data = "Value-Weighted")]
tabular.add_row(columns)
#tabular.add_hline(1,1,cmidruleoption="lr")
tabular.add_hline(2,7,cmidruleoption="lr")
tabular.add_hline(8,13,cmidruleoption="lr")

columns = ["", 
           MultiColumn(2, align = "c", data = "Total"),
           MultiColumn(2, align = "c", data = "Model"),
           MultiColumn(2, align = "c", data = "Information Set"),
           MultiColumn(2, align = "c", data = "Total"),
           MultiColumn(2, align = "c", data = "Model"),
           MultiColumn(2, align = "c", data = "Information Set")]
tabular.add_row(columns)

tabular.add_hline(2,3,cmidruleoption="lr")
tabular.add_hline(4,5,cmidruleoption="lr")
tabular.add_hline(6,7,cmidruleoption="lr")
tabular.add_hline(8,9,cmidruleoption="lr")
tabular.add_hline(10,11,cmidruleoption="lr")
tabular.add_hline(12,13,cmidruleoption="lr")
columns = ["", 
           "H-L", "FF6",
           "H-L", "FF6",
           "H-L", "FF6",
           "H-L", "FF6",
           "H-L", "FF6",
           "H-L", "FF6",
           ]
tabular.add_row(columns)
tabular.add_hline()
tabular.add_row([MultiColumn(13, align = "l", data = "Panel A: Different Random Forest Hyperparameters")])
tabular.add_hline()

for c, row in rf_data.groupby("Config IDs"):
    add1 = [c]
    add2 = [""]
    
    for ix in range(row.shape[0]):
        hml = row["XRet"].values[ix]
        hml_t = row["tval"].values[ix]
        ff6 = row["FF6"].values[ix]
        ff6_t = row["t_FF6"].values[ix]
        
        hml = num_format % hml
        ff6 = num_format % ff6
        
        hml = hml + sig_stars(hml_t)
        ff6 = ff6 + sig_stars(ff6_t)
        
        hml_t = "({})".format(num_format % hml_t)
        ff6_t = "({})".format(num_format % ff6_t)
        
        add1.extend([hml, ff6])
        add2.extend([hml_t,ff6_t ])
    tabular.add_row(add1)
    tabular.add_row(add2)

tabular.add_hline()
tabular.add_row([MultiColumn(13, align = "l", data = "Panbel B: Different Machine Learning Models")])
tabular.add_hline()

for c, row in m_data.groupby("Config IDs"):
    add1 = [c]
    add2 = [""]
    
    for ix in range(row.shape[0]):
        hml = row["XRet"].values[ix]
        hml_t = row["tval"].values[ix]
        ff6 = row["FF6"].values[ix]
        ff6_t = row["t_FF6"].values[ix]
        
        hml = num_format % hml
        ff6 = num_format % ff6
        
        hml = hml + sig_stars(hml_t)
        ff6 = ff6 + sig_stars(ff6_t)
        
        hml_t = "({})".format(num_format % hml_t)
        ff6_t = "({})".format(num_format % ff6_t)
        
        add1.extend([hml, ff6])
        add2.extend([hml_t,ff6_t ])
    tabular.add_row(add1)
    tabular.add_row(add2)     
        
tabular.generate_tex(DATA_RESULTS + "tab12")

# %% create overview table with run configs
all_configs = pd.read_csv( DATA_MFD / "within_across" / "parametrization_overiew.csv", delimiter=";")
all_configs = all_configs[['Config ID', 'model_type', 'max_depth', 'n_estimators', 
                           'subsample', 'max_features', 'min_samples_split', 
                           'colsample_bynode', 'colsample_bylevel', 'colsample_bytree',
                           'min_child_weight', 'lambda', 'eta', "alpha"]]
all_configs = all_configs.rename(columns = {
    "model_type": "Model",
    "max_depth": "Max. depth",
    "n_estimators": "Nbr. trees",
    "subsample": "Subsample",
    "max_features": "Max. Features",
    "min_samples_split": "Min. sample split",
    "colsample_bynode": "by node",
    "colsample_bylevel": "by level",
    "colsample_bytree": "by tree",
    "min_child_weight": "Min. child weight"
    })


tabular = Tabular("c" * 14, booktabs=True)
columns1 = list(all_configs.columns[:7]) + [MultiColumn(3, align = "c", data = "Feature sample")] + list(all_configs.columns[10:])
columns2 = [""]*7 + list(all_configs.columns[7:10]) + [""]*4
tabular.add_row(columns1)
tabular.add_row(columns2)
tabular.add_hline()
for _, row in all_configs.iterrows():
    to_add = []
    for r in row:
        if r is None:
            to_add.append("")
        else:
            to_add.append(r)
    tabular.add_row(to_add)

tabular.generate_tex(DATA_RESULTS + "a_tab_c")
    


    

