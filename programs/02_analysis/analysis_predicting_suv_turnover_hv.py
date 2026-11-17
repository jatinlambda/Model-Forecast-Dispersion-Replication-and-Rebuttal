"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Analyses: Predicting trade volume, volatility etc.
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
from helper import sig_stars, dollar, get_norm_chars
from linearmodels import PanelOLS

flag_winsorize = True
flag_standardize = True

def wins(x, bds):
    q = x.quantile(bds)
    x.loc[x<=q.iloc[0]] = q.iloc[0]
    x.loc[x>=q.iloc[1]] = q.iloc[1]
    return x


from pylatex import (
    Tabular,
    MultiColumn,
)

num_format="%.2f"

# %%% Merge MFD with Characteristics Data
mfd = pd.read_parquet(DATA_MFD / "main_data.pq")

norm_chars = get_norm_chars("unnorm")
mfd = mfd[["date","permno","mfd","ret_exc_lead1m"]]
mfd = pd.merge(mfd, norm_chars, on = ["permno","date"])

# %% Predict future abnormal trading volume and volatility
tvol = pd.read_pickle(DATA_DERIVED / "monthly_turnover.pkl.gz", compression = "gzip")
tvol = tvol.sort_values(["permno","date"])
tvol["next_turnover_21d"] = tvol.groupby("permno").turnover_21d.shift(-1)
tvol["past_turnover_21d"] = tvol.groupby("permno").turnover_21d.shift(1)

rvol = norm_chars[["date","permno","rvol_21d"]]
rvol = rvol.sort_values(["permno","date"])
rvol["next_rvol_21d"] = rvol.groupby("permno").rvol_21d.shift(-1)
rvol["past_rvol_21d"] = rvol.groupby("permno").rvol_21d.shift(1)
rvol["past2_rvol_21d"] = rvol.groupby("permno").rvol_21d.shift(2)
rvol["past3_rvol_21d"] = rvol.groupby("permno").rvol_21d.shift(3)

suv_all = pd.read_parquet(DATA_DERIVED / "suv.pq", columns = ["date","permno", "suv_m"])
suv_all = suv_all.sort_values(["permno","date"])
suv_all["next_suv_m"] = suv_all.groupby("permno").suv_m.shift(-1)
suv_all["past_suv_m"] = suv_all.groupby("permno").suv_m.shift(1)

mfd = mfd.merge(suv_all, on = ["date","permno"], how = "left")

mfd = mfd.merge(tvol[["date","permno","next_turnover_21d","turnover_21d", "past_turnover_21d"]], on = ["date", "permno"], how = "left")
mfd = mfd.merge(rvol[["date","permno","next_rvol_21d", "past_rvol_21d"]], on = ["date", "permno"], how = "left")

mfd = mfd.sort_values(["permno","date"])

controls = ["mfd","be_me", "beta_60m", "market_equity", "ret_12_1", "ret_1_0"]

controls_win = ["ret_12_1", "ret_1_0", "turnover_21d", "rvol_21d", "mfd",
                "be_me", "beta_60m", "market_equity", "next_turnover_21d", "past_turnover_21d",
                "next_rvol_21d", "past_rvol_21d", "suv_m", "next_suv_m", "past_suv_m"]

y_vars = ["suv_m","next_suv_m", "turnover_21d", "next_turnover_21d", "rvol_21d", "next_rvol_21d"]

mfd_win = mfd.copy()
mfd_win = mfd_win.sort_values(["permno","date"])

if flag_winsorize:
    cols_to_win = controls_win + y_vars
    for c in cols_to_win:
        mfd_win[c] = wins(mfd_win[c], [0.005, 0.995])

if flag_standardize:
    cols_to_win = controls_win
    for c in cols_to_win:
        mfd_win[c] = mfd_win.groupby("date")[c].transform(lambda x: (x-x.mean())/x.std())
        
mfd_win = mfd_win.set_index(["permno","date"]).sort_index()

results = []
for y_var in y_vars:
    if not("next" in y_var):
        add_control = controls + ["past_" + y_var]
    else:
        add_control = controls + [y_var.replace("next_","")]
    tmp = mfd_win[add_control + [y_var]].copy()
    tmp = tmp.replace(np.inf, np.nan).replace(-np.inf, np.nan)
    y = tmp[y_var]
    x = tmp[add_control]    
    
    mod = PanelOLS(y, x, entity_effects=True, time_effects = True)
    res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True, low_memory=True)
    results.append(res)

# do table
fname = "tab11"

tabular = Tabular("lcccccc", booktabs=True)
colnames = ["", 
            MultiColumn(2, data = "SUV", align = "c"),
            MultiColumn(2, data = "Turnover", align = "c"),
            MultiColumn(2, data = "Historical Volatility", align = "c")]
tabular.add_row(colnames)
tabular.add_hline(2,3, cmidruleoption = "lr")
tabular.add_hline(4,5, cmidruleoption = "lr")
tabular.add_hline(6,7, cmidruleoption = "lr")

colnames = [""] + 3* [dollar("t"), dollar("t+1")]
tabular.add_row(colnames)
tabular.add_hline(1,7)

add_row = ["MFD"]
add_row_t = [""]
add_row3 = ["Observations"]
add_row4 = ["R-Squared"]

for r in results:
    r1 = r.params["mfd"] 
    r2 = r.tstats["mfd"]
    r1 = num_format % r1 + sig_stars(r2)
    r2 = "({})".format(num_format % r2)
    add_row.append(r1)
    add_row_t.append(r2)
    add_row3.append(r.nobs)
    add_row4.append(num_format % r.rsquared_within)

tabular.add_row(add_row)
tabular.add_row(add_row_t)
#tabular.add_hline()
tabular.add_row(["Controls"] + ["Yes"]*6)
tabular.add_row(["Entity FE"] + ["Yes"]*6)
tabular.add_row(["Time FE"] + ["Yes"]*6)
tabular.add_row(add_row3)
tabular.add_row(add_row4)

tabular.generate_tex(DATA_RESULTS + fname)
print("table writen to {}".format(DATA_RESULTS + fname))


