"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Analyses: MFD around earnings announcement days
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
from joblib import Parallel, delayed
from linearmodels import PanelOLS
from helper import sig_stars, dollar, do_portfolio_sort
from pylatex import Tabular, MultiColumn

def wins(x, bds):
    q = x.quantile(bds)
    x.loc[x<=q.iloc[0]] = q.iloc[0]
    x.loc[x>=q.iloc[1]] = q.iloc[1]
    return x


# %% Settings
max_date = "2022-12-31"
dtype = {
    "permno": "int64",
}

num_format = "%.2f" 

# %% Data
mfd = pd.read_parquet(DATA_MFD / "main_data.pq")

crsp_prices = pd.read_parquet(DATA_RAW / "daily_crsp.pq")
crsp_prices.date = pd.to_datetime(crsp_prices.date, format="%Y-%m-%d")
crsp_prices = crsp_prices.astype(dtype)
    
crsp_daily_vol = crsp_prices.groupby("date")["vol"].sum()
crsp_daily_vol = crsp_daily_vol.to_frame(name = "market_vol").reset_index()
crsp_prices = crsp_prices.merge(crsp_daily_vol, on = "date")
crsp_prices["scaled_vol"] = crsp_prices["vol"] / crsp_prices["market_vol"]
crsp_prices = crsp_prices.sort_values(["date","permno"])
crsp_prices = crsp_prices[crsp_prices.permno.isin(list(mfd.permno.unique()))]
crsp_prices = crsp_prices.astype(dtype)
crsp_prices.date = crsp_prices.date.dt.to_period("D").dt.to_timestamp("D")

# %% Load EA data
gv_mapping = pd.read_csv(DATA_RAW / "usa.csv", usecols = ["permno","eom","gvkey"])
gv_mapping["eom"] = pd.to_datetime(gv_mapping["eom"], format = "%Y%m%d")
gv_mapping["eom"] = gv_mapping["eom"].dt.to_period("M").dt.to_timestamp("M")
gv_mapping = gv_mapping.dropna(subset = "permno")
gv_mapping.eom = gv_mapping.eom.dt.to_period("D").dt.to_timestamp("D")

ea = pd.read_csv(DATA_RAW / "compustat_ea.zip")
ea = ea.dropna(subset = "rdq")
ea = ea[["gvkey","rdq"]].drop_duplicates()
ea["rdq"] = pd.to_datetime(ea.rdq, format = "%Y-%m-%d")
ea["eom"] = ea["rdq"].dt.to_period("M").dt.to_timestamp("M")
ea = ea.merge(gv_mapping, on = ["eom","gvkey"])
ea = ea.drop(columns = ["eom", "gvkey"])

ea = ea.astype(dtype)
ea = ea.sort_values(["rdq","permno"])
ea = ea[ea.rdq > crsp_prices.date.min()]
ea = ea[ea.permno.isin(list(mfd.permno.unique()))]

# %% identify real ea date as in Engelberg et al.
ea_date = crsp_prices.merge(ea, left_on = ["permno", "date"], right_on = ["permno", "rdq"])
ea_date_before = pd.merge_asof(ea, crsp_prices, right_on = ["date"], left_on = ["rdq"], 
                               by = "permno", allow_exact_matches=False, 
                               direction = "backward")
ea_date_after = pd.merge_asof(ea, crsp_prices, right_on = ["date"], 
                              left_on = ["rdq"], by = "permno", 
                              allow_exact_matches=False, direction = "forward")

ea_dates = pd.concat([ea_date, ea_date_before, ea_date_after])
ea_dates = ea_dates.dropna(subset = "date")
ea_dates = ea_dates.dropna(subset = "scaled_vol")

ea_single_date = Parallel(n_jobs=-1, verbose=True, backend = "loky")(
    delayed(lambda x: x[x.scaled_vol == x.scaled_vol.max()])(grp)
    for _, grp in ea_dates.groupby(["permno","rdq"])
)
ea_single_date = pd.concat(ea_single_date)
ea_single_date["ea_flag"] = 1
ea_single_date = ea_single_date.sort_values(["date","permno"])
ea_single_date = ea_single_date[["permno","date","ea_flag"]]
# to be sure
ea_single_date = ea_single_date.drop_duplicates()
ea_single_date = ea_single_date.rename(columns = {"date": "report_date"})

# %% indicators for earnings announcement period
# ea_flag_single gives the exact earnings annoucement day
# ea_flag gives the 3-day window around the exact earnings announcement window
ea_final_date = crsp_prices.merge(ea_single_date, left_on = ["permno", "date"], right_on = ["permno", "report_date"])
ea_final_date["ea_flag_single"] = 1
ea_final_date_before = pd.merge_asof(ea_single_date, crsp_prices, right_on = ["date"], 
                                     left_on = ["report_date"], by = "permno", 
                                     allow_exact_matches=False, direction = "backward")
ea_final_date_before["ea_flag_single"] = 0
ea_date_after = pd.merge_asof(ea_single_date, crsp_prices, right_on = ["date"], 
                              left_on = ["report_date"], by = "permno", 
                              allow_exact_matches=False, direction = "forward")
ea_date_after["ea_flag_single"] = 0

ea_final = pd.concat([ea_final_date, ea_final_date_before, ea_date_after])
ea_final = ea_final[["permno","date","ea_flag","ea_flag_single"]]
ea_final = ea_final.dropna(subset = "date").drop_duplicates()

crsp_prices = crsp_prices[(crsp_prices.date < max_date) & (crsp_prices.date > mfd.date.min())]
crsp_prices = crsp_prices.merge(ea_final, on = ["date","permno"], how = "left")
crsp_prices.ea_flag = crsp_prices.ea_flag.fillna(0)
crsp_prices.ea_flag_single = crsp_prices.ea_flag_single.fillna(0)

# %% keep only permnos which are also in our mfd sample
crsp_prices = crsp_prices.sort_values(["permno","date"])
crsp_prices = crsp_prices[["permno","date","vol","ret","ea_flag","prc","ea_flag_single"]]
crsp_prices.vol = crsp_prices.vol / 10**6

# %% add some controls
crsp_prices["ret_squared"] = crsp_prices.ret**2
control_var = []

for field in ["ret","ret_squared","vol"]:
    for i in range(1,11):
        crsp_prices[field + f"_{i}"] = crsp_prices.groupby(["permno"])[field].shift(i)
        control_var.append(field + f"_{i}")

# %% merge with mfd
mfd = pd.read_parquet(DATA_MFD / "main_data.pq")
 
mfd = mfd.sort_values(["date","permno"])
mfd = mfd.astype(dtype)
mfd = do_portfolio_sort(mfd.copy(), "mfd", "mfd_port", 10, "", 10)
mfd = mfd.drop(columns = ["cdf","min_cdf","bp_stock","bp_n"])

crsp_prices = crsp_prices.sort_values(["date","permno"])

crsp_prices["ym"] = (crsp_prices.date.dt.to_period("M") )
mfd["ym"] = mfd.date.dt.to_period("M") + 1

data = crsp_prices.merge(mfd[["ym","mfd","permno", "mfd_port"]], on = ["permno", "ym"])

# final preparation
data = data.dropna(subset= "mfd")

# %% run regressions
data = data.set_index(["permno","date"])
data.ea_flag_single = data.ea_flag_single.astype("int")
data.ea_flag = data.ea_flag.astype("int")
data.ret *= 100

results = []
reg = PanelOLS.from_formula(
    "ret ~  mfd*ea_flag_single + TimeEffects",
    data=data,
).fit(cov_type="clustered", cluster_entity=False, cluster_time=True, low_memory=True)
results.append(reg)

reg = PanelOLS.from_formula(
    "ret ~  mfd*ea_flag_single + ret_1 + ret_2 + ret_3 + ret_4 + ret_5 + ret_6 + ret_7 + ret_8 + ret_9 + ret_10 + \
    ret_squared_1 + ret_squared_2 + ret_squared_3 + ret_squared_4 + ret_squared_5 + ret_squared_6 + ret_squared_7 + ret_squared_8 + ret_squared_9 + ret_squared_10 + \
        + vol_1 + vol_2 + vol_3 + vol_4 + vol_5 + vol_6 + vol_7 + vol_8 + vol_9 + vol_10 + TimeEffects",
    data=data,
).fit(cov_type="clustered", cluster_entity=False, cluster_time=True, low_memory=True)
results.append(reg)

reg = PanelOLS.from_formula(
    "ret ~  mfd*ea_flag + TimeEffects",
    data=data,
).fit(cov_type="clustered", cluster_entity=False, cluster_time=True, low_memory=True)
results.append(reg)

reg = PanelOLS.from_formula(
    "ret ~  mfd*ea_flag + ret_1 + ret_2 + ret_3 + ret_4 + ret_5 + ret_6 + ret_7 + ret_8 + ret_9 + ret_10 + \
    ret_squared_1 + ret_squared_2 + ret_squared_3 + ret_squared_4 + ret_squared_5 + ret_squared_6 + ret_squared_7 + ret_squared_8 + ret_squared_9 + ret_squared_10 + \
        + vol_1 + vol_2 + vol_3 + vol_4 + vol_5 + vol_6 + vol_7 + vol_8 + vol_9 + vol_10 + TimeEffects",
    data=data,
).fit(cov_type="clustered", cluster_entity=False, cluster_time=True, low_memory=True)
results.append(reg)

# %% do table
ncol = 5
colnames = ["Dep. variable"] + [ dollar("Ret_t^d")]*4

tabular = Tabular("l" + "c" * (ncol - 1), booktabs=True)
to_add =  ["", MultiColumn(2, data = "Panel A: One-day Window", align = "l"), MultiColumn(2, data = "Panel B: Three-day Window", align = "l")]
tabular.add_row(to_add)
tabular.add_hline(2,3, cmidruleoption = "lr")
tabular.add_hline(4,5, cmidruleoption = "lr")
tabular.add_row(colnames)
tabular.add_hline(0,ncol-1)

# append MFD coeff
to_add = ["MFD"]
to_add2 = [""]
for reg in results:
    slope = reg.params["mfd"]
    tval = reg.tstats["mfd"]
    slope = num_format % slope + sig_stars(tval)
    tval = "({})".format(num_format % tval)
    to_add.append(slope)
    to_add2.append(tval)
tabular.add_row(to_add)
tabular.add_row(to_add2)

# append interaction term
to_add = [dollar(r"\textrm{MFD}\times\textrm{EDAY}")]
to_add2 = [""]
for reg in results:
    if "mfd:ea_flag" in reg.params.index:
        flag = "mfd:ea_flag"
    else:
        flag = "mfd:ea_flag_single"
    slope = reg.params[flag]
    tval = reg.tstats[flag]
    slope = num_format % slope + sig_stars(tval)
    tval = "({})".format(num_format % tval)
    to_add.append(slope)
    to_add2.append(tval)
tabular.add_row(to_add)
tabular.add_row(to_add2)

# append EDAY
to_add = ["EDAY"]
to_add2 = [""]
for reg in results:
    if "ea_flag" in reg.params.index:
        flag = "ea_flag"
    else:
        flag = "ea_flag_single"
    slope = reg.params[flag]
    tval = reg.tstats[flag]
    slope = num_format % slope + sig_stars(tval)
    tval = "({})".format(num_format % tval)
    to_add.append(slope)
    to_add2.append(tval)
tabular.add_row(to_add)
tabular.add_row(to_add2)

to_add = ["Lagged Controls?"]
for reg in results:
    if "vol_1" in reg.params.index:
        to_add.append("Yes")
    else:
        to_add.append("No")
tabular.add_row(to_add)

to_add = ["Day Fixed Effects?"]
for reg in results:
    if reg.model.time_effects:
        to_add.append("Yes")
    else:
        to_add.append("No")
tabular.add_row(to_add)

fname = DATA_RESULTS + "tab14"
tabular.generate_tex(fname)
print("table writen to {}".format(fname))    

