"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Summary statistics: produces summary statistics
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
DATA_RESULTS = "results"

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from helper import (do_portfolio_sort, 
                    sig_stars, 
                    dollar, 
                    average_char_decile, 
                    get_norm_chars, 
                    mapper, 
                    get_misp,
                    get_green, 
                    pfs,
                    bp_min_n,
                    bps)

from pylatex import (
    Tabular
)

def wins(x, bds):
    q = x.quantile(bds)
    x.loc[x<=q.iloc[0]] = q.iloc[0]
    x.loc[x>=q.iloc[1]] = q.iloc[1]
    return x

def stats(x):
    return pd.Series(
        {
            "Mean": np.nanmean(x),
            "Sd": np.nanstd(x),
            "10p": np.nanquantile(x,0.1),
            "Q1": np.nanquantile(x,0.25),
            "Q2": np.nanquantile(x, 0.5),
            "Q3": np.nanquantile(x,0.75),
            "90p": np.nanquantile(x, 0.9)}
    )

num_format="%.2f"

# %% Load Data
mfd = pd.read_parquet(DATA_MFD / "main_data.pq")

norm_chars = get_norm_chars("unnorm").copy()
mfd = mfd[["date","permno","mfd","ret_exc_lead1m"]]

# annualize
mfd.mfd *=12
mfd = pd.merge(mfd, norm_chars, on = ["permno","date"])

green = get_green()
misp = get_misp()

mfd = mfd.merge(misp, on = ["date","permno"], how = "left")
mfd = mfd.merge(green, on = ["date","permno"], how = "left")

# %%
# Summary stats main variables
chars = ['r$RET_{t+1}$', 'MFD', 'SUE', 'AG', 'MOM', 'ILLIQ', 'OP', 'IVOL', 'BETA', 'SIZE', 'BM', 'MAX', 'TURN', 'STR', 'AFD']

summ_chars = []
cor_chars = []
for char in chars:
    char_m = mapper[char]
    # summary stats
    tmp = mfd[["date", char_m, "permno"]].dropna(subset = char_m).copy()
    tmp = tmp.sort_values(["date","permno"])
    tmp[char_m] = tmp.groupby("date")[char_m].transform(lambda x: winsorize(x, limits = [0.01, 0.01]))
    if char == "SIZE":
        tmp[char_m] = mfd[char_m] / 10**3
    elif char == "TURN":
        tmp[char_m] = mfd[char_m] * 10**3
    elif char == "r$RET_{t+1}$":
        tmp[char_m] = mfd[char_m] * 10**2
    tmp = tmp.groupby("date")[char_m].apply(lambda x: stats(x)).groupby(level=1).mean()
    tmp.name = char
    tmp = tmp.to_frame().T
    summ_chars.append(tmp)
    
    # correlation
    if not(char == "MFD"):
        corr = mfd[["date", char_m, "permno", "mfd"]].dropna(subset = char_m).copy()
        corr = corr.sort_values(["date","permno"])
        corr[char_m] = wins(corr[char_m], [0.005, 0.995])
        corr = corr.groupby("date").apply(lambda x: x[[char_m,"mfd"]].corr(method = "spearman").iloc[0,1])
        corr = stats(corr)
        corr.name = char
        corr = corr.to_frame().T
        cor_chars.append(corr)
    
summ_chars = pd.concat(summ_chars)
summ_chars = summ_chars[["Mean", "Sd", "10p", "Q1", "Q2", "Q3", "90p"]]
summ_chars = summ_chars.round(2)

cor_chars = pd.concat(cor_chars)
cor_chars = cor_chars[["Mean", "Sd", "10p", "Q1", "Q2", "Q3", "90p"]]

# generate table
ncol = summ_chars.shape[1] +1 
tabular = Tabular("l" + "c" * (ncol - 1), booktabs=True)
colnames = []
for c in list(summ_chars.columns):
    if c == "10p":
        colnames.append(dollar(r"10^\textrm{th}"))
    elif c == "90p":
        colnames.append(dollar(r"90^\textrm{th}"))
    else:
        colnames.append(c)
colnames = [""] + colnames
tabular.add_row(colnames)
tabular.add_hline()
for idx, row in summ_chars.iterrows():
    if idx == "TURN":
        idx = dollar(r"\textrm{TURN }(\times 10^{3})")
    if idx == "SIZE":
        idx = dollar(r"\textrm{SIZE }(\times 10^{-9})")
    if idx == "r$RET_{t+1}$":
        idx = dollar("RET_{t+1}")
    col = [idx]
    for n ,r in row.items():
        r = num_format % r
        col.append(r)
    tabular.add_row(col)

fname = DATA_RESULTS + "/tab1"
tabular.generate_tex(fname)

ncol = cor_chars.shape[1] +1 
tabular = Tabular("l" + "c" * (ncol - 1), booktabs=True)
colnames = []
for c in list(cor_chars.columns):
    if c == "10p":
        colnames.append(dollar(r"10^\textrm{th}"))
    elif c == "90p":
        colnames.append(dollar(r"90^\textrm{th}"))
    else:
        colnames.append(c)
colnames = [""] + colnames
tabular.add_row(colnames)
tabular.add_hline()
for idx, row in cor_chars.iterrows():
    if idx == "r$RET_{t+1}$":
        idx = dollar("RET_{t+1}")
    col = [idx]
    for n ,r in row.items():
        r = num_format % r
        col.append(r)
    tabular.add_row(col)

fname = DATA_RESULTS + "/tab2"
tabular.generate_tex(fname)
# %% 
# Summary stats for MFD deciles
chars = ['MFD', 'SUE', 'AG', 'MOM', 'ILLIQ', 'OP', 'IVOL', 'BETA', 'SIZE', 'BM', 'MAX', 'TURN', 'STR']

mfd = mfd.sort_values(["date","permno"])
mfd = do_portfolio_sort(mfd.copy(), "mfd", "mfd_port", pfs, bps, bp_min_n)
mfd = mfd.drop(columns = ["cdf","min_cdf","bp_stock","bp_n"])

mfd["mfd_stand"] = mfd.groupby("date")["mfd"].transform(lambda x: (x-x.mean())/x.std())
mfd.groupby("mfd_port").mfd_stand.mean()

mfd_decile = mfd.copy()
mfd_decile = mfd_decile.sort_values(["date"])
mfd_chars = []
for c in chars:
    mfd_decile_ = mfd_decile.copy()
    char = mapper[c]
    if c == "SIZE":
        mfd_decile_[char] = mfd_decile_[char] / 10**3
    elif c == "TURN":
        mfd_decile_[char] = mfd_decile_[char] * 10**3
    elif c == "STR":
        mfd_decile_[char] = mfd_decile_[char] * 10**3

    mfd_decile_ = mfd_decile_.groupby(["date"]).filter(lambda x: x.groupby("mfd_port").apply(lambda x: x[char].isnull().mean()).max() < 0.50)
    mfd_char = average_char_decile(mfd_decile_, char, c, agg_type = lambda x: np.nanmedian(x))

    mfd_chars.append(mfd_char)
mfd_chars = pd.concat(mfd_chars)

# generate table
ncol = mfd_chars.shape[1] +1 
tabular = Tabular("l" + "c" * (ncol - 1), booktabs=True)
colnames = [""] + list(mfd_chars.columns)
tabular.add_row(colnames)
tabular.add_hline()
for idx, row in mfd_chars.iterrows():
    if idx == "TURN":
        idx = dollar(r"\textrm{TURN }(\times 10^{3})")
    if idx == "STR":
        idx = dollar(r"\textrm{STR }(\times 10^{3})")
    if idx == "SIZE":
        idx = dollar(r"\textrm{SIZE }(\times 10^{-9})")
    col = [idx]
    for n ,r in row.items():
        r = num_format % r
        if n == "H-L":
            col.append( r + sig_stars(row["t-stat"]))
        elif n == "t-stat":
            col.append( "({})".format( r ))
        else:
            col.append(r)
    tabular.add_row(col)

fname = DATA_RESULTS + "/tab4"
tabular.generate_tex(fname)

# %% Transition Probabilities
mfd_trans = mfd[["date","permno","mfd_port"]].copy()
mfd_trans["ym"] = mfd_trans.date.dt.to_period("M")
trans = []
for ym in np.sort(mfd_trans.ym.unique()):
    actual = mfd_trans[mfd_trans.ym == ym]
    nyear = mfd_trans[mfd_trans.ym == ym + 12]
    if nyear.empty:
        break
    same_permno = list(set(nyear.permno.values) & set(actual.permno.values))
    actual = actual[actual.permno.isin(same_permno)]
    nyear = nyear[nyear.permno.isin(same_permno)]
    tmp = []
    for p in actual.mfd_port.unique():
        permno_p = actual[actual.mfd_port == p]["permno"].values
        for pp in nyear.mfd_port.unique():
            permno_np = nyear[nyear.mfd_port == pp].permno.values
            frac = len(list(set(permno_np) & set(permno_p))) / len(permno_p)
            tmp.append(
                pd.DataFrame(
                    {"port_t": p,
                     "port_t_12": pp,
                     "frac": frac},
                    index = [0]))
            
    tmp = pd.concat(tmp).reset_index(drop = True)
    tmp["ym"] = ym
    trans.append(tmp)
trans = pd.concat(trans)
trans = trans.set_index(["ym","port_t","port_t_12"]).unstack("port_t_12")
trans = trans.groupby("port_t").mean()
trans = (100*trans).round(0).astype(int)

# %%% do Table
ncol = trans.shape[1] +1 
tabular = Tabular("l" + "c" * (ncol - 1), booktabs=True)
cols = [int(x) for x in trans.columns.get_level_values(1).to_list()]
colnames = [""]
for c in cols:
    if c== 1:
        c = "Low"
    elif c == 10:
        c = "High"
    colnames.append(c)
tabular.add_row(colnames)
tabular.add_hline()
for idx, row in trans.iterrows():
    if idx == 1:
        idx = "Low"
    elif idx == 10:
        idx = "High"
    else:
        idx = int(idx)
    col = [idx]
    for r in row.values:
        col.append(r)
    tabular.add_row(col)

fname = DATA_RESULTS + "/ia_tab_a1"
tabular.generate_tex(fname)