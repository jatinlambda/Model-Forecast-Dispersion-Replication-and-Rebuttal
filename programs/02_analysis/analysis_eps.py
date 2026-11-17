"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Analyses: earnings-based MFD
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
from helper import (do_strategy_overview, 
                    sig_stars, 
                    dollar, 
                    fmb_to_table, 
                    pfs, 
                    bp_min_n,
                    bps, 
                    mapper, 
                    mapper_rev, 
                    get_norm_chars,
                    get_green, 
                    get_factors)
from pylatex import (
    Tabular,
    MultiColumn,
    NoEscape
)
        
def get_eps_mfd(ids, names):
    
    data = pd.DataFrame([])
    
    for i in range(len(ids)):        
        config_i = ids[i]
        name = names[i] 
        tmp = pd.read_csv( str(DATA_MFD / "earnings") + os.path.sep + '{}.csv'.format(config_i), index_col = 0, parse_dates = ["date"], infer_datetime_format=True)
        tmp = tmp.rename(columns={"std":name, "mean": name + " MEAN"})
        tmp[name] *=100
        tmp.date = tmp.date.dt.to_period("M").dt.to_timestamp("M")
        
        if i == 0:
            data = tmp
        else:
            data = data.merge(tmp, on = ["date", "permno"], how = "outer")

    return data
    
# %% Load Data
mfd = pd.read_parquet(DATA_MFD / "main_data.pq")

ap_factors = get_factors()
mfd = mfd.dropna(subset ="ret_exc_lead1m")

green = get_green()
mfd = mfd.merge(green, on = ["date","permno"], how ="left")

ids = ["eps",
       "epsyield",
       "epsassets"]
names = ["EPS MFD",
        "Earnings Yield MFD",     
         "EPS to Assets MFD"]

eps_mfd = get_eps_mfd(ids, names)

mfd_eps = mfd.merge(eps_mfd, on = ["date","permno"], how = "left")

# %% univariate portfolio sorts
ap_factors_short = {"CAPM": ap_factors["CAPM"],
                    "FF6": ap_factors["FF6"]}
uni_port = {}
for name in names:
    tmp = mfd_eps.dropna(subset = name).copy()
    strat_vw, _ = do_strategy_overview(tmp.copy(), "vw", name, ap_factors_short, pfs, bps, bp_min_n)
    strat_ew, _ = do_strategy_overview(tmp.copy(), "ew", name, ap_factors_short, pfs, bps, bp_min_n)
    
    strat_vw = strat_vw.loc["H-L"]
    strat_ew = strat_ew.loc["H-L"]
    
    uni_port[name] = {"ew": strat_ew,
                        "vw": strat_vw}

# do table
num_format="%.2f"

def ease_formatter(val, t):
    val = num_format % val
    val = val + sig_stars(t)
    t = "({})".format(num_format % t)
    return val, t

tabular = Tabular("l"+"c" * 4, booktabs=True)
columns = ["Earnings MFD", 
           MultiColumn(2, align = "c", data = "Equal-Weighted"),
           MultiColumn(2, align = "c", data = "Value-Weighted")]
tabular.add_row(columns)
tabular.add_hline(2,3,cmidruleoption="lr")
tabular.add_hline(4,5,cmidruleoption="lr")

columns = ["", 
           "H-L", "FF6",
           "H-L", "FF6",
           ]
tabular.add_row(columns)
tabular.add_hline()

for k, v in uni_port.items():
    to_add1 = [k]
    to_add2 = [""]
    
    ew = v["ew"]
    x, y = ease_formatter(ew["XRet"], ew["tval"])
    to_add1.append(x)
    to_add2.append(y)
    
    x, y = ease_formatter(ew["FF6"], ew["t_FF6"])
    to_add1.append(x)
    to_add2.append(y)
    
    vw = v["vw"]
    x, y = ease_formatter(vw["XRet"], vw["tval"])
    to_add1.append(x)
    to_add2.append(y)
    
    x, y = ease_formatter(vw["FF6"], vw["t_FF6"])
    to_add1.append(x)
    to_add2.append(y)
    
    tabular.add_row(to_add1)
    tabular.add_row(to_add2)
    
tabular.generate_tex(DATA_RESULTS + "ia_tab_h1")

# %% Scatter Plot AFD and MFD
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

cm = 1 / 2.54
width = 15.92 * cm
height = 6 * cm

matplotlib.rcParams.update({"font.size": 8})
matplotlib.rcParams["axes.xmargin"] = 0.02
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.edgecolor"] = "0.15"
matplotlib.rcParams["axes.linewidth"] = 1.25

num_format="%.2f"

colors = sns.color_palette("rocket",4)

xs_corr_afd = mfd_eps.dropna(subset = "afd").groupby("date").apply(lambda x: x[["afd","mfd"]].corr(method = "spearman").loc["afd","mfd"])
xs_corr_afd = xs_corr_afd[xs_corr_afd.index > "1982-12-31"]

xs_corr_eps = mfd_eps.dropna(subset = ["afd", "EPS MFD"]).groupby("date").apply(lambda x: x[["afd","EPS MFD"]].corr(method = "spearman").loc["afd","EPS MFD"])
xs_corr_eyield = mfd_eps.dropna(subset = ["afd", "Earnings Yield MFD"]).groupby("date").apply(lambda x: x[["afd","Earnings Yield MFD"]].corr(method = "spearman").loc["afd","Earnings Yield MFD"])
xs_corr_eassets = mfd_eps.dropna(subset = ["afd", "EPS to Assets MFD"]).groupby("date").apply(lambda x: x[["afd","EPS to Assets MFD"]].corr(method = "spearman").loc["afd","EPS to Assets MFD"])

xs_corr = pd.merge(xs_corr_afd.to_frame(name = "MFD - AFD"),
                   xs_corr_eps.to_frame(name = "EPS MFD - AFD"),
                   left_index = True, right_index = True, how = "outer")

xs_corr = pd.merge(xs_corr,
                   xs_corr_eyield.to_frame(name = "Earnings Yield MFD - AFD"),
                   left_index = True, right_index = True, how = "outer")

xs_corr = pd.merge(xs_corr,
                   xs_corr_eassets.to_frame(name = "Earnings-to-Assets MFD - AFD"),
                   left_index = True, right_index = True, how = "outer")

fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=1000, sharex=True)
sns.kdeplot(xs_corr, ax = ax, palette = "rocket")
leg = ax.get_legend()
leg.set_frame_on(False)
for i in range(len(xs_corr.columns)):
    col = xs_corr.columns[i]
    xval = xs_corr[col].mean()
    ax.axvline(x = xval, color = colors[i], linestyle = "--")
    if i == 0:
        offset = -0.06
    else:
        offset = 0.01    
    plt.text(xval+offset, 0.94, "%.2f" % xval, transform=ax.get_xaxis_transform(), color = colors[i])

fig.tight_layout()
fig.savefig(DATA_RESULTS + "fig2.pdf")

# %% Figure: Median AFD Rank per MFD Rank (also Earnings Yield MFD)

mfd_eps_ranks = mfd_eps.copy()
mfd_eps_ranks = mfd_eps_ranks[mfd_eps_ranks.date > "1982-12-31"]
mfd_eps_ranks = mfd_eps_ranks[mfd_eps_ranks.date <= "2021-12-31"]

mfd_eps_ranks["mfd_rank"] = mfd_eps_ranks.groupby("date")["mfd"].transform(lambda x: pd.qcut(x, 10, labels = False, duplicates="drop"))+1
mfd_eps_ranks["afd_rank"] = mfd_eps_ranks.groupby("date")["afd"].transform(lambda x: pd.qcut(x, 10, labels = False, duplicates="drop"))+1
mfd_eps_ranks["yield_rank"] = mfd_eps_ranks.groupby("date")["Earnings Yield MFD"].transform(lambda x: pd.qcut(x, 10, labels = False, duplicates="drop"))+1

mfd_eps_ranks["yyyy"] = mfd_eps_ranks.date.dt.to_period("Y").apply(lambda x: x.year)

# rank comp
afd_mfd_rank = mfd_eps_ranks.groupby(["yyyy","date","mfd_rank"]).afd_rank.median().groupby(["yyyy","mfd_rank"]).mean().to_frame(name = "afd_rank").reset_index()
afd_yield_rank = mfd_eps_ranks.groupby(["yyyy","date","yield_rank"]).afd_rank.median().groupby(["yyyy","yield_rank"]).mean().to_frame(name = "afd_rank").reset_index()

afd_yield_rank = afd_yield_rank.rename(columns = {"yield_rank": "control_rank"})
afd_yield_rank["typ"] = "Earnings Yield MFD"

afd_mfd_rank = afd_mfd_rank.rename(columns = {"mfd_rank": "control_rank"})
afd_mfd_rank["typ"] = "MFD"

rank_comp = pd.concat([afd_mfd_rank, afd_yield_rank])
rank_comp = rank_comp[rank_comp.control_rank.isin([1,5,10])]
rank_comp.control_rank = rank_comp.control_rank.astype(int)

# value comp
afd_mfd_value = mfd_eps_ranks.groupby(["yyyy","date","mfd_rank"]).afd.median().groupby(["yyyy","mfd_rank"]).mean().to_frame(name = "afd_rank").reset_index()
afd_yield_value = mfd_eps_ranks.groupby(["yyyy","date","yield_rank"]).afd.median().groupby(["yyyy","yield_rank"]).mean().to_frame(name = "afd_rank").reset_index()

afd_yield_value = afd_yield_value.rename(columns = {"yield_rank": "control_rank"})
afd_yield_value["typ"] = "Earnings Yield MFD"

afd_mfd_value = afd_mfd_value.rename(columns = {"mfd_rank": "control_rank"})
afd_mfd_value["typ"] = "MFD"

val_comp = pd.concat([afd_mfd_value, afd_yield_value])
val_comp = val_comp[val_comp.control_rank.isin([1,5,10])]
val_comp.control_rank = val_comp.control_rank.astype(int)

fig = plt.figure(constrained_layout=True, figsize=(width,2.5*height), dpi=300)
subfigs = fig.subfigures(nrows=2, ncols=1)

subfigs[0].suptitle("Panel A: AFD Value")
axs = subfigs[0].subplots(nrows=1, ncols=1, sharey = False)
sns.lineplot(data=val_comp, 
             x="yyyy", y="afd_rank", hue="control_rank",
             color = colors[:-1], style = "typ", ax = axs)

axs.set_ylim(0, 0.25)
handles, labels = axs.get_legend_handles_labels()
axs.legend(handles=handles[1:4]+ handles[5:], labels=labels[1:4] + labels[5:])
axs.set(xlabel='', ylabel='AFD')

subfigs[1].suptitle("Panel B: AFD Rank")
axs = subfigs[1].subplots(nrows=1, ncols=1, sharey = False)
sns.lineplot(data=rank_comp, 
             x="yyyy", y="afd_rank", hue="control_rank",
             color = colors[:-1], style = "typ", ax = axs)

axs.set_ylim(1, 10)
axs.set(xlabel='', ylabel='AFD Rank')
axs.get_legend().remove()
fig.savefig(DATA_RESULTS + "ia_fig_i2.pdf")

# %% Run Fama-MacBeth regressions
norm_chars = get_norm_chars("unnorm")
mfd_eps = mfd_eps.reset_index()
mfd_eps = mfd_eps[["date","permno","mfd","ret_exc_lead1m"] + names]
mfd_eps = pd.merge(mfd_eps, norm_chars, on = ["permno","date"])
mfd_eps = mfd_eps.set_index(["date","permno"])

from linearmodels import FamaMacBeth

flag_winsorize = True
flag_standardize = True

def wins(x, bds):
    q = x.quantile(bds)
    x.loc[x<=q.iloc[0]] = q.iloc[0]
    x.loc[x>=q.iloc[1]] = q.iloc[1]
    return x

# First set of controls

controls = ['MFD','SUE', 'AG', 'MOM', 'ILLIQ', 'OP', 'IVOL', 'BETA',
 'SIZE', 'BM', 'MAX', 'TURN', 'STR']

cols_to_win = [mapper[x] for x in controls]

results = []
for name in names:

    tmp = mfd_eps.dropna(subset = name).copy()
    tmp = tmp.drop(columns = ["mfd"])
    tmp = tmp.rename(columns = {name: "mfd"})

    # arrange for FMB
    tmp = tmp.sort_index()

    # Winsorize data
    if flag_winsorize:
        for c in cols_to_win:
            tmp[c] = wins(tmp[c], [0.005, 0.995])

    if flag_standardize:
        for c in cols_to_win:
            tmp[c] = tmp.groupby("date")[c].transform(lambda x: (x-x.mean())/x.std())
            
    # conform to expected input for linearmodels FMB implementation using (entity, time) as MultiIndex
    tmp = tmp.reset_index().set_index(["permno","date"]).sort_index()
    
    needed_cols = [mapper[co] for co in controls]
    ttmp = tmp[needed_cols + ["ret_exc_lead1m"]].copy()
    y = ttmp.ret_exc_lead1m*100
    x = ttmp[needed_cols]
    x["Const"] = 1
    mod = FamaMacBeth(y, x)
    res = mod.fit(cov_type="kernel", kernel="newey-west")
    results.append(res)
    
fmb_to_table(results, DATA_RESULTS + "ia_tab_h2", renamer = mapper_rev, 
             colnames = names, 
             prec = 2)

# %% Decomp

from analysis_disagreement_proxies import wins, decompose_effect_multi
flag_winsorize = True
winsorize_levels = [0.005, 0.995]
num_format="%.2f"

stage1 = []
stage2 = []

mfd_eps = mfd_eps.reset_index()

for name in names:
    
    test = mfd_eps.copy().dropna(subset = ["mfd", name])

    for i in ["mfd", name]:
        test[f"{i}_rank"] = test.groupby("date")[i].transform(lambda x: pd.qcut(x, 10, labels = False, duplicates="drop"))+1
    test["d_index"] = test[[c+"_rank" for c in ["mfd", name]]].mean(axis = 1)

    stage_1, stage_2 = decompose_effect_multi(test.copy().set_index(["permno","date"]).sort_index(),
                                              "d_index", candidates = ["mfd", name],
                                              ret_def = "ret_exc_lead1m"
                                              )
    
    stage1.append(stage_1)
    stage2.append(stage_2)

# do table

tabular = Tabular("l" + "c" * 6, booktabs=True)
tabular.add_row([MultiColumn(size = 7, align = "l", data = NoEscape("Panel A: $MFD^{R,E}$ on Excess Returns"))])
tabular.add_hline()
tabular.add_row([""] + [MultiColumn(data = x, size = 2, align = "c") for x in names])
tabular.add_hline(2,3, cmidruleoption="lr")
tabular.add_hline(4,5, cmidruleoption="lr")
tabular.add_hline(6,7, cmidruleoption="lr")

const = ["Const"]
slope = [dollar("MFD^{R,E}")]
for i in range(len(stage1)):
    st1 = stage1[i]
    c, ct = st1.params["Const"], st1.tstats["Const"]
    const.append(num_format % c + sig_stars(ct))
    const.append("({})".format(num_format % ct))

    c, ct = st1.params["d_index"], st1.tstats["d_index"]
    slope.append(num_format % c + sig_stars(ct))
    slope.append("({})".format(num_format % ct))

tabular.add_row(const)
tabular.add_row(slope)

tabular.add_hline()
tabular.add_row([MultiColumn(size = 7, align = "l", data = NoEscape("Panel B: Decomposition of $MFD^{R,E}$"))])
tabular.add_hline()
tabular.add_row([""] + [MultiColumn(data = x, size = 2, align = "c") for x in names])
tabular.add_hline(2,3, cmidruleoption="lr")
tabular.add_hline(4,5, cmidruleoption="lr")
tabular.add_hline(6,7, cmidruleoption="lr")


mfd_r = [dollar("MFD^{R}")]
mfd_e = [dollar("MFD^{E}")]
res = ["Residual"]
for i in range(len(stage2)):
    st2 = stage2[i]
    
    c, ct = st2["ratio"].iloc[0]*100, st2["ratio_t"].iloc[0]
    mfd_r.append(num_format % c + sig_stars(ct))
    mfd_r.append("({})".format(num_format % ct))
    
    c, ct = st2["ratio"].iloc[1]*100, st2["ratio_t"].iloc[1]
    mfd_e.append(num_format % c + sig_stars(ct))
    mfd_e.append("({})".format(num_format % ct))
    
    c, ct = st2["ratio"].iloc[2]*100, st2["ratio_t"].iloc[2]
    res.append(num_format % c + sig_stars(ct))
    res.append("({})".format(num_format % ct))
    
tabular.add_row(mfd_r)
tabular.add_row(mfd_e)
tabular.add_row(res)

filename = DATA_RESULTS + "ia_tab_h3"

tabular.generate_tex(filename)
print("Writen to {}".format(filename))
    
    