"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Analyses: Stock-level drivers of MFD and AFD
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
from helper import get_norm_chars, get_green, get_misp, get_ior
from linearmodels import FamaMacBeth
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

cm = 1 / 2.54
width = 15.92 * cm
height = 6 * cm

colors = sns.color_palette("rocket",2)

matplotlib.rcParams.update({"font.size": 8})
matplotlib.rcParams["axes.xmargin"] = 0.02
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.edgecolor"] = "0.15"
matplotlib.rcParams["axes.linewidth"] = 1.25

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

# mispricing score from SYY
misp = get_misp()
misp.columns = [x.lower() for x in misp.columns]

# Load markit data
markit = pd.read_parquet(DATA_RAW / "markit_shortfee.pq")
markit = markit.rename(columns = {"ym": "date"})
markit.date = markit.date.dt.to_period("M").dt.to_timestamp("M")
markit = markit[["IndicativeFee","permno","date"]]
markit = markit[markit.date >= "2004-01-01"]

# Load Institutional Ownership
ior = get_ior()

# same dates
mfd = mfd.merge(green, on = ["date","permno"], how ="left")
mfd = mfd.merge(misp, on = ["date","permno"], how ="left")
mfd = mfd.merge(markit, on = ["date","permno"], how ="left")
mfd = mfd.merge(ior[["date","permno","ior"]], on = ["date","permno"], how = "left")

# %% Complexity etc
mapping_gvkey = pd.read_csv(DATA_RAW / "usa.csv", usecols = ["date","permno","gvkey"])
mapping_gvkey = mapping_gvkey.dropna(subset = ["permno","gvkey"])
mapping_gvkey["gvkey"] = mapping_gvkey["gvkey"].astype("int64")
mapping_gvkey["date"] = pd.to_datetime(mapping_gvkey.date, format = "%Y%m%d").dt.to_period("M").dt.to_timestamp("M")

mfd = mfd.merge(mapping_gvkey[["permno","date","gvkey"]], on = ["permno","date"], how = "left")

mapping_cik = pd.read_csv(DATA_RAW / "compustat_gvkey_cik_link.zip")
mapping_cik = mapping_cik.dropna(subset=["gvkey","cik"])
mapping_cik["cik"] = mapping_cik["cik"].astype("int64")
mapping_cik["datadate"] = pd.to_datetime(mapping_cik.datadate, format = "%Y-%m-%d")
mapping_cik = mapping_cik.sort_values(["datadate", "gvkey"])

tmp = mfd[["date","permno","gvkey"]].dropna()
tmp.gvkey = tmp.gvkey.astype("int64")

tmp = tmp.sort_values("date")
tmp = pd.merge_asof(tmp, mapping_cik[["datadate","cik","gvkey"]], left_on ="date",right_on = "datadate", by = "gvkey")
tmp = tmp.dropna(subset = "cik")
tmp.cik = tmp.cik.astype("int64")

complexity = pd.read_csv(DATA_EXTERNAL / "Loughran_McDonald_Complexity.csv")
complexity["filingdate"] = pd.to_datetime(complexity.filingdate, format = "%Y%m%d") 
complexity = complexity.sort_values(["filingdate","cik"])

tmp_cik = pd.merge_asof(tmp, complexity, left_on = "date", right_on = "filingdate", by = "cik")
tmp_cik = tmp_cik.dropna(subset = "complexity")
tmp_cik= tmp_cik[["date","permno","complexity","log_netfilesize"]]

mfd_driver = mfd.merge(tmp_cik, on = ["date","permno"], how = "left")

mfd_driver = mfd_driver.set_index(["date","permno"])
mfd_driver = mfd_driver.sort_index()

mfd_driver = mfd_driver[["mfd","afd","niq_su","at_gr1","ret_12_1","ami_126d",
                         "ope_be","market_equity","be_me","turnover_126d",
                         "ret_1_0","sale_gr3","taccruals_at","complexity",
                         "log_netfilesize", "age", "ocfq_saleq_std", "tangibility", 
                         "ivol_capm_252d","misp", "beta_60m", "IndicativeFee", "ior", "niq_at"]]

# Winsorize data
if flag_winsorize:
    for c in mfd_driver.columns:
        print("Winsorizing {}".format(c))
        mfd_driver[c] = wins(mfd_driver[c], [0.005, 0.995])

if flag_standardize:
    for c in mfd_driver.columns:
        print("Standardizing {}".format(c))
        mfd_driver[c] = mfd_driver.groupby("date")[c].transform(lambda x: (x-x.mean())/x.std())
        
# conform to expected input for linearmodels FMB implementation using (entity, time) as MultiIndex
mfd_driver = mfd_driver.reset_index().sort_values(["date","permno"]).set_index(["permno","date"])

# %% regressions

mapper_driver = {'SUE': 'niq_su',
 'AG': 'at_gr1',
 'MOM': 'ret_12_1',
 'ILLIQ': 'ami_126d',
 'OP': 'ope_be',
 'IVOL': 'ivol_capm_252d',
 'BETA': 'beta_60m',
 'SIZE': 'market_equity',
 'BM': 'be_me',
 'MAX': 'rmax5_21d',
 'TURN': 'turnover_126d',
 'STR': 'ret_1_0',
 'MFD': 'mfd',
 'AFD': 'afd',
 'INST': 'ior',
 'UTIL': 'Utilisation',
 'IFEE': 'IndicativeFee',
 "SALES": "sale_gr3",
 "ACCRUALS": "taccruals_at",
 "COMPLEX": "complexity",
 "FSIZE": "log_netfilesize",
 "AGE": "age",
 "CFV": "ocfq_saleq_std",
 "TAN": "tangibility",
 "MISP": "misp",
 "BORROWFEE": "IndicativeFee",
 "INST": "ior",
 "RETASSET": "niq_at"}

mapper_driver_rev = {v: k for k, v in mapper_driver.items()}

core_controls = ["SIZE", "BM", "BETA"]

to_add = ["SUE", "OP", "SALES", "RETASSET", "ACCRUALS", "FSIZE", "TURN", "CFV", "MISP", "INST"]

# MFD and AFD
mfd_driver.loc[mfd_driver.index.get_level_values(1)<="1983-01-01","afd"] = np.nan

results = []
for ac in to_add:
    print(f"Working on {ac}")
    for target in ["mfd","afd"]:
        needed_cols = [mapper_driver[co] for co in core_controls + [ac]]
        ttmp = mfd_driver[needed_cols + [target]].copy().dropna()
        y = ttmp[target]
        x = ttmp[needed_cols]
        x["Const"] = 1
        mod = FamaMacBeth(y, x)
        res = mod.fit(cov_type="kernel", kernel="newey-west")
        
        res = pd.DataFrame({
            "target": target.upper(),
            "exog": ac,
            "slope": res.params[mapper_driver[ac]],
            "tstat": res.tstats[mapper_driver[ac]]}, index = [0])
        
        results.append(res)
    
results = pd.concat(results)      


# %% Do plotting
fig, ax = plt.subplots(1, 1, figsize=(1.5*width, height), dpi=1000, sharex=True)
sns.barplot(data=results, ax = ax,
            x = "exog",
            y = "slope",
            hue = "target",
            palette = colors)

hue_order = results['target'].unique()

for i, container in enumerate(ax.containers):
    current_hue = hue_order[i]
    subset = results[results['target'] == current_hue]

    # Create the custom labels with slope and tstat, formatted to 2 decimals
    labels = [f"{row.slope:.2f}\n({row.tstat:.2f})" for index, row in subset.iterrows()]

    # Apply the labels to the bars in the container
    ax.bar_label(container,
                 labels=labels,
                 label_type='edge', # Place label at the "edge" of the bar
                 fontsize=6,
                 padding=2)

# Make more room for the labels below the bars
ax.axhline(0, color='k', linestyle='-', linewidth=0.5, zorder=1)
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min * 1.5, y_max * 1.5)
ax.set(xlabel='', ylabel='Slope Coefficient')
ax.tick_params(axis='x', rotation=15) # Rotate x-labels slightly
ax.legend(frameon=False)
fig.savefig(DATA_RESULTS + "fig3.pdf", bbox_inches='tight')


