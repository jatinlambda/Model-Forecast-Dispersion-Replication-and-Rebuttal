"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Analyses: 3-Pass Fama-MacBeth regression of Giglio and Xiu (2023)
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
DATA_GX = ROOT / "data" / "derived" / "gx_fmb"
DATA_GX_INPUT = ROOT / "data" / "derived" / "gx_inputs"

import pandas as pd
import numpy as np
import glob
from sklearn.decomposition import PCA

import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

cm = 1 / 2.54
width = 15.92 * cm
height = 20 * cm

matplotlib.rcParams.update({"font.size": 8})
matplotlib.rcParams["axes.xmargin"] = 0.02
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.edgecolor"] = "0.15"
matplotlib.rcParams["axes.linewidth"] = 1.25

num_format="%.3f"
num_format_t="%.2f"

from helper import sig_stars, num_format_t

from pylatex import MultiColumn, Tabular


# %% Read in data
def get_eigen_data(files, divider = 1):
    
    data = pd.DataFrame()
    for file in files:
        tmp = pd.read_csv(file)
        tmp.date = pd.to_datetime(tmp.date, format = "%Y-%m-%d").dt.to_period("M").dt.to_timestamp("M")
        tmp = tmp.set_index("date")
        tmp.columns = [file.split(os.path.sep)[-1].replace(".csv","") + "_" + str(x) for x in tmp.columns ]
        if data.empty:
            data = tmp.copy()
        else:
            data = data.merge(tmp, left_index = True, right_index = True)
        data = data.dropna(how = "any")
    
    data = data / divider
    
    cov = data.cov()
    
    pca = PCA(n_components = n_components)
    pca.fit(cov)
    
    eigen_data = pd.DataFrame({"eigen":pca.singular_values_,
                               "eigen_nbr": np.array( range(1,n_components+1))})
    
    return eigen_data

# NEW FILE PATHS
files_ff = glob.glob(os.path.join(DATA_GX_INPUT, "ff","ew_**.csv"))
files_jkp = glob.glob(os.path.join(DATA_GX_INPUT, "jkp","ew_**.csv"))
files_ffjkp = files_ff + files_jkp

files_aptree = glob.glob( os.path.join(DATA_GX_INPUT / "aptree", "aptree_test_assets.csv" ))

n_components = 20
eigen_ffjkp = get_eigen_data(files_ffjkp, divider = 100)
eigen_aptree = get_eigen_data(files_aptree)

# %% do figure
n_cutoff = 5

fig = plt.figure(constrained_layout=True, figsize=(width, height), dpi=300)
subfigs = fig.subfigures(nrows=2, ncols=1)

subfigs[0].suptitle("Panel A: FF + JKP")
axs = subfigs[0].subplots(nrows=1, ncols=2, sharey = False)
sns.barplot(eigen_ffjkp, x="eigen_nbr", y="eigen", ax = axs[0],color = "k")
axs[0].set_xlabel(f"Eigenvalues 1 to {n_components}")
axs[0].tick_params(axis = "x", rotation=45)
axs[0].set_ylabel("")

sns.barplot(eigen_ffjkp[eigen_ffjkp.eigen_nbr>=n_cutoff], x="eigen_nbr", y="eigen", ax = axs[1], color = "k")
axs[1].set_xlabel(f"Eigenvalues {n_cutoff} to {n_components}")
axs[1].set_ylabel("")

axs[1].tick_params(axis = "x", rotation=45)

subfigs[1].suptitle("Panel B: AP Tree")
axs = subfigs[1].subplots(nrows=1, ncols=2, sharey = False)
sns.barplot(eigen_aptree, x="eigen_nbr", y="eigen", ax = axs[0],color = "k")
axs[0].set_xlabel(f"Eigenvalues 1 to {n_components}")
axs[0].tick_params(axis = "x", rotation=45)
axs[0].set_ylabel("")

sns.barplot(eigen_aptree[eigen_aptree.eigen_nbr>=n_cutoff], x="eigen_nbr", y="eigen", ax = axs[1], color = "k")
axs[1].set_xlabel(f"Eigenvalues {n_cutoff} to {n_components}")
axs[1].tick_params(axis = "x", rotation=45)
axs[1].set_ylabel("")

fig.savefig(DATA_RESULTS + "ia_figc1.pdf")

# %% Do table with results
def reformatter_results(data, test_asset, weighting):
    data = data[[x for x in data.columns if not("SE" in x)]]
    data.Row = data.Row.apply(lambda x: x[1:])
    data = data.set_index("Row")
    data = data.stack().reset_index()
    data["npfs"] = data.level_1.apply(lambda x: x.split("_")[0])
    data["level_1"] = data.level_1.apply(lambda x: x.split("_")[1])
    data = data.rename(columns = {0: "val"})
    data = data[data.Row.isin(["mfd_HL"])]
    data.npfs = data.npfs.astype(int)
    data = data[(data.npfs >= 6) & (data.npfs <= 9)]
    data["testassets"] = test_asset
    data["weighting"] = weighting
    data = data.drop(columns = "Row")
    data = data.pivot(columns = "level_1", index = ["npfs", "testassets", "weighting"], values = "val")
    data = data.reset_index()
    
    return data

aptree_vw = pd.read_excel(DATA_GX / "PCAFM_results_aptree_vw__nozero.xlsx")
aptree_vw = reformatter_results(aptree_vw, "AP Tree", "VW")

aptree_ew = pd.read_excel(DATA_GX / "PCAFM_results_aptree_ew__nozero.xlsx")
aptree_ew = reformatter_results(aptree_ew, "AP Tree", "EW")

jkpff_vw = pd.read_excel(DATA_GX / "PCAFM_results_FF_JKP_nozero_vw_.xlsx")
jkpff_vw = reformatter_results(jkpff_vw, "FF + JKP", "VW")
jkpff_ew = pd.read_excel(DATA_GX / "PCAFM_results_FF_JKP_nozero_ew_.xlsx")
jkpff_ew = reformatter_results(jkpff_ew, "FF + JKP", "EW")

results = pd.concat([aptree_vw, aptree_ew, jkpff_vw, jkpff_ew])

results = results.sort_values(["npfs", "testassets", "weighting"],
                              ascending = [True, False, True])


tabular = Tabular("l" + "c" * 4, booktabs=True)
colnames = ["Nbr Factors", MultiColumn(2, data = "FF + JKP", align = "c"),
            MultiColumn(2, data = "AP Tree", align = "c")]
tabular.add_row(colnames)
tabular.add_hline(1, 1)
tabular.add_hline(2, 3, cmidruleoption=("lr"))
tabular.add_hline(4, 5, cmidruleoption=("lr"))

colnames = ["", "EW", "VW", "EW", "VW"]
tabular.add_row(colnames)
tabular.add_hline()

for n, grp in results.groupby("npfs"):
    to_add1 = [n]
    to_add2 = [""]
    
    for b,t in zip(grp.Beta, grp.tStat):
    
        to_add1.append(
            num_format % b + sig_stars(t)
            )
        to_add2.append(
            "({})".format(num_format_t % t)
            )
    
    tabular.add_row(to_add1)
    tabular.add_row(to_add2)

fname = DATA_RESULTS + "ia_tab_c1"
tabular.generate_tex(fname)


