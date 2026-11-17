"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Analyses: aggregate MFD comomvement
"""

import pandas as pd
import numpy as np
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

from helper import (get_green, 
                    get_misp)

num_format="%.2f"

# %% Load Data
mfd = pd.read_parquet(DATA_MFD / "main_data.pq")
    
green = get_green()
mfd = mfd.merge(green, on = ["date","permno"], how ="left")

misp = get_misp()
mfd = mfd.merge(misp, on = ["date", "permno"], how= "left")

vw_mfd = mfd.dropna(subset = "mfd").groupby(["date"]).apply(lambda x: np.sum(x.mfd * x.me) / x.me.sum())
vw_afd = mfd.dropna(subset = "afd").groupby(["date"]).apply(lambda x: np.sum(x.afd * x.me) / x.me.sum())
vw_afd = vw_afd[vw_afd.index >= "1983-01-01"]
vw_misp = mfd.dropna(subset = "MISP").groupby(["date"]).apply(lambda x: np.sum(x.MISP * x.me) / x.me.sum())

joint = vw_mfd.to_frame(name = "mfd").merge(vw_afd.to_frame(name = "afd"), left_index = True, right_index = True, how = "left")
joint = joint.merge(vw_misp.to_frame(name = "MISP"), left_index = True, right_index = True, how = "left")
joint.corr(method = "spearman")
joint = joint.rolling(12).mean()
joint = joint.reset_index()
joint["mfd"] *= 12

# do figure

fig, ax1 = plt.subplots(1, 1, figsize=(width, height), dpi=1000, sharex=True)
twin1 = ax1.twinx()
twin2 = ax1.twinx()
twin2.spines.right.set_position(("axes", 1.15))

sns.lineplot(x = "date", y = "mfd", data = joint, ax = ax1, color = "k", linewidth = 2)
sns.lineplot(x = "date", y = "afd", data = joint, ax = twin1, color = colors[1], linewidth = 0.5)
sns.lineplot(x = "date", y = "MISP", data = joint, ax = twin2, color = colors[2], linewidth = 0.5)

twin1.yaxis.label.set_color(colors[1])
twin2.yaxis.label.set_color(colors[2])

twin1.tick_params(axis='y', colors=colors[1])
twin2.tick_params(axis='y', colors=colors[2])

ax1.set_xlabel('')
ax1.set_ylabel('Aggregate MFD', color='k')
twin1.set_ylabel('Aggregate AFD', color=colors[1])
twin2.set_ylabel('Aggregate Mispricing', color=colors[2])

fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(DATA_RESULTS + "fig1.pdf", dpi = 1000, )





