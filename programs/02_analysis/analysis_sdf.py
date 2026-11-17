"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Analyses: Pricing of MFD according to non-linear SDFs
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
import statsmodels.api as sm
from helper import do_strategy_overview, pfs, bp_min_n, bps, get_factors
import seaborn as sns
import matplotlib.pyplot as plt

cm = 1 / 2.54
sns.set_theme(style="ticks", rc={"font.family": "serif"})
    
# %% Load Data
mfd = pd.read_parquet(DATA_MFD / "main_data.pq")

ap_factors = get_factors()
mfd = mfd.dropna(subset ="ret_exc_lead1m")

# %% univariate portfolio sorts
_, mfd_vw = do_strategy_overview(mfd.copy(), "vw", "mfd", ap_factors, pfs, bps, bp_min_n)
_, mfd_ew = do_strategy_overview(mfd.copy(), "ew", "mfd", ap_factors, pfs, bps, bp_min_n)

# %% Do figure
df = []

# pelger et al sdf
sdf = pd.read_excel(DATA_EXTERNAL / "NLSDF Time-Series.xlsx", skiprows = 6)
sdf.date = (pd.to_datetime(sdf.date, format = "%Y-%m-%d").dt.to_period("M") - 1)
sdf = sdf[["date","SDF"]].dropna()

mfd_sdf = pd.merge(sdf, mfd_vw.reset_index(level = 1, drop = True).rename(columns = {"XRet": "mfd_vw"}), on = "date")
mfd_sdf = pd.merge(mfd_sdf, mfd_ew.reset_index(level = 1, drop = True).rename(columns = {"XRet": "mfd_ew"}), on = "date")

mfd_sdf = mfd_sdf.set_index("date")
mfd_sdf[["mfd_vw","mfd_ew"]] /= 100
mfd_sdf.SDF /= mfd_sdf.SDF.mean()

mfd_sdf["prod_vw"] = mfd_sdf.SDF * mfd_sdf.mfd_vw
mfd_sdf["prod_ew"] = mfd_sdf.SDF * mfd_sdf.mfd_ew
mfd_sdf["const"] = 1

model_ew = sm.OLS(mfd_sdf["prod_ew"], mfd_sdf[["const"]]).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
strat_ew = sm.OLS(mfd_sdf["mfd_ew"], mfd_sdf[["const"]]).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

df.append(pd.DataFrame({"Weighting": "EW", "SDF Model": "DL-SDF", 
                        "Return Type": "Strategy Return", 
                        "Value": strat_ew.params["const"],
                       "t-stat": strat_ew.tvalues["const"]}
                       , index = [0]))
df.append(pd.DataFrame({"Weighting": "EW", "SDF Model": "DL-SDF", "Return Type": "Pricing Error", 
                        "Value": model_ew.params["const"],
                       "t-stat": model_ew.tvalues["const"]}, index = [0]))

model_vw = sm.OLS(mfd_sdf["prod_vw"], mfd_sdf[["const"]]).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
strat_vw = sm.OLS(mfd_sdf["mfd_vw"], mfd_sdf[["const"]]).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

df.append(pd.DataFrame({"Weighting": "VW", "SDF Model": "DL-SDF", 
                        "Return Type": "Strategy Return", 
                        "Value": strat_vw.params["const"],
                       "t-stat": strat_vw.tvalues["const"]}
                       , index = [0]))
df.append(pd.DataFrame({"Weighting": "VW", "SDF Model": "DL-SDF", "Return Type": "Pricing Error", 
                        "Value": model_vw.params["const"],
                       "t-stat": model_vw.tvalues["const"]}, index = [0]))

# cong et al sdf
ptree = pd.read_csv(DATA_EXTERNAL / "factors_p_tree.csv")
ptree["date"] = pd.date_range(start='1/1/1981', end='12/31/2020', freq='M').to_period("M")-1

mfd_ptree = pd.merge(ptree, mfd_vw.reset_index(level = 1, drop = True).rename(columns = {"XRet": "mfd_vw"}), on = "date")
mfd_ptree = pd.merge(mfd_ptree, mfd_ew.reset_index(level = 1, drop = True).rename(columns = {"XRet": "mfd_ew"}), on = "date")

mfd_ptree[["mfd_vw","mfd_ew"]] /= 100
mfd_ptree.X1 /= mfd_ptree.X1.mean()

mfd_ptree["prod_vw"] = mfd_ptree.X1 * mfd_ptree.mfd_vw
mfd_ptree["prod_ew"] = mfd_ptree.X1 * mfd_ptree.mfd_ew

mfd_ptree["const"] = 1

model_ew = sm.OLS(mfd_ptree["prod_ew"], mfd_ptree[["const"]]).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
strat_ew = sm.OLS(mfd_ptree["mfd_ew"], mfd_ptree[["const"]]).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

df.append(pd.DataFrame({"Weighting": "EW", "SDF Model": "T-SDF", 
                        "Return Type": "Strategy Return", 
                        "Value": strat_ew.params["const"],
                       "t-stat": strat_ew.tvalues["const"]}
                       , index = [0]))
df.append(pd.DataFrame({"Weighting": "EW", "SDF Model": "T-SDF", "Return Type": "Pricing Error", 
                        "Value": model_ew.params["const"],
                       "t-stat": model_ew.tvalues["const"]}, index = [0]))
model_vw = sm.OLS(mfd_ptree["prod_vw"], mfd_ptree[["const"]]).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
strat_vw = sm.OLS(mfd_ptree["mfd_vw"], mfd_ptree[["const"]]).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

df.append(pd.DataFrame({"Weighting": "VW", "SDF Model": "T-SDF", 
                        "Return Type": "Strategy Return", 
                        "Value": strat_vw.params["const"],
                       "t-stat": strat_vw.tvalues["const"]}
                       , index = [0]))
df.append(pd.DataFrame({"Weighting": "VW", "SDF Model": "T-SDF", "Return Type": "Pricing Error", 
                        "Value": model_vw.params["const"],
                       "t-stat": model_vw.tvalues["const"]}, index = [0]))
df = pd.concat(df)
df_sorted = df.sort_values(['Weighting', 'SDF Model', 'Return Type'])

df_sorted.Value *=100

colors = sns.color_palette("rocket",2)

fig = plt.figure(figsize=(15.92 * cm, 9 * cm))
gs = fig.add_gridspec(2, 2, height_ratios=[10, 1.5])

ax_ew = fig.add_subplot(gs[0, 0])
ax_vw = fig.add_subplot(gs[0, 1])
ax_legend = fig.add_subplot(gs[1, :])

axes_list = [ax_ew, ax_vw]
hue_order_map = {} # Dictionary to store hue order for each axis

for ax, weighting_type in zip(axes_list, ['EW', 'VW']):
    df_subset = df_sorted[df_sorted['Weighting'] == weighting_type]
    sns.barplot(
        data=df_subset,
        x='SDF Model',
        y='Value',
        hue='Return Type',
        palette=colors,
        ax=ax
    )
    
    handles, labels = ax.get_legend_handles_labels()
    hue_order_map[ax] = labels # Store the hue order for later

    if ax.get_legend():
        ax.get_legend().remove()
        
    # Style each subplot
    ax.set_title(f'{weighting_type} Weighting', fontsize=8)
    ax.set_ylim(bottom=-3.25, top=0.0)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(axis='both', which='major', labelsize=8)

for ax, weighting_type in zip(axes_list, ['EW', 'VW']):
    x_order = [tick.get_text() for tick in ax.get_xticklabels()]
    hue_order = hue_order_map[ax] # Retrieve the saved hue order for this axis

    df_subset = df_sorted[df_sorted['Weighting'] == weighting_type]

    for i, container in enumerate(ax.containers):
        current_hue = hue_order[i]
        
        labels_for_container = []
        for x_val in x_order:
            row = df_subset[
                (df_subset['SDF Model'] == x_val) & 
                (df_subset['Return Type'] == current_hue)
            ].iloc[0]
            
            label = f"{row['Value']:.2f}\n({row['t-stat']:.2f})"
            labels_for_container.append(label)
        
        ax.bar_label(container, labels=labels_for_container, fontsize=7, padding=3)

ax_legend.legend(handles, labels, loc='center', title='', fontsize=8, ncol=2, frameon=False)
ax_legend.get_legend().get_title().set_fontsize(8)
ax_legend.axis('off')

# --- 6. Final Touches ---
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(DATA_RESULTS + "ia_fig_d1.pdf", dpi=300) 

#plt.show()