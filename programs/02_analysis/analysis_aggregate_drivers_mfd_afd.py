"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Analyses: drivers of aggregate MFD and AFD
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
import statsmodels.api as sm

from helper import (pfs, 
                    bp_min_n, 
                    bps, 
                    get_green, 
                    get_factors, 
                    do_strategy_overview)

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

# %% Load Data
mfd = pd.read_parquet(DATA_MFD / "main_data.pq")

ap_factors = get_factors()
mfd = mfd.dropna(subset ="ret_exc_lead1m")

# %% univariate portfolio sorts
_, mfd_vw = do_strategy_overview(mfd.copy(), "vw", "mfd", ap_factors, pfs, bps, bp_min_n)

green = get_green()
mfd = mfd.merge(green, on = ["date","permno"], how ="left")

mfd_afd = mfd.copy()
mfd_afd = mfd_afd[mfd_afd.date > "1982-12-31"]
mfd_afd = mfd_afd[mfd_afd.date <= green.date.max()]

mfd_afd.afd.isna().mean()

mfd_afd = mfd_afd.dropna(subset = "afd")

_, afd_vw = do_strategy_overview(mfd_afd.copy(), "vw", "afd", ap_factors, pfs, bps, bp_min_n)

# %% load macroeconomic time series

vix = pd.read_csv(DATA_EXTERNAL / "VIX_History.csv", delimiter=",")
vix.columns = [x.lower() for x in vix.columns]
vix["date"] = pd.to_datetime(vix["date"], format="%m/%d/%Y")
vix = vix.set_index("date")
vix.index = vix.index.to_period("M")
vix = vix.groupby(vix.index).mean()
vix /= 100
vix = vix.loc[:"2022"]
vix = vix[["close"]]

timeseries = vix.rename(columns = {"close": "vix"}).copy()

vvix = pd.read_csv(DATA_EXTERNAL / "VVIX_History.csv", delimiter=",")
vvix.columns = [x.lower() for x in vvix.columns]
vvix["date"] = pd.to_datetime(vvix["date"], format="%m/%d/%Y")
vvix = vvix.set_index("date")
vvix.index = vvix.index.to_period("M")
vvix = vvix.groupby(vvix.index).apply(lambda x: x.max() - x.min())
vvix /= 100
vvix = vvix.loc[:"2022"]

timeseries = timeseries.merge(vvix, left_index = True, right_index = True, how = "outer")

sentiment = pd.read_excel( DATA_EXTERNAL / 
    "Investor_Sentiment_Data_20220821_POST.xlsx",
    engine="openpyxl",
    sheet_name="DATA",
).dropna(how="all")
sentiment["date"] = pd.to_datetime(sentiment["yearmo"].astype("int"), format="%Y%m")
sentiment = sentiment.set_index("date")
sentiment.index = sentiment.index.to_period("M")
sentiment = sentiment["SENT_ORTH"]
sentiment = sentiment[sentiment != "."]
sentiment = sentiment.astype("f4")
sentiment = sentiment.to_frame()
sentiment = sentiment.groupby(sentiment.index).mean()
sentiment = sentiment.dropna()
sentiment = sentiment.loc["1976-08":"2022"]

timeseries = timeseries.merge(sentiment, left_index = True, right_index = True, how = "outer")

catfin = pd.read_excel(DATA_EXTERNAL / "CATFIN_2022.xlsx")
catfin["date"] = pd.to_datetime(catfin["yyyymm"], format="%Y-%m-%d")
catfin = catfin.set_index("date")
catfin.index = catfin.index.to_period("M")
catfin = catfin[["CATFIN"]]
catfin = catfin.groupby(catfin.index).mean()
catfin = catfin.loc["1976-08":"2022"]

timeseries = timeseries.merge(catfin, left_index = True, right_index = True, how = "outer")

psl_sentiment = pd.read_excel(DATA_EXTERNAL / "pls_sentiment_indices_2023_.xlsx")
psl_sentiment["date"] = pd.to_datetime(psl_sentiment["yyyymm"], format="%Y%m")
psl_sentiment = psl_sentiment.set_index("date")
psl_sentiment.index = psl_sentiment.index.to_period("M")
psl_sentiment = psl_sentiment[["PLS_SENT othogonalized (to macro econ vars)"]]
psl_sentiment = psl_sentiment.groupby(psl_sentiment.index).mean()
psl_sentiment = psl_sentiment.loc["1976-08":"2022"]

timeseries = timeseries.merge(psl_sentiment, left_index = True, right_index = True, how = "outer")

# %% merge with afd and mfd
timeseries.index += -1

to_test = list(timeseries.columns)

timeseries = timeseries.merge(mfd_vw.reset_index(level =1, drop = True).rename(columns = {"XRet": "mfd_vw"}), left_index = True,
                              right_index = True, how = "outer")

timeseries = timeseries.merge(afd_vw.reset_index(level =1, drop = True).rename(columns = {"XRet": "afd_vw"}), left_index = True,
                              right_index = True, how = "outer")

# %% do regressions
results = []
for col in to_test:
    for ts in ["mfd_vw","afd_vw"]:
        tmp = timeseries[[col, ts]].copy().dropna()
        tmp["const"] = 1
        
        mod = sm.OLS(tmp[ts],tmp[["const", col]]).fit(cov_type = "HAC", cov_kwds = {"maxlags": 6})
        
        results.append(
            pd.DataFrame(
                {"slope": mod.params[col],
                 "tstat": mod.tvalues[col],
                 "exog": col,
                 "endog": ts
                    },
                index = [0]
                )
            )
results = pd.concat(results)

results.endog = results.endog.map({"mfd_vw": "MFD", "afd_vw": "AFD"})
results.exog = results.exog.map({"vix": "VIX Index",
                                 "vvix": "VVIX Index",
                                 "CATFIN": "CATFIN",
                                 "PLS_SENT othogonalized (to macro econ vars)": "PLS Sentiment Index",
                                 "SENT_ORTH": "Baker-Wurgler (orth.)"})

# %% Do plotting
fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=1000, sharex=True)
sns.barplot(data=results, ax = ax,
            x = "exog",
            y = "slope",
            hue = "endog",
            palette = colors)

hue_order = results['endog'].unique()

# Loop through each container (one container per hue category)
for i, container in enumerate(ax.containers):
    # Get the correct subset of the data for the current container
    current_hue = hue_order[i]
    subset = results[results['endog'] == current_hue]

    # Create the custom labels with slope and tstat, formatted to 2 decimals
    labels = [f"{row.slope:.2f}\n({row.tstat:.2f})" for index, row in subset.iterrows()]

    # Apply the labels to the bars in the container
    ax.bar_label(container,
                 labels=labels,
                 label_type='edge', # Place label at the "edge" of the bar
                 fontsize=6,
                 padding=2)

# 5. Adjust plot aesthetics
# Make more room for the labels below the bars
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min * 1.2, y_max)
ax.set(xlabel='', ylabel='Slope Coefficient')
ax.tick_params(axis='x', rotation=15) # Rotate x-labels slightly
ax.legend(frameon=False)
fig.savefig(DATA_RESULTS + "/ia_fig_i1.pdf", bbox_inches='tight')


