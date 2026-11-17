"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Add next month excess return and market cap to MFD measure
"""

# %%
# imports

import pickle
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed

from pathlib import Path

def get_project_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2]
    else:
        # interactive / notebook / IDE execution
        return Path.cwd().resolve().parents[1]

ROOT = get_project_root()
DATA_DERIVED = ROOT / "data" / "derived"
DATA_MFD = ROOT / "data" / "mfd"
DATA_RAW = ROOT / "data" / "raw"

# %% Load MFD data
mfd = pd.read_csv(DATA_MFD / 'mfd.csv', index_col = 0, parse_dates = ["date"], 
                  infer_datetime_format=True,
                  usecols =["date","permno","std", "mean"]).reset_index()
mfd = mfd.rename(columns={"std":"mfd", "mean": "mfd_mean"})
mfd.mfd *=100
mfd.date = mfd.date.dt.to_period("M").dt.to_timestamp("M")
mfd = mfd.dropna(subset = "mfd")
mfd = mfd.sort_values(["date","permno"]).set_index(["date","permno"])

# %% load returns and market cap data
dtypes = {'iid': 'str'}
usa_data = pd.read_csv(DATA_RAW / 'usa.csv', dtype=dtypes,
                       usecols = ["date","permno",
                                  "crsp_shrcd", "crsp_exchcd", "prc",
                                  "ret_exc_lead1m", "market_equity"])
usa_data = usa_data[(usa_data.crsp_shrcd == 10) | (usa_data.crsp_shrcd == 11)]
usa_data = usa_data[(usa_data.crsp_exchcd == 1) | (usa_data.crsp_exchcd == 2) | (usa_data.crsp_exchcd == 3)]

usa_data = usa_data[usa_data.date>=19660731]
usa_data.dropna(subset=['permno'], inplace = True)
usa_data.date = pd.to_datetime(usa_data.date, format = "%Y%m%d").dt.to_period("M").dt.to_timestamp("M")

usa_data = usa_data.set_index(["date","permno"])
usa_data = usa_data.drop(columns = ["crsp_shrcd", "crsp_exchcd", "prc"]).rename(columns = {"market_equity": "me"})

mfd = mfd.merge(usa_data, on = ["date","permno"])
mfd = mfd.reset_index()

# %% save
mfd.to_parquet(DATA_MFD / "main_data.pq")