"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Prepares test portfolios for 3-pass FMB of Giglio and Xiu (2021)
"""

from pathlib import Path

def get_project_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2]
    else:
        # interactive / notebook / IDE execution
        return Path.cwd().resolve().parents[1]

ROOT = get_project_root()

import pandas as pd
import glob
import os

#%%
# Read in and clean raw Bryzgalova et al. (2025) data
files = glob.glob(os.path.join(ROOT,"data/external/aptree_3_characteristics/**.csv"))

data = pd.DataFrame()

for file in files:
    tmp = pd.read_csv(file)
    tmp.Date = pd.to_datetime(tmp.Date, format="%Y%m").dt.to_period("M").dt.to_timestamp("M")
    tmp = tmp.set_index("Date")
    tmp.columns = [file.split(os.path.sep)[-1].replace(".csv", "") + "_" + str(x) for x in tmp.columns]
    if data.empty:
        data = tmp.copy()
    else:
        data = data.merge(tmp, left_index=True, right_index=True)
    data = data.dropna(how="any")

data = data.dropna(axis=0)
data.index.name = "date"
data.index = data.index - pd.DateOffset(months=1)

#%% Keep common dates as GX/FMB controls

path = os.path.join(ROOT,"data/derived/gx_inputs")
control_path = os.path.join(path,'controls')

# Read in controls
vw_controls = pd.read_csv(os.path.join(control_path, 'vw_gx_controls.csv'))
vw_controls.date = pd.to_datetime(vw_controls.date, format="%Y-%m-%d").dt.to_period("M").dt.to_timestamp("M")
vw_controls = vw_controls.set_index('date')

ew_controls = pd.read_csv(os.path.join(control_path, 'ew_gx_controls.csv'))
ew_controls.date = pd.to_datetime(ew_controls.date, format="%Y-%m-%d").dt.to_period("M").dt.to_timestamp("M")
ew_controls = ew_controls.set_index('date')

# Keep common dates
common_dates = data.index.intersection(vw_controls.index)

data = data.loc[common_dates]
vw_controls = vw_controls.loc[common_dates]
ew_controls = ew_controls.loc[common_dates]

# Save data
save_path = os.path.join(path,'aptree')
data.to_csv(os.path.join(save_path, 'aptree_test_assets.csv'))
vw_controls.to_csv(os.path.join(save_path, 'vw_controls_aptree.csv'))
ew_controls.to_csv(os.path.join(save_path, 'ew_controls_aptree.csv'))
