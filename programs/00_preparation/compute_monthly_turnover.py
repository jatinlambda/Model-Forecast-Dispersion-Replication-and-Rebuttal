"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Constructs monthly turnover
"""

import pandas as pd

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

# %%
crsp = pd.read_csv(DATA_RAW / "usa.csv", usecols=[
    "eom", "permno", "crsp_shrcd", "crsp_exchcd", "source_crsp", "shares", "tvol"])

crsp["date"] = pd.to_datetime(crsp.eom, format = "%Y%m%d")
crsp = crsp.drop(columns = "eom")
crsp = crsp[crsp.crsp_shrcd.isin([10,11,12])]
crsp = crsp[crsp.crsp_exchcd.isin([1,2,3])]
crsp = crsp[crsp.source_crsp == 1]

# fill trading vol with 0 if it is na to make it small but not drop it
crsp.tvol = crsp.tvol.fillna(0)
crsp.tvol *= 100
crsp.shares *= 10**6

crsp["turnover_21d"] = crsp.tvol / crsp.shares

crsp[["date","permno","turnover_21d"]].to_pickle(DATA_DERIVED / "monthly_turnover.pkl.gz", compression = "gzip")



