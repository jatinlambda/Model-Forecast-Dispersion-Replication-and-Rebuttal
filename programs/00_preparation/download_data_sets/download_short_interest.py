"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Analyses: Downloads data from CRSP and constructs short interest
"""

import pandas as pd
import numpy as np
import wrds

from pathlib import Path

def get_project_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[3]
    else:
        # interactive / notebook / IDE execution
        return Path.cwd().resolve().parents[2]

ROOT = get_project_root()

DATA_RAW = ROOT / "data" / "raw"

# %%% Short Interest from Compustat
shortint_file = DATA_RAW / "short_interest_compustat.zip"
shortint = pd.read_csv(shortint_file)
shortint.datadate = pd.to_datetime(shortint.datadate, format = "%Y-%m-%d")
shortint["ym"] = shortint.datadate.dt.to_period("M")
shortint = shortint.sort_values(["datadate", "gvkey", "iid"]).set_index("datadate")
shortint = shortint.groupby(["gvkey","iid","ym"])[["shortint", "shortintadj"]].mean() #agg("first")
shortint = shortint.reset_index()

ccm = pd.read_csv(DATA_RAW / "ccm_link.csv")
ccm.DATE = pd.to_datetime(ccm.DATE, format = "%Y%m%d")
ccm["ym"] = ccm.DATE.dt.to_period("M")
ccm.iid = ccm.iid.apply(lambda x:int( x.replace("X","")))

rsi = shortint.merge(ccm, on = ["gvkey","iid","ym"])
rsi = rsi.rename(columns = {"PERMNO": "permno"})
rsi = rsi[["permno","ym","shortint","shortintadj"]]

# set sample date range
begdate = '03/01/1972'
enddate = '12/31/2022'

# sql similar to crspmerge macro
conn=wrds.Connection()
crsp_m = conn.raw_sql(f"""
                      select a.permno, a.date, 
                      a.shrout, a.cfacshr
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '{begdate}' and '{enddate}'
                      """, date_cols=['date']) 
conn.close()

# change variable format to int
crsp_m[['permno']]= crsp_m[['permno']].astype(int)
crsp_m['tso']=crsp_m['shrout']*crsp_m['cfacshr']*1e3 # total shares out adjusted
crsp_m["ym"] = crsp_m.date.dt.to_period("M")

rsi = rsi.merge(crsp_m, on = ["ym","permno"])
rsi["rsi"] = rsi.shortintadj / rsi.tso
rsi = rsi.rename(columns = {"ym": "period"})
rsi = rsi[["period","permno","rsi"]]
rsi.period = rsi.period.dt.to_timestamp("M")
rsi = rsi[~np.isinf(rsi.rsi)]
rsi.to_parquet(DATA_RAW / "short_interest.pq")
