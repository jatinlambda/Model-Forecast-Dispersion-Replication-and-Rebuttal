"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Downloads daily and monthly stock returns from CRSP via WRDS API
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

# %% download
start_date = "'1960-01-31'"
crsp_database_type = "d"
db = wrds.Connection()

crsp_prices = db.raw_sql(
    f"""
        select distinct     a.permno,
                            a.date,
                            a.prc,
                            a.askhi,
                            a.bidlo,
                            a.ask,
                            a.bid,
                            a.openprc,
                            a.vol,
                            a.ret, 
                            a.cfacshr
        from
                crsp.{crsp_database_type}sf as a
        where
                        a.date >= {start_date}
    """
)

crsp_prices.date = pd.to_datetime(crsp_prices.date, format = "%Y-%m-%d")
crsp_prices.to_parquet(DATA_RAW /"crsp_daily_long.pq")


crsp_database_type = "m"
crsp_prices = db.raw_sql(
    f"""
        select distinct     a.permno,
                            a.date,
                            a.prc,
                            a.vol,
                            a.ret, 
                            a.cfacshr
        from
                crsp.{crsp_database_type}sf as a
        where
                        a.date >= {start_date}
    """
)

crsp_prices.date = pd.to_datetime(crsp_prices.date, format = "%Y-%m-%d")
crsp_prices.to_parquet(DATA_RAW / "crsp_monthly_long.pq")