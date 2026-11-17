"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Constructs option-based disagreement
"""

import pandas as pd
import glob

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

def reader(file):
    data = pd.read_csv(file)
    data.quote_date = pd.to_datetime(data.quote_date, format = "%Y-%m-%d")
    
    if "expiration_date" in data.columns:
        
        open_buy_cols = ["firm_open_buy_vol", "bd_open_buy_vol", "cust_lt_100_open_buy_vol",
                    "cust_100_199_open_buy_vol", "cust_gt_199_open_buy_vol",
                    "procust_lt_100_open_buy_vol", "procust_100_199_open_buy_vol",
                    "procust_gt_199_open_buy_vol",
                    ]
        open_sell_cols = [
            "firm_open_sell_vol", "bd_open_sell_vol", "cust_lt_100_open_sell_vol",
                        "cust_100_199_open_sell_vol", "cust_gt_199_open_sell_vol",
                        "procust_lt_100_open_sell_vol", "procust_100_199_open_sell_vol",
                        "procust_gt_199_open_sell_vol"]
        
        close_buy_cols = ["firm_close_buy_vol", "bd_close_buy_vol", "cust_lt_100_close_buy_vol",
                    "cust_100_199_close_buy_vol", "cust_gt_199_close_buy_vol",
                    "procust_lt_100_close_buy_vol", "procust_100_199_close_buy_vol",
                    "procust_gt_199_close_buy_vol",
                    ]
        close_sell_cols = [
            "firm_close_sell_vol", "bd_close_sell_vol", "cust_lt_100_close_sell_vol",
                        "cust_100_199_close_sell_vol", "cust_gt_199_close_sell_vol",
                        "procust_lt_100_close_sell_vol", "procust_100_199_close_sell_vol",
                        "procust_gt_199_close_sell_vol"]
        
        data.expiration_date = pd.to_datetime(data.expiration_date, format = "%Y-%m-%d")
        data["ttm"] = (data.expiration_date - data.quote_date) / pd.Timedelta(days = 1)    
    else:
        open_buy_cols = ["firm_open_buys", "cust_open_buys_small", "cust_open_buys_medium",
                    "cust_open_buys_large"]
        
        open_sell_cols = ["firm_open_sells", "cust_open_sells_small", "cust_open_sells_medium",
                    "cust_open_sells_large"]
        
        close_buy_cols = ["firm_close_buys", "cust_close_buys_small", "cust_close_buys_medium",
                    "cust_close_buys_large"]

        close_sell_cols = ["firm_close_sells", "cust_close_sells_small", "cust_close_sells_medium",
                    "cust_close_sells_large"]
        
        data = data.rename(columns = {"type": "call_put_flag"})
                
        data.expiry = pd.to_datetime(data.expiry, format = "%Y-%m-%d")
        data["ttm"] = (data.expiry - data.quote_date) / pd.Timedelta(days = 1)        
        
    
    data["open_buy_vol"] = data[open_buy_cols].sum(axis = 1) 
    data["open_sell_vol"] = data[open_sell_cols].sum(axis = 1) 
    data["close_buy_vol"] = data[close_buy_cols].sum(axis = 1) 
    data["close_sell_vol"] = data[close_sell_cols].sum(axis = 1) 
    
    data = data[(data.ttm >= 10) & (data.ttm <= 180)]
    
    data = data.groupby(["quote_date", "call_put_flag"]).apply(lambda x: x[["open_buy_vol", "open_sell_vol", "close_buy_vol", "close_sell_vol"]].sum())
    data = data.unstack(level = 1).fillna(0)
    data.columns = ['_'.join(x) for x in data.columns]
    
    return data

# %% settings
data = "data_location/"
files = glob.glob(data + "*.csv")

# %% construction of data
data = []
for file in files:
    print(file)
    data.append(reader(file))
data = pd.concat(data)

data["pos"] = data[["open_buy_vol_C", "open_sell_vol_P", "close_buy_vol_C", "close_sell_vol_P"]].sum(axis = 1)
data["neg"] = data[["open_sell_vol_C", "open_buy_vol_P", "close_sell_vol_C", "close_buy_vol_P"]].sum(axis = 1)

data_monthly = data.copy()
data_monthly["ym"] = data_monthly.index.to_period("M")
data_monthly = data_monthly.reset_index(drop = True).set_index("ym")

data_monthly = data_monthly.groupby("ym")[["neg", "pos"]].sum()
data_monthly = data_monthly.reset_index()
data_monthly.ym = data_monthly.ym.dt.to_timestamp("M")
data_monthly = data_monthly.rename(columns = {"ym": "date"})
data_monthly.to_parquet(DATA_DERIVED / "spx_disagreement.pq")