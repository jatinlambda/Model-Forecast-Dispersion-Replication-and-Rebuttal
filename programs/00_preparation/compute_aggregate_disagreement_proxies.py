"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Constructs aggregate disagreement proxies
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

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
    "eom", "permno", "crsp_shrcd", "crsp_exchcd", "source_crsp", "shares", "tvol", "prc", "ret", "me", "ivol_capm_252d"])

crsp["eom"] = pd.to_datetime(crsp.eom, format = "%Y%m%d")
crsp = crsp[crsp.crsp_shrcd.isin([10,11,12])]
crsp = crsp[crsp.crsp_exchcd.isin([1,2,3])]
crsp = crsp[crsp.prc >= 5]
crsp = crsp[crsp.source_crsp == 1]

# %% aggregate unexpected trading volume according to Garfinkel 2006
window = 60

market_ret = crsp.groupby("eom").apply(lambda x: (x["ret"] * x["me"]).sum() / x.me.sum())
turn_over = crsp.groupby("eom").apply(lambda x: x.tvol.sum() / x.shares.sum())
turn_over = np.log(turn_over)
turn_over = pd.merge(turn_over.to_frame(name = "turnover"), market_ret.to_frame(name = "ret"), left_index = True, right_index = True)
turn_over = turn_over[1:]

market_unexplained_vol = []
for date in turn_over.index.values:
    print(f"Computing aggregate unexpected trading volume for {date}")
    tmp = turn_over[:date].copy()
    if tmp.shape[0] >= window:
        tmp = tmp[-window:]
        res = AutoReg(endog = tmp.turnover.values, lags = 4, trend = "n").fit()
        tmp = tmp[4:]
        tmp["volume"] = res.resid
        tmp["ret_plus"] = np.maximum(tmp.ret, 0)
        tmp["ret_minus"] = np.minimum(tmp.ret, 0)
        tmp["const"] = 1
        res = sm.OLS(tmp.volume, tmp[["const", "ret_plus", "ret_minus"]]).fit()
        
        uv = res.resid.iloc[-1]
        dsuv = uv / res.resid.std()
        market_unexplained_vol.append(pd.DataFrame({
            "suv": dsuv}, index = [date]))
market_unexplained_vol = pd.concat(market_unexplained_vol)       
        
# %% Aggregate IVOL
ivol = crsp[["eom", "me", "ivol_capm_252d"]].groupby("eom").apply(lambda x: (x.ivol_capm_252d * x.me).sum() / x.me.sum())
ivol = ivol.to_frame(name = "agg_ivol")

# %% merge and save
aggregate_measures = pd.merge(ivol, market_unexplained_vol, left_index = True, right_index = True)
aggregate_measures.to_parquet(DATA_DERIVED / "aggregate_measures.pq")