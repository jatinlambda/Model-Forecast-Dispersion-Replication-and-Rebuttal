"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Constructs monthly standardized unexpected volume
"""
# %% imports
import pandas as pd
from joblib import Parallel, delayed
from joblib.parallel import parallel_backend

import statsmodels.api as sm
    
def permno_grouper_monthly(grp):
    yms = grp.date.unique()
    dhandler = pd.DataFrame({"start": yms[:-11], "end": yms[:-11] + 11}).sort_values("start")
    
    res = []
    
    for _, row in dhandler.iterrows():
        s = row["start"]
        e = row["end"]
        tmp = grp[(grp.date >= s) & (grp.date <= e)].copy()
        if tmp.shape[0] >= 6:
            
            results = sm.OLS(tmp.vol_adj,tmp[["const", "ret_neg", "ret_pos"]]).fit()
            resid = results.resid
            suv = resid / resid.std()
            tmp["suv"] = suv.values
            suv_m = tmp.groupby("date").suv.mean()
            
            if e in suv_m.index:
                res.append(pd.DataFrame({
                    "ym": e, "suv": suv_m[e], "permno": grp.permno.iloc[0]}, index = [0]))

    if not(len(res) == 0):
        res = pd.concat(res)
        return res

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
        
# %% Monthly suv
crsp_m = pd.read_parquet(DATA_RAW / "crsp_monthly_long.pq")
crsp_m["date"] = crsp_m.date.dt.to_period("M")

crsp_m["ret_pos"] = crsp_m.ret
crsp_m.loc[crsp_m.ret < 0, "ret_pos"] = 0
crsp_m["ret_neg"] = crsp_m.ret
crsp_m.loc[crsp_m.ret > 0, "ret_neg"] = 0
crsp_m["ret_neg"] = crsp_m["ret_neg"].abs()
crsp_m["const"] = 1
crsp_m["vol_adj"] = crsp_m.vol * crsp_m.cfacshr
crsp_m = crsp_m.dropna(subset = ["ret", "vol_adj"], how = "any")

# sort
crsp_m = crsp_m.sort_values(["date","permno"])

# compute SUV on monthly basis according to Garfunkel (2009)
with parallel_backend("loky", n_jobs=20):
    suv_m = Parallel(verbose=True)(
        delayed(permno_grouper_monthly)(grp)
        for _, grp in crsp_m.groupby(["permno"])
    )
suv_m = pd.concat(suv_m).reset_index(drop = True)
suv_m = suv_m.dropna()
suv_m["ym"] = suv_m.ym.dt.to_timestamp("M")
suv_m = suv_m.rename(columns = {"ym": "date", "suv": "suv_m"})

suv_m.to_parquet(DATA_DERIVED / "suv.pq")


