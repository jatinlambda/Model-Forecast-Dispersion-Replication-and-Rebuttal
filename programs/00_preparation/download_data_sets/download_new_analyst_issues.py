
"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Follows code by Qingyi Freda Song Drechsler to compute new analyst issues
"""
import pandas as pd
import numpy as np
import wrds
import pickle as pkl
from dateutil.relativedelta import *
import datetime

from joblib import Parallel, delayed
from joblib.parallel import parallel_backend

from pathlib import Path

def get_project_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[3]
    else:
        # interactive / notebook / IDE execution
        return Path.cwd().resolve().parents[2]

ROOT = get_project_root()

DATA_RAW = ROOT / "data" / "raw"

###################
# Connect to WRDS #
###################
conn=wrds.Connection()

# set sample date range
begdate = '01/01/1980'
enddate = '12/31/2022'

# set CRSP date range a bit wider to guarantee collecting all information
crsp_begdate = '01/01/1970'
crsp_enddate = '12/31/2023'

with open(DATA_RAW / 'iclink.pkl', 'rb') as f:
    iclink = pkl.load(f)
    
    
_ccm = conn.raw_sql(""" select gvkey, lpermco as permco, lpermno as permno, 
                        linkdt, linkenddt 
                        from crsp.ccmxpf_linktable 
                        where usedflag=1 
                        and linkprim in ('P', 'C')""", date_cols=['linkdt', 'linkenddt'])

_ccm[['permco', 'permno']] = _ccm[['permco', 'permno']].astype(int)

# Fill linkenddt missing value (.E in SAS dataset) with today's date
_ccm['linkenddt'] = _ccm.linkenddt.fillna(datetime.date.today())

_sec = conn.raw_sql(""" select ibtic, gvkey from comp.security """)

gvkey = pd.merge(_ccm, _sec.loc[_sec.ibtic.notna()], how='left', on=['gvkey'])


iclink_hq = iclink.loc[(iclink.score <=1)]

gvkey = pd.merge(gvkey, iclink_hq, how='left', on=['permno'])

# fill missing ticker with ibtic
gvkey.ticker = np.where(gvkey.ticker.notnull(), gvkey.ticker, gvkey.ibtic)

# Keep relevant columns and drop duplicates if there is any
gvkey = gvkey[['gvkey', 'permco', 'permno', 'linkdt', 'linkenddt','ticker']]

gvkey = gvkey.drop_duplicates()

# date ranges from gvkey
# min linkdt for ticker and permno combination
gvkey_mindt = gvkey.groupby(['ticker','permno']).linkdt.min().reset_index()

# max linkenddt for ticker and permno combination
gvkey.linkenddt = pd.to_datetime(gvkey.linkenddt)
gvkey_maxdt = gvkey.groupby(['ticker','permno']).linkenddt.max().reset_index()

# link date range 
gvkey_dt = pd.merge(gvkey_mindt, gvkey_maxdt, how='inner', on=['ticker','permno'])

ibes_temp = conn.raw_sql(f"""
                        select *
                        from ibes.detu_epsus 
                        where fpedats between '{begdate}' and '{enddate}'
                        and (fpi='1')
                        """, date_cols = ['revdats', 'anndats', 'fpedats'])

# merge to get date range linkdt and linkenddt to fulfill date requirement
ibes_temp = pd.merge(ibes_temp, gvkey_dt, how='left', on=['ticker'])
ibes_temp = ibes_temp.loc[(ibes_temp.linkdt<=ibes_temp.anndats) & (ibes_temp.anndats <= ibes_temp.linkenddt)]
ibes_temp["ym"] = pd.to_datetime(ibes_temp.actdats).dt.to_period("M")

# match again with crsp monthly ncusip to avoid spurious matches as in permno == 51633
crsp_map = conn.raw_sql(
    """
                            select distinct     a.permno,
                                                a.ncusip,
                                                a.cusip,
                                                a.tsymbol,
                                                a.ticker,
                                                a.shrcd,
                                                a.namedt,       
                                                a.nameendt
                            from
                                crsp.msenames as a
                            where
                                    a.nameendt >= '1970-01-01'
                                AND
                                    a.namedt <= a.nameendt
                            """
)

crsp_map["namedt"] = pd.to_datetime(crsp_map["namedt"], format="%Y-%m-%d")
crsp_map["nameendt"] = pd.to_datetime(crsp_map["nameendt"], format="%Y-%m-%d")
crsp_map = crsp_map[["permno", "ncusip", "nameendt"]]
dtype = {"permno": "int64"}
crsp_map = crsp_map.astype(dtype)
ibes_temp = ibes_temp.astype(dtype)
crsp_map = crsp_map.sort_values("nameendt")
ibes_temp["actdats"] = pd.to_datetime(ibes_temp["actdats"], format="%Y-%m-%d")
ibes_temp = ibes_temp.sort_values("actdats")

ibes_temp = pd.merge_asof(
    ibes_temp, crsp_map, by="permno", left_on=["actdats"], right_on=["nameendt"], direction="forward"
)

ibes_temp = ibes_temp[ibes_temp.cusip == ibes_temp.ncusip]

# at least 3 analysts
ibes_temp = ibes_temp[ibes_temp.actdats <= ibes_temp.fpedats]
ibes_temp = ibes_temp.groupby(["permno", "fpedats"]).filter(lambda x: x.shape[0] >= 3)

def num_est(grp):
    grp["fpedats_ym"] = pd.to_datetime(grp.fpedats).dt.to_period("M")

    s = grp.ym.min()
    e = grp.fpedats_ym.max()
        
    running_ct = 0
    
    issued_broker = []
    
    res = []
    while s <= e:
        tmp_ym = grp[grp.ym == s]
        
        if tmp_ym.empty:
            issued_ct = 0
            
        else:
                
            # get list of unique brokers in month
            broker_m = list(tmp_ym.estimator.unique())
            
            # select new unique brokers
            broker_new = [x for x in broker_m if not(x in issued_broker)]
            
            issued_ct = len(broker_m)
    
            running_ct = running_ct + len(broker_new)
            
            if len(broker_new) > 0:
                issued_broker.extend(broker_new)
        
        res.append(pd.DataFrame({
            "ym": s,
            "issued": issued_ct,
            "running_issued": running_ct,
            "permno": grp.permno.iloc[0],
            "fpedats_ym": e}, index = [0]))
        s += 1
    res = pd.concat(res)
    return res

with parallel_backend("loky", n_jobs=20):
    numest = Parallel(verbose=True)(
        delayed(num_est)(grp)
        for _, grp in ibes_temp.groupby(["permno","fpedats"])
    )
numest = pd.concat(numest)

numest["ratio_analyst_issue"] =  numest.issued / numest.running_issued
numest["date"] = numest.ym.dt.to_timestamp("M")
numest = numest.drop(columns = ["ym"])
numest = numest.sort_values(["permno","date","fpedats_ym"])
numest = numest.groupby(["permno","date"]).last()

numest.drop(columns = "fpedats_ym").to_parquet(DATA_RAW / "numest_ratio.pq")