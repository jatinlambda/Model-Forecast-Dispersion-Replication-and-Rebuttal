
"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Follows code by Qingyi Freda Song Drechsler to calculate IOR
"""

import pandas as pd
import numpy as np
import wrds
from joblib import Parallel, delayed


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

conn = wrds.Connection()

######################################
# Step 1                             #
# CRSP Block                         #
######################################

# set sample date range
begdate = '03/01/1980'
enddate = '12/31/2023'

# sql similar to crspmerge macro

crsp_m = conn.raw_sql(f"""
                      select a.permno, a.date, 
                      a.ret, a.vol, a.shrout, a.prc, a.cfacpr, a.cfacshr
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '{begdate}' and '{enddate}'
                      and b.shrcd between 10 and 11
                      """, date_cols=['date']) 

# change variable format to int
crsp_m[['permno']]= crsp_m[['permno']].astype(int)

# get month and quarter-end dates
crsp_m['mdate']=crsp_m['date']+pd.offsets.MonthEnd(0)
crsp_m['qdate']=crsp_m['date']+pd.offsets.QuarterEnd(0)

# calculate adjusted price, total shares and market cap
crsp_m['p']=crsp_m['prc'].abs()/crsp_m['cfacpr'] # price adjusted
crsp_m['tso']=crsp_m['shrout']*crsp_m['cfacshr']*1e3 # total shares out adjusted
crsp_m['me'] = crsp_m['p']*crsp_m['tso']/1e6 # market cap in $mil

# keep only relevant columns
crsp_m = crsp_m[['permno','mdate','qdate','date','cfacshr', 'p', 'tso','me']]

# For each stock(permno), each quarter (qdate), find the last monthly date (mdate)
qend = crsp_m[['permno','mdate','qdate']].groupby(['permno','qdate'])['mdate'].max().reset_index()

# Merge back to keep last monthly observation for each quarter
crsp_qend = pd.merge(crsp_m, qend, how='inner', on=['permno','qdate','mdate'])

######################################
# Step 2                             #
# Merge TR 13f s34type1 and s34type3 #
######################################

fst_vint = conn.raw_sql("""
                      select rdate, fdate, mgrno, mgrname
                      from tfn.s34type1 
                      """, date_cols=['rdate','fdate']) 

# Keep first vintage with holding data for each mgrno-rdate combo
min_fdate = fst_vint.groupby(['mgrno','rdate'])['fdate'].min().reset_index()

# Merge back with the fst_vint data to keep only the first vintage records
fst_vint = pd.merge(fst_vint, min_fdate, how='inner', on=['mgrno','rdate','fdate'])

# Sort by mgrno and rdate and create lag_rdate to calculate gap
fst_vint = fst_vint.sort_values(['mgrno', 'rdate'])
fst_vint['lag_rdate']=fst_vint.groupby(['mgrno'])['rdate'].shift(1)

# Number of quarters gap between rdate and lag_rdate
fst_vint['rdate_year']=fst_vint.rdate.dt.year
fst_vint['rdate_qtr'] =fst_vint.rdate.dt.quarter

fst_vint['lag_rdate_year']=fst_vint.lag_rdate.dt.year
fst_vint['lag_rdate_qtr'] =fst_vint.lag_rdate.dt.quarter

fst_vint['qtr'] = (fst_vint.rdate_year - fst_vint.lag_rdate_year)*4 + (fst_vint.rdate_qtr - fst_vint.lag_rdate_qtr)


# label first_report flag
fst_vint['first_report'] = ((fst_vint.qtr.isnull()) | (fst_vint.qtr>=2))
fst_vint = fst_vint.drop(['qtr'],axis=1)

# Last report by manager or missing 13F reports in the next quarter(s)
fst_vint = fst_vint.sort_values(['mgrno','rdate'], ascending=[True, False])
fst_vint['lead_rdate']=fst_vint.groupby(['mgrno'])['rdate'].shift(1)

# Number of quarters gap between lead_rdate and rdate
fst_vint['lead_rdate_year']=fst_vint.lead_rdate.dt.year
fst_vint['lead_rdate_qtr'] =fst_vint.lead_rdate.dt.quarter

fst_vint['qtr'] = (fst_vint.lead_rdate_year - fst_vint.rdate_year)*4 + (fst_vint.lead_rdate_qtr - fst_vint.rdate_qtr)


# label last_report flag
fst_vint['last_report'] = ((fst_vint.qtr.isnull()) | (fst_vint.qtr>=2))
fst_vint = fst_vint.drop(['qtr'],axis=1)

fst_vint = fst_vint[(fst_vint['rdate']<=enddate) & (fst_vint['rdate']>=begdate)]\
.drop(['lag_rdate','lead_rdate'], axis=1)

# Add total number of 13f filers during each quarter
fst_vint = fst_vint.sort_values(['rdate', 'mgrno'])

NumInst = fst_vint.groupby(['rdate'])['mgrno'].count().reset_index()\
.rename(columns={'mgrno':'NumInst'})

fst_vint = pd.merge(fst_vint, NumInst, how='left', on='rdate')
fst_vint = fst_vint.drop(['mgrname'],axis=1)

######################################
# Step 3                             #
# Extract Holdings and Adjust Shares #
######################################

s34type3 = conn.raw_sql("""
                      select fdate, mgrno, cusip, shares
                      from tfn.s34type3
                      """, date_cols=['fdate']) 

holdings_v1 = pd.merge(fst_vint, s34type3, how='inner', on=['fdate','mgrno'] )

# Map 13F's historical cusip to CRSP's permno information
crsp = conn.raw_sql("""
                    select distinct permno, ncusip
                    from crsp.msenames
                    where ncusip != ''
                    """)

holdings_v2 = pd.merge(holdings_v1, crsp, how='inner', left_on='cusip', right_on='ncusip')
holdings_v2 = holdings_v2.drop(['cusip','ncusip'], axis=1)

######################################
# Step 4                             #
# Adjust Shares Using CRSP CFACSHR   #
# Align at Vintage Dates             #
######################################

holdings = pd.merge(holdings_v2, crsp_qend[['qdate','permno','cfacshr']], \
                    how='inner', left_on=['permno','fdate'], right_on=['permno','qdate'])

# Calculate Adjusted Shares
holdings['shares_adj']=holdings['shares']*holdings['cfacshr']
holdings=holdings.drop(['shares','qdate','cfacshr','fdate'], axis=1)

# Sanity Checks for Duplicates - Ultimately, Should be 0 Duplicates
holdings = holdings.drop_duplicates(subset=['permno','rdate','mgrno'])
crsp_qend = crsp_qend.drop_duplicates(subset=['permno','qdate'])

######################################
# Step 5                             #
# Calculate Institutional Measures   #
# At Security Level                  #
######################################

holdings = holdings[holdings['shares_adj']>0]

# Number of Owners
io_numowners = holdings.groupby(['permno','rdate'])['shares_adj'].count().reset_index()\
.rename(columns={'shares_adj':'numowners'})

# Number of Institutions
io_numinst = holdings.groupby(['permno','rdate'])['NumInst'].max().reset_index()\
.rename(columns={'NumInst':'numinst'})

# New Old Institutions and Total Shares held by institutions
io_total = holdings.groupby(['permno','rdate'])['first_report','last_report','shares_adj']\
.sum().reset_index()\
.rename(columns={'first_report':'newinst', 'last_report':'oldinst','shares_adj':'io_total'})
io_total.head()

# USS
tmp = holdings[['permno','rdate','shares_adj']]
tmp['shares_adj2'] = tmp['shares_adj']**2
io_uss = tmp.groupby(['permno','rdate'])['shares_adj2'].sum().reset_index()\
.rename(columns={'shares_adj2':'io_ss'})

# combine various metrics together
io_metrics = pd.merge(io_numowners, io_numinst, how='inner', on=['permno','rdate'])
io_metrics = pd.merge(io_metrics, io_total, how='inner', on =['permno','rdate'])
io_metrics = pd.merge(io_metrics, io_uss, how='inner', on = ['permno','rdate'])

# Calculate IO DBreadth and Concentration Metrics         

io_metrics['ioc_hhi'] = io_metrics['io_ss']/(io_metrics['io_total']**2)
io_metrics['d_owner'] = io_metrics['numowners'] - io_metrics['oldinst']
io_metrics = io_metrics.sort_values(['permno','rdate'])

# Create lag_numinst and lag_d_owner for breadth calculation
io_metrics['lag_numinst'] = io_metrics.groupby(['permno'])['numinst'].shift(1)
io_metrics['lag_d_owner'] = io_metrics.groupby(['permno'])['d_owner'].shift(1)

# Calculate change in breadth (dbreath)
io_metrics['dbreadth'] = ((io_metrics['numowners'] - io_metrics['newinst']) - io_metrics['lag_d_owner'])\
/io_metrics['lag_numinst']

# Keep only relevant columns
io_metrics = io_metrics[['permno','rdate','numowners','io_total','ioc_hhi','dbreadth']]

######################################
# Step 6                             #
# Add CRSP Market Data to Holdings   #
# At Calendar Quarter End            #
######################################

# Note: a right join is necessary to indentify common stocks with no 13F data
io_ts = pd.merge(io_metrics, crsp_qend[['permno','qdate','p','tso','me']], \
                 how='right', left_on=['permno','rdate'], right_on=['permno','qdate'])
io_ts = io_ts.sort_values(['permno','rdate'])

# keep only records with tso>0
io_ts = io_ts[io_ts['tso']>0]
io_ts['ior'] = (io_ts['io_total']/io_ts['tso']).fillna(0)
io_ts['io_missing'] = np.where(io_ts['rdate'].isna(), 1, 0 )
io_ts['io_g1'] = np.where(io_ts['ior']>1, 1, 0)

# fill in missing rdate with valid qdate 
io_ts['rdate'] = np.where(io_ts['rdate'].isna(), io_ts['qdate'], io_ts['rdate'])

# Sanity check for duplicates
io_ts = io_ts.drop_duplicates(subset=['rdate', 'permno']).sort_values(['rdate','permno'])


io_ts = io_ts.sort_values(["permno","qdate"])
io_ts = io_ts.set_index("qdate")

ior = Parallel(n_jobs=20, verbose=True, backend = "threading")(
    delayed(lambda x: x.resample("M").asfreq().fillna(method = "ffill"))(grp)
    for _, grp in io_ts.groupby("permno")
)
ior = pd.concat(ior)
ior = ior.reset_index().rename(columns = {"qdate":"date"})
ior.to_parquet(DATA_RAW / "ior_dbreadth_hhi.pq")