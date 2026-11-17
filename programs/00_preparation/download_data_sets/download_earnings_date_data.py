
"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Construct data set for earnings-based MFD
"""

import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from joblib.parallel import parallel_backend
import os
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
DATA_DERIVED = ROOT / "data" / "derived"

# %% download necessary data

ibes_link = pd.read_pickle(DATA_RAW / 'iclink.pkl')
# just merge if perfect historical cusip match
ibes_link = ibes_link[ibes_link.score == 0]

conn = wrds.Connection()

ibes_act = conn.raw_sql(f"""
                        select * from ibes.actu_epsus 
                        where pdicity='QTR'
                        """, date_cols = ['repdats', 'fpedats'])
                        
ibes_act = ibes_act.dropna(subset = "value")

start_date = "'1969-01-01'"
crsp_prices = conn.raw_sql(
    f"""
        select distinct     permno,
                            date,
                            prc,
                            cfacshr, 
                            shrout
        from
                crsp.dsf
        where
                        date >= {start_date}
    """
)

# ---- specify format
names = {"prc": "close"}
crsp_prices.date = pd.to_datetime(crsp_prices.date, format="%Y-%m-%d")
dtype = {
    "permno": "int64",
}
crsp_prices = crsp_prices.astype(dtype)
crsp_prices = crsp_prices.rename(columns=names)
crsp_prices.close = crsp_prices.close.abs()

# %% expand ibes data
ibes_act["pends"] = pd.to_datetime(ibes_act.pends)
ibes_act["actdats"] = pd.to_datetime(ibes_act.actdats)

ibes_act["pends_ym"] = ibes_act.pends.dt.to_period("M")
ibes_act["next_quarter_end"] = ibes_act.pends_ym - 3
ibes_act["next_quarter_start"] = ibes_act.pends_ym - 5
ibes_act = ibes_act.merge(ibes_link[["ticker","permno"]], on = "ticker")

ibes_permnos = list(ibes_act.permno.unique())

def expander(row):
    tmp = pd.DataFrame({"period": pd.period_range(row.next_quarter_start, row.next_quarter_end)})
    tmp["permno"] = row["permno"]
    tmp["ibes_ticker"] = row["ticker"]
    tmp["eps_next_q"] = row["value"]
    tmp["actdats"] = row["actdats"]
    
    return tmp

with parallel_backend("loky", n_jobs=5):
    ibes_data = Parallel(verbose=True)(
        delayed(expander)(row)
        for _, row in ibes_act.iterrows()
    )
ibes_data = pd.concat(ibes_data)

# merge cfacshr 
ibes_adjusted = ibes_data.copy()
ibes_adjusted["period"] = ibes_adjusted["period"].dt.to_timestamp("M")
ibes_adjusted = ibes_adjusted.rename(columns = {"period": "date"})
ibes_adjusted = ibes_adjusted.sort_values(["date"])

crsp_prices = crsp_prices[crsp_prices.permno.isin(list(ibes_adjusted.permno.unique()))] 
crsp_prices = crsp_prices.sort_values(["date"])

ibes_adjusted = pd.merge_asof(ibes_adjusted, crsp_prices[["date","permno","cfacshr", "close", "shrout"]].rename(columns = {"cfacshr": "cfac_eom"}), 
                              direction = "backward",
                              allow_exact_matches = True,
                              on = "date", by = "permno")

ibes_adjusted.actdats = ibes_adjusted.actdats.fillna(ibes_adjusted.date + pd.Timedelta(days = 1))
ibes_adjusted = ibes_adjusted.sort_values(["actdats"])

ibes_adjusted = pd.merge_asof(ibes_adjusted, crsp_prices[["date","permno","cfacshr"]].rename(columns = {"cfacshr": "cfac_ann", "date": "actdats"}), 
                              direction = "backward",
                              allow_exact_matches = True,
                              on = "actdats", by = "permno")

ibes_adjusted.cfac_eom = ibes_adjusted.cfac_eom.fillna(1)
ibes_adjusted.cfac_ann = ibes_adjusted.cfac_ann.fillna(1)
ibes_adjusted["eps_next_q_adj"] = ibes_adjusted.eps_next_q * ibes_adjusted.cfac_eom / ibes_adjusted.cfac_ann
ibes_adjusted["eps_next_q_adj_rel"] = ibes_adjusted.eps_next_q_adj / ibes_adjusted.close 


# quarter forecasts start in 1984
ibes_adjusted = ibes_adjusted[ibes_adjusted.date >= "1984-01-01"]

# %% get JKP data
dtypes = {'iid': 'str'}

jkp = pd.read_excel(DATA_RAW / 'Factor Details.xlsx')['abr_jkp'].dropna().tolist()
usa_data = pd.read_csv(DATA_RAW / 'usa.csv',dtype=dtypes, usecols = ["crsp_shrcd", "crsp_exchcd", "date", "eom", "permno", "sic", "prc", "assets", "source_crsp"] + jkp)

# Clean data

# Keep only common stocks listed in the three prime exchanges
usa_data = usa_data[(usa_data.crsp_shrcd == 10) | (usa_data.crsp_shrcd == 11)]
usa_data = usa_data[(usa_data.crsp_exchcd == 1) | (usa_data.crsp_exchcd == 2) | (usa_data.crsp_exchcd == 3)]

# Remove utility and financial stocks
usa_data.sic = usa_data.sic.astype(str)
usa_data['SIC1'] = usa_data.sic.apply(lambda x:x[0])
usa_data['SIC2'] = usa_data.sic.apply(lambda x:x[:2])
usa_data['SIC4'] = usa_data.sic.apply(lambda x:x[:4])
usa_data = usa_data[(usa_data.SIC1 != '6') & (usa_data.SIC2 != '49')]

# Remove stocks with price not higher than 5
usa_data = usa_data[usa_data.prc > 5]

# Drop observations before July 1966
usa_data = usa_data[usa_data.date>=19660731]

# Sort by date and PERMNO 
usa_data.eom = pd.to_datetime(usa_data.eom, format = "%Y%m%d")
usa_data = usa_data.merge(ibes_adjusted[["date","permno","eps_next_q_adj", "eps_next_q_adj_rel", "shrout"]].rename(columns = {"date": "eom"}), on = ["permno", "eom"])
usa_data["eps_next_q_adj_assets"] = usa_data["eps_next_q_adj"] / (usa_data["assets"] / (usa_data["shrout"] / 10**3) )

id_vars = ['date','permno']
usa_data.set_index(id_vars,inplace=True)
usa_data.sort_index(level=[0,1],inplace=True)

eps_next_q = usa_data[['eps_next_q_adj', 'eps_next_q_adj_rel', 'eps_next_q_adj_assets']]

JKP_char = usa_data[jkp]

def charNorm(df):
    # Fill in missing data with cross-sectional median value
    df_medfill = df.groupby(level=0,group_keys=False).apply(lambda x: x.fillna(x.median(),axis=0))

    # Rank-standardize the cross-section
    df_ranknorm = 2 * df_medfill.groupby(level='date').rank() / df_medfill.groupby(level='date').count() - 1

    # Fill remaining NaNs with 0
    df_ranknorm = df_ranknorm.fillna(0)
    
    return df_ranknorm

JKP_ranknorm = charNorm(JKP_char)

eps_next_q.to_pickle(DATA_RAW / 'eps_next_q.pkl.zip',compression='zip')
JKP_ranknorm.to_pickle(DATA_RAW / 'JKP_ranknorm_eps.pkl.zip',compression='zip')

