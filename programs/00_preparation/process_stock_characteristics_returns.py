"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Extracts returns, market cap, and stock characteristics from 
Jensen et al. (2023) raw data
"""

import pandas as pd
import os
import numpy as np


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

# %% Load data data
dtypes = {'iid': 'str'}
dataloc = '../../data/raw/'
usa_data = pd.read_csv(DATA_RAW / 'usa.csv', dtype=dtypes)

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

usa_data.dropna(subset=['permno'])

# Sort by date and PERMNO 
id_vars = ['date','permno']
usa_data.set_index(id_vars,inplace=True)
usa_data.sort_index(level=[0,1],inplace=True)

# Get JKP factors, 1-month ahead excess returns
jkp = pd.read_excel(DATA_RAW / 'Factor Details.xlsx')['abr_jkp'].dropna().tolist()
JKP_char = usa_data[jkp]

RET = usa_data[['ret_exc_lead1m']]

MKTEQ = usa_data[['market_equity']]

def charNorm(df):
    # Fill in missing data with cross-sectional median value
    df_medfill = df.groupby(level=0,group_keys=False).apply(lambda x: x.fillna(x.median(),axis=0))

    # Rank-standardize the cross-section
    df_ranknorm = 2 * df_medfill.groupby(level='date').rank() / df_medfill.groupby(level='date').count() - 1

    # Fill remaining NaNs with 0
    df_ranknorm = df_ranknorm.fillna(0)
    
    return df_ranknorm

JKP_ranknorm = charNorm(JKP_char)

# Returns data (full)
RET.to_parquet(DATA_DERIVED / 'RET.pq')

# Market equity data (full)
MKTEQ.to_parquet(DATA_DERIVED / 'MKTEQ.pq')

# Characteristics data (153 chars)
JKP_char.to_parquet(DATA_DERIVED / 'JKP_chars.pq')
JKP_ranknorm.to_parquet(DATA_DERIVED / 'JKP_ranknorm.pq')
