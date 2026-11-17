"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Prepares test portfolios for 3-pass FMB of Giglio and Xiu (2021)
"""

from pathlib import Path

def get_project_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2]
    else:
        # interactive / notebook / IDE execution
        return Path.cwd().resolve().parents[1]

ROOT = get_project_root()

import pandas as pd
import os

mfd = pd.read_parquet(os.path.join(ROOT,"data/mfd/main_data.pq"))

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)


from helper_prepare import get_norm_chars, mapper

from pandas.tseries.offsets import MonthEnd
import numpy as np

flag_winsorize = True
flag_standardize = False
    
#%% Functions

# winsorize
def wins(x, bds):
    q = x.quantile(bds)
    x.loc[x<=q.iloc[0]] = q.iloc[0]
    x.loc[x>=q.iloc[1]] = q.iloc[1]
    return x

# function to calculate value weighted return
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan
    
# save output 
def save_csv(df, file_path, file_name):
    df.to_csv(os.path.join(file_path, file_name + '.csv'), index=True, header=True)

save_path = os.path.join(ROOT,"data/derived/gx_inputs")

# Function to assign quantiles
def assign_quantiles(df, col, n_quantiles):
    return df.groupby('date', group_keys=False)[col].apply(
        lambda x: pd.qcut(x.rank(method='first'), q=n_quantiles, labels=False, duplicates='drop') + 1
    )


# %%% Read in and clean raw data

# Get JKP data
norm_chars = get_norm_chars("unnorm")

# Merge characteristic with MFD data
mfd = mfd.reset_index()[["date","permno","mfd","ret_exc_lead1m"]]
mfd = pd.merge(mfd, norm_chars, on = ["permno","date"])
mfd['date']=mfd['date']+MonthEnd(0) # Line up date to be end of month

# contemp. return
add_data = pd.read_csv(os.path.join(ROOT,"data/raw/usa.csv"), usecols = ["permno","eom","ret_exc"])

add_data = add_data.rename(columns = {"eom":"date"})
add_data.date = pd.to_datetime(add_data.date, format = "%Y%m%d")
add_data['date']=add_data['date']+MonthEnd(0)
mfd = mfd.merge(add_data, on = ["date","permno"], how = "left")

# Winsorize and standardize data
mfd_win = mfd.copy()
mfd_win = mfd_win.set_index(["permno","date"])
if flag_winsorize:
    cols_to_win = [x for x in mfd_win.columns] #if not(x in ["ret_exc_lead1m", "ret_exc"])]
    for c in cols_to_win:
        print("Winsorizing {}".format(c))
        mfd_win[c] = wins(mfd_win[c], [0.01, 0.99])

if flag_standardize:
    cols_to_win = [x for x in mfd_win.columns if not(x in ["ret_exc_lead1m", "ret_exc"])]
    for c in cols_to_win:
        print("Standardizing {}".format(c))
        mfd_win[c] = mfd_win.groupby("date")[c].transform(lambda x: (x-x.mean())/x.std())


#%% Build Fama-French July–June portfolio weights

# keep December market cap
mfd_win = mfd_win.reset_index()
mfd_win['date'] = pd.to_datetime(mfd_win['date'])
mfd_win['year']=mfd_win['date'].dt.year
mfd_win['month']=mfd_win['date'].dt.month
decme=mfd_win[mfd_win['month']==12]
decme=decme[['permno','date','market_equity','year']].rename(columns={'market_equity':'dec_me'})

### July to June dates
mfd_win['ffdate']=mfd_win['date']+MonthEnd(-6)
mfd_win['ffyear']=mfd_win['ffdate'].dt.year
mfd_win['ffmonth']=mfd_win['ffdate'].dt.month
mfd_win['1+retx']=1+mfd_win['ret_exc']
mfd_win=mfd_win.sort_values(by=['permno','date'])

# cumret by stock
mfd_win['cumretx']=mfd_win.groupby(['permno','ffyear'])['1+retx'].cumprod()

# lag cumret
mfd_win['lcumretx']=mfd_win.groupby(['permno'])['cumretx'].shift(1)

# lag market cap
mfd_win['lme']=mfd_win.groupby(['permno'])['market_equity'].shift(1)

# if first permno then use me/(1+retx) to replace the missing value
mfd_win['count']=mfd_win.groupby(['permno']).cumcount()
mfd_win['lme']=np.where(mfd_win['count']==0, mfd_win['market_equity']/mfd_win['1+retx'], mfd_win['lme'])

# baseline me
mebase=mfd_win[mfd_win['ffmonth']==1][['permno','ffyear', 'lme']].rename(columns={'lme':'mebase'})

# merge result back together
mfd_win1=pd.merge(mfd_win, mebase, how='left', on=['permno','ffyear'])
mfd_win1['wt']=np.where(mfd_win1['ffmonth']==1, mfd_win1['lme'], mfd_win1['mebase']*mfd_win1['lcumretx'])

mfd_win1 = mfd_win1.sort_values(by=['permno','date'])

decme['year']=decme['year']+1
decme=decme[['permno','year','dec_me']]

mfd_win1 = pd.merge(mfd_win1, decme, how='inner', on=['permno','year'])
mfd_win1 = mfd_win1.sort_values(by=['permno','date']).drop_duplicates()

    
#%% Assign deciles

all_controls = mfd.drop(columns=['date', 'permno']).columns

# Assign deciles (10 bins)
for col in all_controls:
    mfd_win1[f"{col}_decile"] = assign_quantiles(mfd_win1, col, 10).reset_index(drop=True)

#%% Generate weighted returns for each decile

# store portfolio assignment as of June
portfolio_cols = [f'{col}_decile' for col in all_controls]
mfd_win2 = mfd_win1[['permno', 'date'] + portfolio_cols]

mfd_win3 = pd.merge(
    mfd_win2,
    mfd_win1[['permno', 'date','ret_exc_lead1m','wt']],
    how='left', on=['permno', 'date']
)

# Calculate value-weighted and equal-weighted returns for each control variable
portfolio_returns_vw = {}  # Value-weighted portfolio returns
portfolio_returns_ew = {}  # Equal-weighted portfolio returns

# Process each control variable
for var in all_controls:
    port_name = f'{var}_decile'

    # Value-weighted returns (same as before)
    var_ret_vw = mfd_win3.groupby(['date', port_name]).apply(wavg, 'ret_exc_lead1m', 'wt').to_frame().reset_index().rename(columns={0: 'vwret'})

    # Equal-weighted returns (simple mean)
    var_ret_ew = mfd_win3.groupby(['date', port_name])['ret_exc_lead1m'].mean().reset_index().rename(columns={'ret_exc_lead1m': 'ewret'})

    portfolio_returns_vw[var] = var_ret_vw
    portfolio_returns_ew[var] = var_ret_ew

#%% Control lists 

# Sort out GX control variables
# Controls from FMB code
controls_list = (["MFD", "BETA", "SIZE", "BM", "MOM"],
            ["MFD", "BETA", "SIZE", "BM", "AG", "OP", "MOM"],
            ["MFD", 'SUE', 'AG', 'MOM', 'ILLIQ', 'OP', 'IVOL', 'BETA',
              'SIZE', 'BM', 'MAX', 'TURN', 'STR'],
            )
controls_mapped = [[mapper[control] for control in control_set] for control_set in controls_list]
gx_controls = list(set(item for sublist in controls_mapped for item in sublist))
gx_keep = gx_controls.copy()

# Don't include MFD or returns columns when generating test portfolios
remaining_factors = [
    var for var in all_controls
    if var not in {'ret_exc_lead1m', 'ret_exc', 'mfd'}
]

#%% Create control factors

# Store factor data
factor_data_vw = {}  # Value-weighted factors
factor_data_ew = {}  # Equal-weighted factors

# Process each control variable
for var in gx_keep:
    # Pivot VW returns
    var_factors_vw = portfolio_returns_vw[var].pivot(index='date', columns=f'{var}_decile', values='vwret').reset_index()
    var_factors_vw.columns.name = None  # Remove column name attribute
    var_factors_vw = var_factors_vw.reset_index()

    # Pivot EW returns
    var_factors_ew = portfolio_returns_ew[var].pivot(index='date', columns=f'{var}_decile', values='ewret').reset_index()
    var_factors_ew.columns.name = None
    var_factors_ew = var_factors_ew.reset_index()

    # Determine if deciles (D1-D10) exist
    available_columns = var_factors_vw.columns
    high_bucket = 10 
    low_bucket = 1

    # Compute High - Low spreads for VW and EW
    var_factors_vw[f'W{var}_HL'] = var_factors_vw[high_bucket] - var_factors_vw[low_bucket]
    var_factors_ew[f'E{var}_HL'] = var_factors_ew[high_bucket] - var_factors_ew[low_bucket]

    factor_data_vw[var] = var_factors_vw
    factor_data_ew[var] = var_factors_ew
    

factor_data_vw = {key: df.dropna() for key, df in factor_data_vw.items()}
factor_data_ew = {key: df.dropna() for key, df in factor_data_ew.items()}

# Outputs
# Function to combine factors for a given list of variables and factor dataset
def compute_combined_factors(var_list, factor_dataset, weight_type):
    combined = factor_dataset[var_list[0]][['date']].copy()  # Initialize with the date column
    
    for var in var_list:
        # Determine if it's quintiles (Q1-Q5) or deciles (D1-D10)
        available_columns = factor_dataset[var].columns
        high_bucket = f'W{var}_HL' if weight_type == 'VW' else f'E{var}_HL'
        
        if high_bucket in available_columns:
            combined[high_bucket] = factor_dataset[var][high_bucket]
        else:
            print(f"Warning: {high_bucket} not found for {var}, skipping...")

    combined = combined.set_index('date') 
    combined = combined.fillna(0)  

    return combined

gx_control_factors_vw = compute_combined_factors(gx_keep, factor_data_vw, 'VW') 
gx_control_factors_ew = compute_combined_factors(gx_keep, factor_data_ew, 'EW')

# Drop EW rows before VW starts
vw_start_date = gx_control_factors_vw.index.min()
gx_control_factors_ew = gx_control_factors_ew.loc[
    gx_control_factors_ew.index >= vw_start_date
]

# Save
save_csv(gx_control_factors_vw, os.path.join(save_path, 'controls'), 'vw_gx_controls')
save_csv(gx_control_factors_ew, os.path.join(save_path, 'controls'), 'ew_gx_controls')


#%% Create test asset portfolios

# Store portfolio data
portfolio_data_vw = {}  # Value-weighted portfolios
portfolio_data_ew = {}  # Equal-weighted portfolios

# Process each control variable
for var in all_controls:
    # Pivot VW returns
    var_portfolios_vw = portfolio_returns_vw[var].pivot(index='date', columns=f'{var}_decile', values='vwret').reset_index()
    var_portfolios_vw.columns.name = None  # Remove column name attribute
    var_portfolios_vw = var_portfolios_vw.reset_index()

    # Pivot EW returns
    var_portfolios_ew = portfolio_returns_ew[var].pivot(index='date', columns=f'{var}_decile', values='ewret').reset_index()
    var_portfolios_ew.columns.name = None
    var_portfolios_ew = var_portfolios_ew.reset_index()

    portfolio_data_vw[var] = var_portfolios_vw
    portfolio_data_ew[var] = var_portfolios_ew

portfolio_data_vw = {key: df.dropna() for key, df in portfolio_data_vw.items()}
portfolio_data_vw = {key: df.dropna() for key, df in portfolio_data_ew.items()}

# Function to combine portfolios for a given list of variables and portfolio dataset
def compute_combined_portfolios(var_list, weight_type):
    if weight_type == 'VW':
        dic = portfolio_data_vw.copy()
    elif weight_type == 'EW':
        dic = portfolio_data_ew.copy()
    else:
        print('Not an acceptable weight')
    combined = dic[var_list[0]][['date']].copy()  # Initialize with the date column
    for var in var_list:
        df = dic[var]
        for portfolio in list(range(1,11)):
            if portfolio in df.columns:
                col_name = f'W{var}_{portfolio}' if weight_type == 'VW' else f'E{var}_{portfolio}'
                combined[col_name] = df[portfolio]
            else:
                print(f"Warning: {portfolio} not found in {var}, skipping...")
    combined = combined.set_index('date') 
    combined = combined.dropna(axis=1)
    combined = combined[combined.index.isin(gx_control_factors_vw.index)]
    return combined

# Compute for gx_keep and remaining factors on VW and EW datasets
gx_test_portfolios_vw = compute_combined_portfolios(remaining_factors, 'VW')
gx_test_portfolios_ew = compute_combined_portfolios(remaining_factors, 'EW')

save_csv(gx_test_portfolios_vw, os.path.join(save_path,'jkp'), 'vw_jkp_test_port')
save_csv(gx_test_portfolios_ew, os.path.join(save_path,'jkp'), 'ew_jkp_test_port')


#%% Format Fama French test portfolios

# Define input and output directories
input_dir = os.path.join(ROOT,"data/external/fama_french_raw")
ffpath = os.path.join(ROOT,"data/derived/gx_inputs/ff_temp")

# List all CSV files in the input directory
csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]

# Process each file
for file in csv_files:
    file_path = os.path.join(input_dir, file)
    print(f"Processing: {file}")

    # Read the file into a list to locate data positions
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Find the start of each section
    vw_start = next(i for i, line in enumerate(lines) if "Average Value Weighted Returns -- Monthly" in line) + 1
    ew_start = next(i for i, line in enumerate(lines) if "Average Equal Weighted Returns -- Monthly" in line) + 1

    # Find the end positions (first blank line after each section)
    vw_end = next((i for i in range(vw_start, len(lines)) if lines[i].strip() == ""), len(lines))
    ew_end = next((i for i in range(ew_start, len(lines)) if lines[i].strip() == ""), len(lines))

    # Read VW data
    vw_data = pd.read_csv(file_path, skiprows=vw_start, nrows=vw_end - vw_start)
    vw_data.rename(columns={vw_data.columns[0]: "date"}, inplace=True)

    # Read EW data
    ew_data = pd.read_csv(file_path, skiprows=ew_start, nrows=ew_end - ew_start)
    ew_data.rename(columns={ew_data.columns[0]: "date"}, inplace=True)

    #  Remove last row if it's not numeric (likely a section header)
    if not vw_data.iloc[-1, 1:].apply(pd.to_numeric, errors="coerce").notna().all():
        vw_data = vw_data.iloc[:-1]
    
    if not ew_data.iloc[-1, 1:].apply(pd.to_numeric, errors="coerce").notna().all():
        ew_data = ew_data.iloc[:-1]

    # Define output file paths
    vw_output_file = os.path.join(ffpath, f"vw_{file}")
    ew_output_file = os.path.join(ffpath, f"ew_{file}")

    # Save processed data
    vw_data.to_csv(vw_output_file, index=False)
    ew_data.to_csv(ew_output_file, index=False)

    print(f"Saved VW to {vw_output_file}, EW to {ew_output_file}")

print("Processing complete.")


clean_csv = [f for f in os.listdir(ffpath) if f.lower().endswith('.csv')]

# Dictionary to store results
processed_portfolios = {}

# Process each file
for file in clean_csv:
    print(f"Processing {file}...")
    
    # Read the CSV
    df = pd.read_csv(os.path.join(ffpath, file))
    
    # Convert dates
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m')
    df['date'] = df['date'] + pd.offsets.MonthEnd(0)

    # Filter to match factor dates
    df = df[df['date'].isin(gx_control_factors_vw.index)]
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    df = df.set_index('date')
    
    # Save
    file_name = os.path.splitext(file)[0].lower()
    save_csv(df, os.path.join(save_path,'ff'), file_name)

