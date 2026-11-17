"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Computes proxy for sparsity in the stock-level cross-section
"""

import pandas as pd
import numpy as np
np.random.seed(987654321)
from scipy.stats.mstats import winsorize

from sklearn.linear_model import LassoCV
    
from sklearn.model_selection import KFold
class KFoldMonths(KFold):
    def __init__(self, n_splits=5, weighted=False):
        self.n_splits = n_splits
        self.weighted = weighted

    def split(self, X, y=None, groups=None):
        if self.weighted:
            dates = X.groupby(level=0).size().sort_index()
            cum_w = dates.cumsum() / dates.sum()
            idx = np.searchsorted(cum_w, np.linspace(0, 1, self.n_splits, endpoint=False)[1:])
            dates_split = np.array_split(dates.index, idx)
        else:
            dates_split = np.array_split(np.sort(X.index.get_level_values(0).unique()), self.n_splits)
        for i in range(self.n_splits):
            val_dates = dates_split[i]
            val_index = np.argwhere(X.index.get_level_values(0).isin(val_dates)).T[0]

            test_dates = []
            for j in set(range(0, self.n_splits)) - set([i]):
                test_dates.append(dates_split[j])

            test_dates = np.concatenate(test_dates)

            test_index = np.argwhere(X.index.get_level_values(0).isin(test_dates)).T[0]
            yield test_index, val_index
            
from pathlib import Path

def get_project_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2]
    else:
        # interactive / notebook / IDE execution
        return Path.cwd().resolve().parents[1]

ROOT = get_project_root()

DATA_DERIVED = ROOT / "data" / "derived"

# %% Settings
window_length = 120

# %% Load data
norm_chars = pd.read_parquet(DATA_DERIVED / 'JKP_ranknorm.pq')
mret = pd.read_parquet(DATA_DERIVED / 'RET.pq')

norm_chars = norm_chars.reset_index()
norm_chars.date = pd.to_datetime(norm_chars.date, format = "%Y%m%d")
norm_chars = norm_chars.set_index(["date","permno"])

mret = mret.rename(columns = {"ret_exc_lead1m": "ret"})
mret = mret.reset_index()
mret.date = pd.to_datetime(mret.date, format = "%Y%m%d")
mret = mret.set_index(["date","permno"])
mret = mret.dropna()

intersection = pd.merge(norm_chars.reset_index(), mret.reset_index(), on = ["date","permno"])
intersection = intersection.set_index(["date","permno"]).sort_index()
mret = intersection[["ret"]]
norm_chars = intersection[[c for c in intersection.columns if not c=="ret"]]
   
dates = list(norm_chars.index.get_level_values(level=0).unique())

results = []
for i in range(len(dates) - window_length):
    start_idx = i
    
    sample_dates = dates[start_idx:(i+window_length+1)]
    predict_date = sample_dates[-1]
    sample_dates = sample_dates[:-1]

    chars = norm_chars.loc[sample_dates].copy()
    ret = mret.loc[sample_dates].copy()
    ret = ret.groupby("date").ret.transform(lambda x: winsorize(x, [0.01, 0.01]))
        
    kf = KFoldMonths(n_splits=3, weighted=True)
    
    reg = LassoCV(cv=kf.split(chars,ret), random_state=0).fit(chars.values, np.squeeze(ret.values))
    results.append(
        pd.DataFrame({
            "predict_date": predict_date,
            "n_non_zero_char": (np.abs(reg.coef_) > 0).sum()}, index = [0]))

results = pd.concat(results)
results.to_parquet(DATA_DERIVED / "lasso_sparsity_number.pq")
