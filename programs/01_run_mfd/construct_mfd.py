"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Constructs machine forecast disagreement (MFD) from rank-normalized JKP
characteristics and next-month excess returns.

------------------------------------------------------------------------------
MODIFIED for the "Do optimists set prices?" extension (J. Lamba, Nov 2026).
Original file: data/pkg_extracted/programs/01_run_mfd/construct_mfd.py

Changes vs. the authors' version (model / data logic is unchanged):
  1. Per stock-month we now store the full forecast distribution summary:
     mean, std (= MFD, identical to the paper), quantiles q10/q20/q50/q80/q90,
     and the split spreads
         opt    = q80 - q50     pes    = q50 - q20
         opt90  = q90 - q50     pes10  = q50 - q10
  2. The 100 (or K) per-investor forecasts are saved in wide format
     (one row per permno, one float32 column per investor), so any other
     shape statistic can be computed later without retraining.
  3. Command-line parameters for the scaled-down run (number of investors,
     trees, test window, CPU count, output folder).
  4. Resumable: work is organised in refit blocks (fit_frequency months).
     A block whose monthly files all exist is skipped on rerun.
  5. Reproducible per block: each forest gets random_state derived from
     (seed, investor, block start date) instead of the global NumPy RNG, so a
     resumed run gives the same numbers as an uninterrupted one. Investor
     feature subsets still come from the global RNG seeded at 987654321, so
     investor k sees the same 76 characteristics whatever K is.
  6. Winsorized training returns are only computed when a refit happens
     (the original recomputed them every month and discarded them).

Defaults reproduce the paper's configuration (K=100, 2000 trees, full sample).
Scaled-down run from the proposal:
  python programs/01_run_mfd/construct_mfd.py --n-investors 30 \
      --n-estimators 1000 --test-start 2005-01 --test-end 2023-12
------------------------------------------------------------------------------
"""

import argparse
import datetime
import glob
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

np.random.seed(987654321)


# %% Command line

def parse_args():
    default_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    p = argparse.ArgumentParser(description="Construct MFD and forecast-distribution quantiles")
    p.add_argument("--n-investors", type=int, default=100)
    p.add_argument("--n-features", type=int, default=76, help="characteristics each investor sees")
    p.add_argument("--model-type", choices=["rf", "ridge"], default="rf",
                   help="investor model: random forest (paper's main) or ridge (paper's config 17)")
    p.add_argument("--alpha", type=float, default=0.01, help="ridge penalty (paper's config 17: 0.01)")
    p.add_argument("--n-estimators", type=int, default=2000)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--window", type=int, default=120, help="rolling training window, months")
    p.add_argument("--fit-frequency", type=int, default=12, help="refit every N months")
    p.add_argument("--test-start", default=None, help="first forecast month, YYYY-MM (default: first possible)")
    p.add_argument("--test-end", default=None, help="last forecast month, YYYY-MM (default: last available)")
    p.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.2, 0.5, 0.8, 0.9])
    p.add_argument("--no-save-forecasts", action="store_true", help="skip saving per-investor forecasts")
    p.add_argument("--n-jobs", type=int, default=default_jobs)
    p.add_argument("--seed", type=int, default=987654321)
    p.add_argument("--no-sklearnex", action="store_true", help="use plain scikit-learn")
    p.add_argument("--data-dir", default=None, help="folder with JKP_ranknorm.pq and RET.pq (default: <root>/data/derived)")
    p.add_argument("--out", default=None, help="output folder (default: <root>/data/mfd)")
    return p.parse_args()


args = parse_args() if __name__ == "__main__" else None


def maybe_patch_sklearn():
    if args is not None and args.no_sklearnex:
        return False
    if platform.system() not in ("Windows", "Linux"):
        return False
    try:
        from sklearnex import patch_sklearn  # provided by scikit-learn-intelex
        patch_sklearn()
        return True
    except Exception:
        return False


SKLEARNEX_ENABLED = maybe_patch_sklearn()
if not SKLEARNEX_ENABLED:
    print("sklearnex not available or disabled; using standard scikit-learn.")

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

try:
    import xgboost as xgb  # only needed for model_type == "xgbm"
except ImportError:
    xgb = None

n_jobs = args.n_jobs if args is not None else 16


# %% Classes (authors' code; ForecastModel / Investor.fit accept random_state)

class ForecastModel(BaseEstimator):
    def __init__(self, m_type, params, random_state=None):
        self.m_type = m_type

        if self.m_type == "lasso":
            self.model = Lasso(fit_intercept=True, **params)
        elif self.m_type == "OLS":
            self.model = LinearRegression(fit_intercept=True, **params)
        elif self.m_type == "ridge":
            self.model = Ridge(fit_intercept=True, **params)
        elif self.m_type == "huber":
            self.model = HuberRegressor(fit_intercept=True, **params, epsilon=1.35)
        elif self.m_type == "enet":
            self.model = ElasticNet(fit_intercept=True, **params)
        elif self.m_type == "rf":
            self.model = RandomForestRegressor(criterion='squared_error',
                                               n_jobs=n_jobs,
                                               random_state=random_state,
                                               **params)
        elif self.m_type == "xgbm":
            if xgb is None:
                raise ImportError("xgboost is required for model_type='xgbm'")
            self.model = xgb.XGBRegressor(tree_method="hist",
                                          objective="reg:squarederror",
                                          eval_metric="rmse",
                                          device="cpu",
                                          verbosity=0,
                                          nthread=8,
                                          random_state=random_state,
                                          **params)

    def fit(self, X, y):
        if self.m_type == "rf":
            self.model.fit(X, np.ravel(y))
        else:
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return - mean_squared_error(y, self.predict(X))


class Investor():

    def __init__(self, model_type, model_params, data_params):
        self.transformation = data_params["transformation"]
        self.m = data_params["m"]
        self.gamma = data_params["gamma"]
        self.random_feature = data_params["random_feature"]
        self.subsample = data_params.get("subsample", 1)
        self.colsample = data_params.get("colsample", 1)
        self.model_type = model_type
        self.model_params = model_params
        self.selected_features = None

    def set_weights(self, num_chars, feature_names):
        if self.transformation:
            if self.colsample < 1:
                m_frac = int(np.round(num_chars * self.colsample))
                tmp = np.zeros((num_chars, m_frac))
                enable = np.sort(list(np.random.choice(range(0, num_chars), m_frac, replace=False)))
                self.selected_features = [feature_names[x] for x in enable]
                tmp[enable] = np.eye(m_frac)
                weight_matrix = np.random.normal(loc=0, scale=self.gamma, size=(m_frac, self.m))
                self.weight_matrix = np.matmul(tmp, weight_matrix)
            else:
                self.weight_matrix = np.random.normal(loc=0, scale=self.gamma, size=(num_chars, self.m))
                self.selected_features = ["transformed_" + str(x) for x in range(self.m)]
        else:
            tmp = np.zeros((num_chars, self.m))
            enable = np.sort(list(np.random.choice(range(0, num_chars), self.m, replace=False)))
            self.selected_features = [feature_names[x] for x in enable]
            tmp[enable] = np.eye(self.m)
            self.weight_matrix = tmp

    def transform_chars(self, chars):
        chars = chars @ self.weight_matrix
        if self.random_feature.lower() == "relu":
            chars[chars < 0] = 0
        elif self.random_feature.lower() == "fourier":
            chars = np.hstack([np.sin(chars), np.cos(chars)])
        return chars

    def to_dict(self):
        config = {
            "transformation": self.transformation,
            "m": self.m,
            "gamma": self.gamma,
            "random_feature": self.random_feature,
            "model_type": self.model_type,
            "subsample": self.subsample,
            "colsample": self.colsample,
        }
        config.update(self.model_params)
        return config

    def fit(self, chars, ret, random_state=None):
        if self.subsample < 1:
            rng = np.random.default_rng(random_state)
            ids = rng.choice(ret.shape[0], int(self.subsample * ret.shape[0]), replace=False)
            chars = chars.iloc[ids].sort_index()
            ret = ret.iloc[ids].sort_index()
        chars = self.transform_chars(chars.values)
        self.model = ForecastModel(self.model_type, self.model_params, random_state=random_state)
        self.model.fit(chars, ret.values)

    def predict(self, predict_chars):
        """Returns a 1-D numpy array of forecasts, aligned with predict_chars rows."""
        return self.model.predict(self.transform_chars(predict_chars.values))


# %% Forecast-distribution summary (new)

def summarize_forecasts(P, permnos, quantiles):
    """P: (n_stocks, K) matrix of forecasts. Returns one row per stock."""
    out = pd.DataFrame({"permno": permnos})
    out["mean"] = P.mean(axis=1)
    out["std"] = P.std(axis=1, ddof=1)          # == pandas .std(), the paper's MFD
    qs = np.quantile(P, quantiles, axis=1)      # (n_q, n_stocks), linear interpolation
    qname = {q: f"q{int(round(q * 100)):02d}" for q in quantiles}
    for q, row in zip(quantiles, qs):
        out[qname[q]] = row
    if {0.2, 0.5, 0.8} <= set(quantiles):
        out["opt"] = out["q80"] - out["q50"]
        out["pes"] = out["q50"] - out["q20"]
    if {0.1, 0.5, 0.9} <= set(quantiles):
        out["opt90"] = out["q90"] - out["q50"]
        out["pes10"] = out["q50"] - out["q10"]
    return out


class MFD():

    def __init__(self, window_length, fit_frequency, winsorize, store_path, seed,
                 quantiles, save_forecasts):
        self.window_length = window_length
        self.fit_frequency = fit_frequency
        self.winsorize = winsorize
        self.store_path = Path(store_path)
        self.seed = seed
        self.quantiles = sorted(quantiles)
        self.save_forecasts = save_forecasts
        self.investors = []

    def append_investor(self, investors):
        if isinstance(investors, list):
            self.investors = self.investors + investors
        else:
            self.investors.append(investors)

    def n_investors(self):
        return len(self.investors)

    def set_data(self, norm_chars, mret):
        intersection = pd.merge(norm_chars.reset_index(), mret.reset_index(), on=["date", "permno"])
        intersection = intersection.set_index(["date", "permno"]).sort_index()

        self.mret = intersection[["ret"]]
        self.norm_chars = intersection[[c for c in intersection.columns if not c == "ret"]]
        self.dates = list(self.norm_chars.index.get_level_values(level=0).unique())
        self.num_chars = self.norm_chars.shape[1]

        # investor k's feature subset depends only on the global seed and k
        for i in range(self.n_investors()):
            self.investors[i].set_weights(self.num_chars, list(self.norm_chars.columns))

        # record which characteristics each investor sees
        self.store_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({k: inv.selected_features for k, inv in enumerate(self.investors)}) \
            .to_csv(self.store_path / "investor_features.csv", index=False)

    def _paths(self, date):
        tag = pd.Timestamp(date).strftime("%Y%m")
        return (self.store_path / f"month={tag}_disp.pq",
                self.store_path / f"month={tag}_forecasts.pq")

    def _done(self, date):
        disp, fc = self._paths(date)
        return disp.exists() and (fc.exists() or not self.save_forecasts)

    def _block_seed(self, investor, block_date):
        ym = int(pd.Timestamp(block_date).strftime("%Y%m"))
        return int(np.random.SeedSequence([self.seed, investor, ym]).generate_state(1)[0])

    def run(self, test_start=None, test_end=None):
        # forecast month index j uses training months j-window .. j-1
        first = self.window_length
        if test_start is not None:
            first = max(first, next(j for j, d in enumerate(self.dates) if d >= pd.Timestamp(test_start)))
        last = len(self.dates) - 1
        if test_end is not None:
            last = max(j for j, d in enumerate(self.dates) if d <= pd.Timestamp(test_end) + pd.offsets.MonthEnd(0))
        predict_idx = list(range(first, last + 1))
        print(f"Forecast months: {self.dates[first].date()} .. {self.dates[last].date()} "
              f"({len(predict_idx)}), training from {self.dates[first - self.window_length].date()}, "
              f"K={self.n_investors()} investors")

        blocks = [predict_idx[b:b + self.fit_frequency] for b in range(0, len(predict_idx), self.fit_frequency)]
        for block in blocks:
            todo = [j for j in block if not self._done(self.dates[j])]
            if not todo:
                print(f"{self.dates[block[0]].date()}: block complete, skipping")
                continue

            # fit on the window ending just before the block's first month
            # (same timing as the original: refit every fit_frequency months)
            j0 = block[0]
            sample_dates = self.dates[j0 - self.window_length:j0]
            chars = self.norm_chars.loc[sample_dates]
            ret = self.mret.loc[sample_dates]
            if self.winsorize:
                ret = ret.groupby("date").ret.transform(lambda x: winsorize(x, [0.01, 0.01]))

            t0 = time.time()
            print(f"{datetime.datetime.now():%H:%M:%S} fitting block {self.dates[j0].date()} "
                  f"on {len(sample_dates)} months, {len(ret):,} obs")
            for k, inv in enumerate(self.investors):
                inv.fit(chars, ret, random_state=self._block_seed(k, self.dates[j0]))
            print(f"    fit done in {time.time() - t0:.0f}s")
            del chars, ret

            for j in todo:
                predict_date = self.dates[j]
                predict_chars = self.norm_chars.loc[predict_date]
                P = np.column_stack([inv.predict(predict_chars) for inv in self.investors]).astype(np.float64)
                permnos = predict_chars.index.values

                disp = summarize_forecasts(P, permnos, self.quantiles)
                disp["date"] = predict_date
                disp_path, fc_path = self._paths(predict_date)

                if self.save_forecasts:
                    fc = pd.DataFrame(P.astype(np.float32), columns=[f"inv_{k}" for k in range(P.shape[1])])
                    fc.insert(0, "permno", permnos)
                    fc.insert(0, "date", predict_date)
                    fc.to_parquet(fc_path, index=False)
                disp.to_parquet(disp_path, index=False)  # written last: marks the month as done
                print(f"    {predict_date.date()}: {len(permnos):,} stocks")

        self.aggregate_singles(self.dates[first], self.dates[last])

    def aggregate_singles(self, start, end):
        files = sorted(glob.glob(str(self.store_path / "month=*_disp.pq")))
        d = pd.concat([pd.read_parquet(f) for f in files])
        d = d[(d.date >= start) & (d.date <= end)]
        self.dispersion = d.sort_values(["date", "permno"]).reset_index(drop=True)

    def save(self):
        # mfd.csv keeps the authors' columns (permno, mean, std, date) first so
        # post_process_mfd_data.py works unchanged; new columns are appended.
        cols = ["permno", "mean", "std", "date"]
        cols += [c for c in self.dispersion.columns if c not in cols]
        self.dispersion[cols].to_csv(self.store_path / "mfd.csv")
        self.dispersion[cols].to_parquet(self.store_path / "mfd_quantiles.pq", index=False)
        print(f"Saved {len(self.dispersion):,} stock-months to {self.store_path}/mfd.csv and mfd_quantiles.pq")


# %% Main

def get_project_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2]
    return Path.cwd().resolve()


if __name__ == "__main__":
    ROOT = get_project_root()
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / "data" / "derived"
    out_dir = Path(args.out) if args.out else ROOT / "data" / "mfd"

    # Read only the months this run needs (training window + test period).
    # Filtering inside pyarrow avoids loading the full 2 GB file per process.
    filters = []
    if args.test_start is not None:
        lo = pd.Timestamp(args.test_start) - pd.DateOffset(months=args.window + 1)
        filters.append(("date", ">=", int(lo.strftime("%Y%m%d"))))
    if args.test_end is not None:
        hi = pd.Timestamp(args.test_end) + pd.offsets.MonthEnd(0)
        filters.append(("date", "<=", int(hi.strftime("%Y%m%d"))))
    filters = filters or None
    norm_chars = pd.read_parquet(data_dir / 'JKP_ranknorm.pq', filters=filters)
    mret = pd.read_parquet(data_dir / 'RET.pq', filters=filters)

    norm_chars = norm_chars.reset_index()
    norm_chars.date = pd.to_datetime(norm_chars.date, format="%Y%m%d")
    norm_chars = norm_chars.set_index(["date", "permno"])

    mret = mret.rename(columns={"ret_exc_lead1m": "ret"})
    mret = mret.reset_index()
    mret.date = pd.to_datetime(mret.date, format="%Y%m%d")
    mret = mret.set_index(["date", "permno"])
    mret = mret.dropna()

    # only keep months the run can use (training window + test period)
    if args.test_start is not None:
        lo = pd.Timestamp(args.test_start) - pd.DateOffset(months=args.window + 1)
        norm_chars = norm_chars[norm_chars.index.get_level_values(0) >= lo]
        mret = mret[mret.index.get_level_values(0) >= lo]
    if args.test_end is not None:
        hi = pd.Timestamp(args.test_end) + pd.offsets.MonthEnd(0)
        norm_chars = norm_chars[norm_chars.index.get_level_values(0) <= hi]
        mret = mret[mret.index.get_level_values(0) <= hi]

    # paper configuration (authors' values); n, trees and depth adjustable
    investors = []
    for i in range(args.n_investors):
        data_params = {
            'transformation': False,
            'm': args.n_features,
            'gamma': 0.1,
            'random_feature': 'None',
            'colsample': 1,
            'subsample': 1,
        }
        if args.model_type == "ridge":
            # deterministic: investors differ only through their characteristic subsets
            hyperparams = {"alpha": args.alpha}
        else:
            hyperparams = {
                "max_depth": args.max_depth,
                "n_estimators": args.n_estimators,
                "max_features": 0.05,
                "max_samples": 0.05,
                "min_samples_split": 5,
            }
        investors.append(Investor(args.model_type, hyperparams, data_params))

    mfd = MFD(args.window, args.fit_frequency, winsorize=True, store_path=out_dir,
              seed=args.seed, quantiles=args.quantiles,
              save_forecasts=not args.no_save_forecasts)
    mfd.append_investor(investors)
    mfd.set_data(norm_chars, mret)
    # set_data keeps its own merged copy; free the originals (saves ~1-2 GB per
    # process, matters when several shards share a 32 GB Slurm allocation)
    del norm_chars, mret
    import gc; gc.collect()
    mfd.run(args.test_start, args.test_end)
    mfd.save()
