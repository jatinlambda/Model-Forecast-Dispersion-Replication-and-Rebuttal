"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Defines frequently used settings and functions 
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

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
DATA_EXTERNAL = ROOT / "data" / "external"
DATA_MFD = ROOT / "data" / "mfd"  
DATA_RESULTS = ROOT / "results"

# %% settings

# Graphics
cm = 1 / 2.54
width = 15.92 * cm
height = 9 * cm

# tables
num_format="%.2f"
num_format_t="%.2f"

# % lags in NW
maxlags = 6

# % portfolio sorts
bps = ""
bp_min_n = 10 
pfs = 10

mapper = {
    "SUE": "niq_su", # standard earnings surprise
    "AG": "at_gr1", # asset growth
    "MOM": "ret_12_1", # t-12 t-2 momentum
    "ILLIQ": "ami_126d", # amihud illiquidity
    "OP": "ope_be", # gross profitability
    "IVOL": "ivol_capm_252d", # ivol prev. month
    "BETA": "beta_60m", # market beta
    "SIZE": "market_equity",
    "BM": "be_me",
    "MAX": "rmax5_21d", # max return prev 1 month
    "TURN": "turnover_126d",
    "STR": "ret_1_0", # short term reversal,
    "MFD": "mfd",
    "AFD": "afd",
    "MISP": "MISP",
    "MFDxMISP": "mfd_MISP",
    "MFDxINST": "mfd_ior",
    "INST": "ior",
    "UTIL": "Utilisation",
    "IFEE": "IndicativeFee",
    "MFDxUTIL": "mfd_Utilisation",
    "MFDxIFEE": "mfd_IndicativeFee",
    "MFDxARB": "mfd_arb",
    "ARB": "arb",
    "r$RET_{t+1}$": "ret_exc_lead1m",
    "r$RET_{t}$": "ret_exc_contemp",
    "MFM": "mfd_mean",
    "MFD_TOTAL": "mfd_total",
    "MFD_WITHIN": "mfd_within_model",
    "MFD_ACROSS": "mfd_across_model"
    }

mapper_rev = {v: k for k, v in mapper.items()}

savepath = "../../results/"

# FF data
from pandas_datareader.famafrench import get_available_datasets

def read_dataset(dataset_name, verbose=False):
    """Reads Fama/French datasets."""

    # get the number of rows to skip.
    read = False
    skipped_rows = 0
    while read is False:
        try:
            data = pd.read_csv(
                ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/" + dataset_name + "_CSV.zip"),
                skiprows=skipped_rows,
                index_col=0,
                nrows=1000,
            )
            read = True
        except (pd.errors.ParserError, IndexError, UnicodeDecodeError) as e:
            if verbose:
                print(e, ", skipping an additional row.")
            skipped_rows += 1

    # get row of first "real" date:
    if (data.index.dtype) == "object":
        dates = data.index.str.extract(r"^(\d{8}|\d{6})", expand=False)
        ii_first_date = np.where(~dates.isnull())[0][0]
        if ii_first_date > 0:
            ii_first_date += 1
    else:
        ii_first_date = 0

    # read in data:
    data = pd.read_csv(
        ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/" + dataset_name + "_CSV.zip"),
        skiprows=ii_first_date + skipped_rows,
        index_col=0,
    )

    # skip annual data at the bottom:
    annual_data_cutoff = np.where(data.isnull().all(axis=1))[0]
    if len(annual_data_cutoff > 0):
        data = data.iloc[: annual_data_cutoff[0]]

    # check type of data: monthly vs. daily:
    if "daily" in dataset_name:
        data.index = pd.to_datetime(data.index, format="%Y%m%d")
    else:
        data.index = pd.to_datetime(data.index, format="%Y%m").to_period("M")

    # data is given in percentage points:
    data = data.astype("f8")
    data /= 100

    return data

def get_datasets():
    return get_available_datasets()

# %% Compute factor returns
import datetime

def get_factors():

    ff = read_dataset("F-F_Research_Data_5_Factors_2x3")
    mom = read_dataset("F-F_Momentum_Factor")
    mom.columns = ["MOM"]
    ff = ff.merge(mom, left_index=True, right_index=True)
    rf = ff[["RF"]]
    ff = ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]]
    
    stambaugh = pd.read_csv(DATA_EXTERNAL / "stambaugh.csv")
    stambaugh["YYYYMM"] = pd.to_datetime(stambaugh["YYYYMM"], format="%Y%m").dt.to_period("M")
    stambaugh = stambaugh.set_index("YYYYMM")
    stambaugh = stambaugh.drop(columns=["RF"])
    
    hxz = pd.read_csv(DATA_EXTERNAL / "q5_factors_monthly_2021.csv")
    hxz["date"] = hxz.apply(lambda r: datetime.datetime(int(r["year"]), int(r["month"]),1), axis = 1)
    hxz["YYYYMM"] = hxz["date"].dt.to_period("M")
    hxz = hxz.set_index("YYYYMM")
    hxz = hxz.drop(columns=["R_F","year","month","date"])
    hxz /= 100
    
    dhs = pd.read_csv(DATA_EXTERNAL / "DHS.csv")
    dhs["Date"] = pd.to_datetime(dhs.Date, format = "%Y%m").dt.to_period("M")
    dhs = dhs.rename(columns = {"Date":"date"})
    dhs = dhs.set_index("date")
    dhs /= 100
    dhs = dhs.merge(ff[["Mkt-RF"]], left_index = True, right_index = True)
    
    factors = {}
    factors["RF"] = rf
    factors["CAPM"] = ff[["Mkt-RF"]]
    factors["FF6"] = ff[["Mkt-RF","SMB","HML","RMW","CMA","MOM"]]
    factors["HXZ"] = hxz
    factors["SY"] = stambaugh
    factors["DHS"] = dhs
    
    return factors

def get_international_factors():
    import os
    folder = DATA_EXTERNAL / "kenneth_french_international"
    folder = str(folder) + os.path.sep
    
    names = ["Developed", "Developed_ex_US", "Europe"]
    
    international = {}
    
    for name in names:
        ff5 = pd.read_csv(folder + name + "_5_Factors.csv", skiprows = 6, header = 0, 
                          skipfooter=34, engine = "python")
        ff5 = ff5.rename(columns = {"Unnamed: 0": "date"}).drop(columns = "RF")
        ff5.date = pd.to_datetime(ff5.date, format = "%Y%m").dt.to_period("M")
        ff5 = ff5.set_index("date")
        ff5 = ff5 / 100
        mom = pd.read_csv(folder + name + "_MOM_Factor.csv", skiprows = 6, header = 0, 
                          skipfooter=34, engine = "python")
        mom = mom.rename(columns = {"Unnamed: 0": "date"})
        mom.date = pd.to_datetime(mom.date, format = "%Y%m").dt.to_period("M")
        mom = mom.set_index("date")
        mom = mom / 100
        
        ff6 = pd.merge(ff5, mom, left_index = True, right_index = True).reset_index()
        
        international[name.lower()] = ff6
    
    return international

# %% load data

def get_short_interest():
    rsi = pd.read_parquet(DATA_RAW / "short_interest.pq")
    rsi = rsi.rename(columns = {"period":"date"})
    #rsi.date = rsi.date.dt.to_timestamp("M")
    return rsi
    
def get_norm_chars(typ = "norm"):
    if typ == "unnorm":
        norm_chars = pd.read_parquet(DATA_DERIVED / 'JKP_chars.pq')
    elif typ == "norm":
        norm_chars = pd.read_parquet(DATA_DERIVED / 'JKP_ranknorm.pq')
    else:
        norm_chars = []
        print("Unknown data from ... return None")
    if not(norm_chars is None):
        norm_chars =norm_chars.reset_index()
        norm_chars.date = pd.to_datetime(norm_chars.date, format = "%Y%m%d")
        norm_chars.date = norm_chars.date.dt.to_period("M").dt.to_timestamp("M")
    return norm_chars

def get_misp():
    misp = pd.read_csv(DATA_EXTERNAL / "Misp_Score.txt", delimiter = ";")
    misp["date"] = pd.to_datetime(misp.yyyymm, format = "%Y%m")
    misp.date = misp.date.dt.to_period("M").dt.to_timestamp("M")
    misp = misp.drop(columns = ["yyyymm"])
    misp = misp.rename(columns = {"avg_score":"MISP"})
    
    misp2 = pd.read_csv(DATA_RAW / "misp.csv")
    misp2["date"] = pd.to_datetime(misp2.yyyymm, format = "%Y%m")
    misp2.date = misp2.date.dt.to_period("M").dt.to_timestamp("M")
    misp2 = misp2.drop(columns = ["yyyymm"])
    misp2 = misp2.rename(columns = {"misp":"MISP"})
    misp2 = misp2[misp2.date >= "2017-01-01"]
    misp = pd.concat([misp, misp2]).sort_values(["date","permno"])
    
    return misp

def get_green():
    green_file = DATA_RAW / "green.csv"

    green = pd.read_csv(green_file, usecols = ["DATE","permno","disp"])
    green["DATE"] = pd.to_datetime(green.DATE, format = "%Y%m%d")
    green = green.rename(columns = {"DATE":"date"})
    green = green.sort_values(["date","permno"])
    green = green.dropna(subset = ["disp"])
    green = green.rename({"disp": "afd"},axis = 1)

    # same dates
    green.date = green.date.dt.to_period("M").dt.to_timestamp("M")
    
    return green

def get_ior():
    ior = pd.read_parquet(DATA_RAW / "ior_dbreadth_hhi.pq")
    return ior

# %%% Portfolio sorts and risk-attribution

def ecdf(grp, col, port_name, pfs):
    grp["cdf"] = grp[col].rank(method="dense", pct = True)
    grp["min_cdf"] = grp["cdf"].min()
    grp.loc[grp.min_cdf == grp.cdf,"cdf"] = 0.00000001
    grp[port_name] = np.ceil(grp.cdf * pfs)
    grp.loc[grp[port_name] == 0,port_name] = 1
    return grp

def do_portfolio_sort(sub, col, port_name = "port", pfs = 10, bps = "", bp_min_n = 10):
    if bps == "nyse":
        sub["bp_stock"] = ((sub.crsp_exchcd  == 1) & np.isnan(sub.comp_exchg)) | ((sub.comp_exchg  == 1) & np.isnan(sub.crsp_exchcd))
    elif bps == "":
        sub["bp_stock"] = True    
    
    sub_bp_n = sub.groupby("date").bp_stock.sum().to_frame()
    sub_bp_n.columns = ["bp_n"]
    if "bp_n" in sub:
        sub = sub.drop(columns = ["bp_n"])
    sub = pd.merge(sub, sub_bp_n, right_index = True, left_on = ["date"])
    sub = sub.loc[sub.bp_n >= bp_min_n]
    
    sub = sub.groupby("date", as_index = False).apply(lambda x: ecdf(x, col, port_name, pfs))
    return sub
    
def do_portfolio_stats(grp, weighting, ret_col = "ret_exc_lead1m", w_col = "market_equity"):
    grp = grp.dropna(subset = ret_col)
    if weighting == "ew":
        ret = grp[ret_col].mean()
    elif weighting == "vw":
        ret = (grp[ret_col] * grp[w_col]).sum() / grp[w_col].sum()
    else:
        ret = (grp[ret_col] * grp.me_cap).sum() / grp.me_cap.sum()
    return ret

def do_risk_attribution(returns, factors_name, ap_factors, col_port = "port", col_return = "XRet"):
    factors = ap_factors[factors_name].copy()
    factors.index = factors.index - 1
    factors = sm.add_constant(factors)
    cols = factors.columns.values
    data = returns.reset_index().merge(factors, left_on = "date", right_index = True)
    data = data.set_index(["date",col_port])
    gp = data.groupby([col_port],sort=False)
    alphas = gp.apply(
            lambda x: sm.OLS(x[col_return], x[cols])
            .fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
            .params["const"])
    alphas.name = factors_name
    alphas = alphas.to_frame()
    t_val =  gp.apply(
            lambda x: sm.OLS(x[col_return], x[cols])
            .fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
            .tvalues["const"])
    t_val.name = "t_"+factors_name
    t_val = t_val.to_frame()
    
    res = pd.merge(alphas, t_val, left_index = True, right_index = True)
    return res

def do_bivariate_portfolio_sorts_spreads(sub, weighting, col, controls, ap_factors, pfs, w_col = "me"):
    
    pfs_c = pfs[0]
    pfs_ = pfs[1]
    
    res = dict()
    for control in controls:
        tmp = sub.copy()
        tmp = do_portfolio_sort(tmp.dropna(subset = [control]), control, "port_c", pfs_c, "", pfs_c)
        tmp = tmp.groupby("port_c", as_index = False).apply(lambda x: do_portfolio_sort(x, col, "port_", pfs_, "", pfs_))
        
        tmp.port_c = tmp.port_c.astype(int)
        tmp.port_ = tmp.port_.astype(int)

        pf_ret = tmp.groupby(["date","port_c","port_"]).apply(lambda x: do_portfolio_stats(x, weighting, "ret_exc_lead1m", w_col)).to_frame()
        pf_ret.columns = ["ret"]
        pf_ret *= 100
        pf_ret = pf_ret.reset_index()
        pf_ret.date = pf_ret.date.dt.to_period("M")
        
        #rf = ap_factors["RF"]
        #pf_ret = pf_ret.merge(rf, left_on = "date", right_index = True)
        pf_ret["XRet"] = pf_ret["ret"] #- pf_ret["RF"]
        pf_ret = pf_ret.set_index(["date","port_c","port_"])
            
        returns = pf_ret["XRet"]
        returns = returns.unstack()
        returns = returns.reset_index().set_index(["port_c","date"])
    
        returns["H-L"] = returns[pfs_] - returns[1]
        
        hml = returns.loc[pfs_c]- returns.loc[1]
        hml["port_c"] = "H-L"
        hml = hml.reset_index().set_index(["port_c", "date"])
        
        tmp = pd.concat((returns, hml)) 
        tmp = tmp[["H-L"]]
        
        
        means = tmp.groupby("port_c").mean()

        ts = (
            tmp.groupby("port_c")
            .apply(
                lambda x: x.apply(
                    lambda y: sm.OLS(y.dropna(), np.ones(y.dropna().shape)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags, "kernel": "bartlett"}).tvalues
                )
            )
            .reset_index(level=1, drop=True)
        )
        
        if not(ap_factors is None):
            for k in ap_factors:
                con = ap_factors[k].copy()
                con_col = list(con.columns)
                con.index.name = "date"
                con = tmp.reset_index().merge(con.reset_index(), on = "date").set_index(["port_c","date"])
                con["const"] = 1
                
                alpha = con.groupby("port_c").apply(lambda x: sm.OLS(x["H-L"], x[["const"] + con_col]).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags, "kernel": "bartlett"}).params["const"])
                tstat = con.groupby("port_c").apply(lambda x: sm.OLS(x["H-L"], x[["const"] + con_col]).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags, "kernel": "bartlett"}).tvalues["const"])
                alpha = alpha.to_frame(name = k)
                means = means.merge(alpha, left_index = True, right_index = True)
                tstat = tstat.to_frame(name = k)
                ts = ts.merge(tstat, left_index = True, right_index = True)
            
        rest = {"means": means,
                    "tstat": ts}
        res[control] = rest
    
    return res

def do_bivariate_sorts_controlling(sub, weighting, col, controls, ap_factors, pfs, bps, bp_min_n, w_col = "market_equity", typ = "dependent"):
    
    pfs_c = pfs[0]
    pfs_ = pfs[1]
    bps_c = bps[0]
    bps_ = bps[1]
    bp_min_n_c = bp_min_n[0]
    bp_min_n_ = bp_min_n[1]
    
    res = pd.DataFrame([])
    for control in controls:
        tmp = sub.copy()
        if typ == "dependent":
            tmp = do_portfolio_sort(tmp.dropna(subset = [control]), control, "port_c", pfs_c, bps_c, bp_min_n_c)
            tmp = tmp.groupby("port_c", as_index = False).apply(lambda x: do_portfolio_sort(x, col, "port_", pfs_, bps_, bp_min_n_))
        else:
            tmp = tmp.dropna(subset = [control])
            tmp = do_portfolio_sort(tmp, control, "port_c", pfs_c, bps_c, bp_min_n_c)
            tmp = do_portfolio_sort(tmp, col, "port_", pfs_, bps_, bp_min_n_)
        tmp.port_c = tmp.port_c.astype(int)
        tmp.port_ = tmp.port_.astype(int)
        
        returns = tmp.groupby(["date","port_c","port_"]).apply(lambda x: do_portfolio_stats(x, weighting, w_col = w_col)).to_frame()
        returns.columns = ["ret"]
        returns *= 100
        returns = returns.reset_index()
        returns.date = returns.date.dt.to_period("M")
        
        returns = returns.groupby(["date","port_"]).ret.mean()
        returns = returns.unstack()
        returns.columns = returns.columns.get_level_values(0)
        returns["H-L"] = returns[pfs_] - returns[1]
        returns = returns.stack().to_frame()
        returns.columns = ["XRet"]
        returns = returns.sort_index(level=(0,1))
        
        
        grouper = returns.groupby("port_")
        strat = grouper.apply(
                lambda x: (x["XRet"].mean(),sm.OLS(x["XRet"], np.ones(x["XRet"].shape))
                .fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
                .tvalues[0]))
        
        for k in ap_factors.keys():
            if k != "RF":
                t = do_risk_attribution(returns[returns.index.get_level_values("port_") == "H-L"], k, ap_factors, col_port = "port_")
                strat[k] = (t[k].iloc[0], t["t_"+k].iloc[0])
                
        strat = strat.to_frame(control)
        
        if res.empty:
            res = strat
        else:
            res = res.merge(strat, left_index = True, right_index = True)
    
    return res
        
def do_strategy_overview(sub, weighting, col, ap_factors, pfs, bps, bp_min_n, ret_col = "ret_exc_lead1m", w_col = "me"):
    sub = do_portfolio_sort(sub, col, "port", pfs, bps, bp_min_n)
    sub.port = sub.port.astype(int)
    
    pf_ret = sub.groupby(["date","port"]).apply(lambda x: do_portfolio_stats(x, weighting, ret_col, w_col)).to_frame()
    pf_ret.columns = ["ret"]
    pf_ret *= 100
    pf_ret = pf_ret.reset_index()
    pf_ret.date = pf_ret.date.dt.to_period("M")
   
    #rf = ap_factors["RF"]
    #pf_ret = pf_ret.merge(rf, left_on = "date", right_index = True)
    pf_ret["XRet"] = pf_ret["ret"] #- pf_ret["RF"]
    
    returns = pf_ret.groupby(["date", "port"])["XRet"].mean()
    returns = returns.unstack()
    #returns.columns = "Lo 2 3 4 5 6 7 8 9 Hi".split()
    returns["H-L"] = returns[pfs] - returns[1]
    returns = returns.stack()
    returns.name = "XRet"
    returns = returns.to_frame()
    returns.index.names = ["date", "port"]
    
    grouper = returns.groupby("port")
    strat = grouper.mean()
    strat["tval"] = grouper.apply(
            lambda x: sm.OLS(x["XRet"], np.ones(x["XRet"].shape))
            .fit(cov_type="HAC", cov_kwds={"maxlags": maxlags,
                                           "kernel": "bartlett"})
            .tvalues)
    
    
    if ap_factors is not None:
        for k in ap_factors.keys():
            if k != "RF":
                t = do_risk_attribution(returns, k, ap_factors)
                strat = strat.merge(t, left_index = True, right_index = True)
      
    return strat, returns[returns.index.get_level_values("port") == "H-L"]


def bivariate_sort(sub, weighting, cols, ap_factors, pfs, bps, bp_min_n, ret_col = "ret_exc_lead1m", w_col = "me", typ = "dependent", include_alphas = None, abbrev = None, maxlags = maxlags):
    
    def index_renamer(i, m):
        if i == 1:
            return "Low"
        elif i == m:
            return "High"
        else:
            return str(i)
    
    
    col_c = cols[0]
    col_ = cols[1]
    pfs_c = pfs[0]
    pfs_ = pfs[1]
    bps_c = bps[0]
    bps_ = bps[1]
    bp_min_n_c = bp_min_n[0]
    bp_min_n_ = bp_min_n[1]
    
    if typ == "dependent":
        sub = do_portfolio_sort(sub, col_c, "port_c", pfs_c, bps_c, bp_min_n_c)
        sub = sub.groupby("port_c", as_index = False).apply(lambda x: do_portfolio_sort(x, col_, "port_", pfs_, bps_, bp_min_n_))
    elif typ == "independent":
        sub = do_portfolio_sort(sub, col_c, "port_c", pfs_c, bps_c, bp_min_n_c)
        sub = do_portfolio_sort(sub, col_, "port_", pfs_, bps_, bp_min_n_)
    else:
        print("Type is not dependent nor independent... return None")
        return None
        
    sub.port_c = sub.port_c.astype(int)
    sub.port_ = sub.port_.astype(int)
    
    pf_ret = sub.groupby(["date","port_c","port_"]).apply(lambda x: do_portfolio_stats(x, weighting, ret_col, w_col)).to_frame()
    pf_ret.columns = ["ret"]
    pf_ret *= 100
    pf_ret = pf_ret.reset_index()
    pf_ret.date = pf_ret.date.dt.to_period("M")
    
    pf_ret["XRet"] = pf_ret["ret"] #- pf_ret["RF"]
    pf_ret = pf_ret.set_index(["date","port_c","port_"])
        
    returns = pf_ret["XRet"]
    returns = returns.unstack()
    if returns.shape[1] == 10:
        returns.columns = "Low 2 3 4 5 6 7 8 9 High".split(" ")
    else:
        returns.columns = "Low 2 3 4 High".split(" ")
    returns = returns.reset_index().set_index(["port_c","date"])

    returns["H-L"] = returns["High"] - returns["Low"]
    
    hml = returns.loc[pfs_c]- returns.loc[1]
    hml["port_c"] = "H-L"
    hml = hml.reset_index().set_index(["port_c", "date"])
    
    tmp = pd.concat((returns, hml))    
    
    means = tmp.groupby("port_c").mean()

    ts = (
        tmp.groupby("port_c")
        .apply(
            lambda x: x.apply(
                lambda y: sm.OLS(y.dropna(), np.ones(y.dropna().shape)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags, "kernel": "bartlett"}).tvalues
            )
        )
        .reset_index(level=1, drop=True)
    )
    
    if include_alphas is not None:
        hl = tmp[["H-L"]]
        for k in include_alphas:
            con = ap_factors[k].copy()
            con_col = list(con.columns)
            
            con.index = con.index - 1
            
            con.index.name = "date"
            con = hl.reset_index().merge(con.reset_index(), on = "date").set_index(["port_c","date"])
            con["const"] = 1
            
            alpha = con.groupby("port_c").apply(lambda x: sm.OLS(x["H-L"], x[["const"] + con_col]).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags, "kernel": "bartlett"}).params["const"])
            tstat = con.groupby("port_c").apply(lambda x: sm.OLS(x["H-L"], x[["const"] + con_col]).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags, "kernel": "bartlett"}).tvalues["const"])
            alpha = alpha.to_frame(name = k)
            means = means.merge(alpha, left_index = True, right_index = True)
            tstat = tstat.to_frame(name = k)
            ts = ts.merge(tstat, left_index = True, right_index = True)
    
    if abbrev is not None:
        colc = abbrev
    else:
        colc = col_c
    
    means.index = pd.Index([colc + " " + index_renamer(x,pfs_c) for x in means.index], name = "port_c")
    
    ts.index = means.index
    
    return means, ts

def bivariate_sort_to_latex(panelA, panelB, panelC, tsB, tsC, fname, num_format="%.2f"):
    name = panelA.index.values[0]
    n_pf = panelA.shape[1]
    n_a = panelB.shape[1] - (n_pf - 1)
    additional_rf = list(set(panelB.columns)-set(panelA.columns))
    ncol = 1+ n_pf + 2*len(additional_rf)
    
    for add in ["H-L"] + additional_rf:
        idx = panelB.columns.to_list().index(add)
        panelB.insert(idx+1, column = "t-stat_"+add, value = tsB[add].values)
        
    if panelC is not None:
        for add in ["H-L"] + additional_rf:
            idx = panelC.columns.to_list().index(add)
            panelC.insert(idx+1, column = "t-stat_"+add, value = tsC[add].values)

    tabular = Tabular("l" + "c" * (ncol - 1), booktabs=True)
    to_add =  [MultiColumn(ncol, data = "Panel A: Average {} in MFD Decile Portfolio".format(name), align = "l")]
    tabular.add_row(to_add)
    tabular.add_hline(0,n_pf)
    to_add = [""] + list(panelA.columns) + [""] * 2 * n_a
    tabular.add_row(to_add)
    tabular.add_hline(0,n_pf)
    to_add = [name]
    for n,v in panelA.T.iterrows():
        v = num_format % v.iloc[0]
        if n in "H-L":
            v = v + sig_stars(panelA["t-stat"].iloc[0])
        to_add.append(v)
    to_add = to_add + [""] * 2 * n_a
    tabular.add_row(to_add)
    tabular.add_hline()
    
    if panelC is not None:
        addon = "Equal-Weight"
    else:
        addon = ""
    
    to_add =  [MultiColumn(ncol, data = "Panel B: {} Bivariate Portfolio Sort on {}".format(addon, name), align = "l")]
    tabular.add_row(to_add)
    tabular.add_hline()
    to_add = [""] + [x.split("_")[0] for x in panelB.columns]
    tabular.add_row(to_add)
    tabular.add_hline(0,ncol-1)
    for idx, row in panelB.iterrows():
        to_add = [idx]
        for n, r in row.items():
            r = num_format % r
            if n in ["H-L"] + additional_rf:
                r = r + sig_stars(row["t-stat_"+n])
                
            to_add.append( r )
        tabular.add_row(to_add)

    if panelC is not None:
        tabular.add_hline()
        to_add =  [MultiColumn(ncol, data = "Panel C: Value-Weight Bivariate Portfolio Sort on {}".format(name), align = "l")]
        tabular.add_row(to_add)
        tabular.add_hline()
        to_add = [""] + [x.split("_")[0] for x in panelC.columns]
        tabular.add_row(to_add)
        tabular.add_hline(0,ncol-1)
        for idx, row in panelC.iterrows():
            to_add = [idx]
            for n, r in row.items():
                r = num_format % r
                if n in ["H-L"] + additional_rf:
                    r = r + sig_stars(row["t-stat_"+n])
                    
                to_add.append( r )
            tabular.add_row(to_add)
    
    tabular.generate_tex(fname)
    print("table writen to {}".format(fname))
    

def bivariate_sort_to_latex_no_avg(panelB, tsB, n_pf, fname, num_format="%.2f"):
    
    for add in ["H-L", "FF6"]:
        idx = panelB.columns.to_list().index(add)
        panelB.insert(idx+1, column = "t-stat_"+add, value = tsB[add].values)
        
    ncol = 10

    tabular = Tabular("l" + "c" * (ncol - 1), booktabs=True)
    to_add = [""] + [x.split("_")[0] for x in panelB.columns]
    tabular.add_row(to_add)
    tabular.add_hline(0,ncol-1)
    for idx, row in panelB.iterrows():
        to_add = [idx]
        for n, r in row.items():
            r = num_format % r
            if n in ["H-L", "FF6"]:
                r = r + sig_stars(row["t-stat_"+n])
                
            to_add.append( r )
        tabular.add_row(to_add)

    tabular.generate_tex(fname)
    print("table writen to {}".format(fname))


# %%% Tables

from pylatex import (
    Tabular,
    MultiColumn,
    MultiRow,
    Math
)

def dollar(x):
    return Math(data=x, inline=True, escape=False)

def average_char_decile(mfd_in, char, c, port = "mfd_port", agg_type = lambda x: np.nanmedian(x)):
    ttmp = mfd_in.copy()
    ttmp = ttmp.sort_values("date")
    mfd_char = ttmp.groupby(["date",port])[char].apply(agg_type) # np.nanmedian nanmean
    mfd_char = mfd_char.unstack()
    if 10 in ttmp[port].unique():
        mfd_char.columns = "Low 2 3 4 5 6 7 8 9 High".split(" ")
    else:
        mfd_char.columns = "Low 2 3 4 High".split(" ")
    mfd_char["H-L"] = mfd_char["High"] - mfd_char["Low"]
    hl = mfd_char[["H-L"]].copy()
    hl["const"] = 1
    hl = hl.dropna()
    mfd_char = mfd_char.mean()
    tval = sm.OLS(hl["H-L"], hl["const"]).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags}).tvalues["const"]
    mfd_char["t-stat"] = tval
    return mfd_char.to_frame(name = c).T

def float_to_str(numvar, prec=3):
    if np.isnan(numvar):
        return ""
    else:
        fmt = "{:." + str(prec) + "f}"
        return fmt.format(numvar)

def thousander_separator(value):
    return "{:,}".format(value)

def sig_stars(val):
    return "*" * np.sum(np.abs(val) > np.array([1.6449, 1.96, 2.5758]))

def each_specification(res, prec=3, c=None):
    params = pd.Series(
        [(float_to_str(x, prec) + sig_stars(y), float_to_str(y, prec)) for x, y in zip(res.params, res.tstats)],
        index=res.params.index,
    )
    params["Observations"] = thousander_separator(res.nobs)
    params["R-squared"] = float_to_str(res.rsquared * 100)
    if c:
        params.name = c
    else:
        params.name = res.model.dependent.vars[0]
    return params.to_frame()

def fmb_to_table(
        results,
        file,
        kick = ["R-squared"],
        prec = 3,
        colnames = None,
        order = None,
        renamer = None):
    
    def apply_renamer(renamer, inp):
        if inp in renamer.keys():
            return renamer[inp]
        else:
            return inp
        
    if colnames is not None:
        if len(colnames) < len(results):
            colnames = np.repeat(colnames, len(results))
    else:
        colnames = [x.model.dependent.vars[0] for x in results ]
    
    all_controls = np.concatenate([ x.params.index.values for x in results])
    indexes = np.unique(all_controls, return_index=True)[1]
    all_controls = list(all_controls[sorted(indexes)])
    
    # move Const as number 1
    all_controls = ["Const"] + [c for c in all_controls if not(c=="Const")]
    
    # order
    if order is not None:
        order = ["Const"] + order
        all_controls = order + [x for x in all_controls if x not in order]
    
    data = pd.concat([each_specification(x, prec, c) for c,x in zip(colnames, results)], axis = 1, sort = False)
    
    end_cols = list(set(data.index)-set(all_controls))
    
    data = data.loc[all_controls + end_cols]
    
    if kick:
        data = data.drop(index = kick)
    
    if renamer:
        if not(colnames[0]):
            data.columns = [apply_renamer(renamer,c) for c in data.columns]
        data.index = [apply_renamer(renamer,i) for i in data.index]
        
    ncol = data.shape[1]
    tabular = Tabular("l" + "c" * (ncol), booktabs=True)
    tabular.add_row([""]+list(data.columns))
    tabular.add_hline()
    
    for index, row in data.iterrows():
        if index == "Observations":
            tabular.add_hline(1, ncol+1)
        if index == "R-squared":
            index = dollar(r"R^2 \text{ (\%)}")
        
        if np.any([isinstance(x, list) for x in row]) | np.any([isinstance(x, tuple) for x in row]):
            cells = [index]
            for r in row:
                if isinstance(r, list) | isinstance(r, tuple):
                    cells.append(r[0])
                else:
                    if np.isnan(r):
                        cells.append("")
                    else:
                        cells.append(r)
            tabular.add_row(cells)
            cells = [""]
            for r in row:
                if isinstance(r, list) | isinstance(r, tuple):
                    cells.append("(" + r[1] + ")")
                else:
                    cells.append("")
            tabular.add_row(cells)
        else:
            cells = [index]
            for r in row:
                cells.append(r)
            tabular.add_row(cells)

        if index == "Dependent":
            tabular.add_hline()
            
    tabular.generate_tex(file)
    print("Writen to {}".format(file))
