# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 09:01:03 2026

@author: moerk
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
