# Definitions

Reference for every variable, control and factor model used in the results and slides.
Code locations: controls in `scripts/test_miller.py` (`CONTROLS`), factor models in
`scripts/paper_tables.py` (`factor_models`), both copied from the authors'
`programs/02_analysis/helper.py`.

## Disagreement measures (one value per stock per month)

Each stock-month has K = 100 forecasts of next-month excess return, one per "investor".
An investor is one model fitted on a random 76 of the 153 JKP characteristics
(the same 76 for investor k in every variant). Training: rolling 120-month window,
refit every 12 months (Aug–Jul blocks), returns winsorised 1%/1% each month.

| Term | Definition |
|---|---|
| **MFD** | Standard deviation of the 100 forecasts (the paper's measure; `std`) |
| **Mean forecast** | Average of the 100 forecasts (`mean`): the investors' consensus expected return |
| **Q20, Q50, Q80** | 20th, 50th, 80th percentile of the 100 forecasts |
| **Optimist spread** | Q80 − Q50 (`opt`) |
| **Pessimist spread** | Q50 − Q20 (`pes`) |
| **W (width)** | Q80 − Q20 = optimist + pessimist spread |
| **A (asymmetry)** | (Q80 − Q50) − (Q50 − Q20) = optimist − pessimist spread. Miller predicts A < 0 |

## Model variants

| Variant | Investor model | Source of disagreement | Data |
|---|---|---|---|
| **Forest** (paper's main) | Random forest: depth 3, 2,000 trees, 5% of features and rows per tree | Different characteristic subsets + forest randomness | `data/mfd_full/` |
| **Ridge** (paper's config 17) | Ridge regression, penalty α = 0.01 | Different characteristic subsets only (deterministic) | `data/mfd_ridge/` |
| **Single ridge forecast** | One ridge on all 153 characteristics, same window | None (a control, not an MFD) | `data/ridge/` |

## The paper's 12 controls (Fama-MacBeth regressions)

| Label | JKP variable | Meaning |
|---|---|---|
| **BETA** | `beta_60m` | Market beta, 60 monthly returns |
| **SIZE** | `market_equity` | Market capitalisation |
| **BM** | `be_me` | Book equity / market equity |
| **MOM** | `ret_12_1` | Momentum: return from month t−12 to t−2 |
| **STR** | `ret_1_0` | Short-term reversal: return in month t |
| **AG** | `at_gr1` | Asset growth, 1 year |
| **OP** | `ope_be` | Operating profit / book equity |
| **SUE** | `niq_su` | Standardised unexpected (quarterly) earnings |
| **ILLIQ** | `ami_126d` | Amihud illiquidity, 126 days |
| **IVOL** | `ivol_capm_252d` | Idiosyncratic volatility vs CAPM, 252 days |
| **MAX** | `rmax5_21d` | Average of the 5 highest daily returns in the month |
| **TURN** | `turnover_126d` | Share turnover, 126 days |

Paper's three specifications (Table 6): **(1)** MFD + BETA, SIZE, BM, MOM;
**(2)** + AG, OP; **(3)** all 12 controls. "+ mean" adds the mean forecast to each.

Regression conventions: monthly cross-sectional OLS; regressors winsorised at 0.5/99.5%
(pooled) and z-scored within each month, so slopes are **% per month per 1 SD**;
dependent variable = next-month excess return in %; Newey-West t-statistics.

## Factor models (portfolio alphas)

Alpha = intercept of the portfolio's monthly excess return regressed on the factors;
factor month aligned with the holding month; Newey-West (6 lags) t-statistics.

| Label | Model | Factors | Data window |
|---|---|---|---|
| **CAPM** | Capital asset pricing model | Market excess return | full |
| **FF6** | Fama-French 5 + momentum | Market, size (SMB), value (HML), profitability (RMW), investment (CMA), momentum (UMD) | full (Ken French library) |
| **q-factor** | Hou-Xue-Zhang q⁵ model | Market, size (ME), investment (IA), profitability (ROE), expected growth (EG) | to Dec 2021 |
| **SY** | Stambaugh-Yuan mispricing model | Market, size, management (MGMT), performance (PERF) | to Dec 2016 |
| **DHS** | Daniel-Hirshleifer-Sun behavioural model | Market, post-earnings drift (PEAD), financing (FIN) | to Dec 2022 |

## Portfolio sorts

| Term | Definition |
|---|---|
| **Decile sort** | Each month, stocks ranked on MFD into 10 groups; next-month return of each |
| **H-L** | High-minus-low: decile 10 (highest MFD) minus decile 1 |
| **EW / VW** | Equal-weighted / value-weighted (by market equity) portfolio returns |
| **Within mean-forecast quintiles** | Dependent 5×10 sort: stocks first split into mean-forecast quintiles, then into MFD deciles within each; decile returns averaged across quintiles. Compares stocks with similar expected returns |

## Samples

| Name | Period |
|---|---|
| Paper period | Aug 1976 – Nov 2022 (formation months) |
| Out of sample | Dec 2022 – Nov 2025 |
| Shorting-constraint tests | 2006 – 2025 (short interest from Compustat) |
