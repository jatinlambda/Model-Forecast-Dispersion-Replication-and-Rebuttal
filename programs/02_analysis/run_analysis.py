"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Runs all analyses files
"""

from pathlib import Path
import subprocess
import sys

# project root = two levels above this file
ROOT = Path(__file__).resolve().parents[2]
PREP_DIR = ROOT / "programs" / "02_analysis"

scripts = [
    # provide summary statistics
    "summary_stats.py",
    
    # aggregate mfd dynamics
    "analysis_aggregate_mfd_movement.py",
    
    # performing portfolio sort analyses
    "analysis_portfolio_sorts.py",
    
    # study drivers of aggregate MFD and AFD
    "analysis_aggregate_drivers_mfd_afd.py",
    
    # portfolio sort results for different parametrizations of MFD
    "analysis_different_parametrizations.py",
    
    # relation of MFD to established disagreement proxies
    "analysis_disagreement_proxies.py",
    
    # drivers of stock-level MFD and AFD
    "analysis_drivers_mfd_afd.py",
    
    # MFD pricing around earnings annoucement days
    "analysis_earnings_announcement_days.py",
    
    # analysis of earnings-based MFD
    "analysis_eps.py",
    
    # Fama MacBeth regressions
    "analysis_fama_macbeth.py",
    
    # MFD Fama MacBeth results for international stocks
    "analysis_international.py",
    
    # Bivariate portfolio sorts
    "analysis_bivariate_sorts.py",
    
    # MFD's predictive power for contemp. and future volatility, turnover
    "analysis_predicting_suv_turnover_hv.py",
    
    # Pricing of MFD long-short portfolio by non-linear SDFs
    "analysis_sdf.py",
    
    # Disagreement across and within information set and model divergence
    "analysis_within_across_models.py",
    
    # 3-pass Fama MacBeth regression of Gigliu and Xiu (2021)
    "analysis_gx_pca.py"
    ]


for s in scripts:
    script_path = PREP_DIR / s
    print(f"Running {script_path.relative_to(ROOT)}")

    subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        cwd=ROOT
    )


