"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Runs all files for processing raw data for later use (i) to construct
machine forecast disagreement, and (ii) in analyses
"""


from pathlib import Path
import subprocess
import sys

# project root = two levels above this file
ROOT = Path(__file__).resolve().parents[2]
PREP_DIR = ROOT / "programs" / "00_preparation"

scripts = [
    # extracts monthly excess returns and stock characteristics; also 
    # rank-normalizes stock characteristics
    "process_stock_characteristics_returns.py",
    
    # computes aggregate disagreement proxies
    "compute_aggregate_disagreement_proxies.py",
    
    # compute stock-level standardised unexpected volume
    "compute_monthly_suv.py",
    
    # compute stock-level monthly turnover
    "compute_monthly_turnover.py",
    
    # compute the number of non-zero slope coefficients rolling over time
    "compute_sparsity_in_entire_crosssection.py",

    ]

for s in scripts:
    script_path = PREP_DIR / s
    print(f"Running {script_path.relative_to(ROOT)}")

    subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        cwd=ROOT
    )
