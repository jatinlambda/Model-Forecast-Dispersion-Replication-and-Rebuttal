"""
Replication code for:
Machine Forecast Disagreement

Authors:
Turan G. Bali, Bryan T. Kelly, Mathis Moerke, and Jamil Rahman

Purpose:
Constructs main variable of interest, machine forecast disagreement, and 
runs a small post-process script
"""

from pathlib import Path
import subprocess
import sys

# project root = two levels above this file
ROOT = Path(__file__).resolve().parents[2]
PREP_DIR = ROOT / "programs" / "01_run_mfd"

scripts = [
    # construct machine forecast disagreement measure
    "construct_mfd.py",
    
    # adds next month's excess returns and market cap
    "post_process_mfd_data.py",
    
    # generate test assets and factors for 3-pass PCA Fama-Macbeth regression (following Giglio and Xiu (2021))
    "gx_prep_data.py",
    "gx_prep_data_aptree.py",
    
    ]

for s in scripts:
    script_path = PREP_DIR / s
    print(f"Running {script_path.relative_to(ROOT)}")

    subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        cwd=ROOT
    )