# Disagreement or Low Forecasts? A replication and critique of *Machine Forecast Disagreement*

This repo rebuilds the Machine Forecast Disagreement (MFD) signal from Bali, Kelly, Mörke and Rahman, *Machine Forecast Disagreement* (Review of Financial Studies, forthcoming), at the paper's full scale, then tests whether it really measures investor disagreement.

**Slides:** [`presentation/slides.pdf`](presentation/slides.pdf) (20 slides). Speaker notes are in [`presentation/speaker_notes.md`](presentation/speaker_notes.md).

## The claim being tested

The paper trains 100 machine-learning "investors". Each one is a random forest that sees a random 76 of the 153 Jensen-Kelly-Pedersen stock characteristics. Every month, each investor forecasts each stock's next-month return. MFD is the standard deviation of a stock's 100 forecasts. Stocks with the highest MFD underperform stocks with the lowest MFD by about 1% a month. The paper reads this as Miller (1977) overpricing: when shorting is costly, prices reflect the optimists.

## Findings

The same 100 forecasts also give a **mean forecast**, which is the forests' own expected return. MFD and the mean forecast have a cross-sectional correlation of about −0.5. None of the paper's controls is the mean forecast.

| Test | Result |
|---|---|
| **1. Rebuild** (gate) | The spread reproduces. High-minus-low MFD earns −1.02% a month value-weighted (t −4.0), against −1.09% from the authors' published file. Six-factor alphas are −0.61 and −0.60. |
| **2. Key regression** | With the paper's 12 controls, MFD's Fama-MacBeth slope is −0.196 (t −5.47). Adding the mean forecast takes it to −0.006 (t −0.13). On the authors' own MFD file the slope goes from −0.23 to −0.02 (t −0.40). |
| **3. Double sorts** | Within mean-forecast quintiles, the MFD high-minus-low alpha averages −0.06 (t −0.4). Within MFD quintiles, the mean-forecast spread averages +1.28 (t 7.9). |
| **4. Robustness** | Across five control sets, up to all 153 characteristics plus industry, MFD alone has t between −5.4 and −7.0. With the mean added, no t is significant. The slope is also near zero at every horizon from 1 to 12 months. |
| **5. Ridge investors** | Swapping the forests for ridge regressions on the same feature subsets gives the same pattern. Each model's MFD is absorbed by that model's own mean forecast. |
| **Miller's optimist test** | Splitting MFD into an optimist spread (Q80 − Q50) and a pessimist spread (Q50 − Q20), both sides matter about equally without the mean and neither matters with it. |

What survives is a corner of small, extreme stocks in the lowest mean-forecast quintile. There the effect is not significant value-weighted, it shrinks as the mean is held fixed more tightly, and standard controls absorb it. Even the most bullish forests expect less from these stocks than from a typical stock, so there are no optimists to overprice them. My reading is that MFD measures how hard a stock is to model, not how much investors disagree.

**Open question.** In hard-to-model stocks, model dispersion may still partly track real disagreement. Testing this needs a measure that is not built from the same characteristics, such as lending fees or analyst dispersion.

All results cover the paper's sample, August 1976 to November 2022, with an out-of-sample check from December 2022 to November 2025. Variable, control and factor-model definitions are in [`docs/definitions.md`](docs/definitions.md). The original research proposal is in [`docs/proposal.pdf`](docs/proposal.pdf).

## Repository layout

```
programs/        Authors' replication code (00_preparation, 01_run_mfd, 02_analysis); see programs/README.md
scripts/         My downloads, MFD runners and tests (one script per test)
results/         Output tables (CSV/TXT) and figures (PNG) for every test
presentation/    Slide deck: slides.pdf, slides.html (open in a browser, arrow keys to move), speaker notes
docs/            Definitions of every variable and model, and the original proposal
```

Main scripts, grouped by test:

| Step | Script | Output |
|---|---|---|
| Data | `download_jkp_usa.py`, `download_replication.py` | `data/raw/`, `data/replication_package/` |
| Build MFD | `run_full_mfd.sh`, `run_blocks.sh` (wrap `programs/01_run_mfd/construct_mfd.py`) | `data/mfd_full/` |
| Rebuild check | `test_miller.py` (step 0), `compare_mfd.py` (prints to console) | `results/miller/` |
| Key regression and Miller split | `test_miller.py`, `check_mean_forecast.py` (prints to console) | `results/miller/` |
| Double sorts | `double_sorts_mean_mfd.py`, `test_within_q1.py`, `gap_by_subbin.py` | `results/double_sorts/`, `results/within_q1/` |
| Control ladder and horizons | `test_controls.py`, `test_mfd_mean_controls.py`, `test_miller_step4.py` | `results/controls/`, `results/mfd_mean_controls/`, `results/miller/` |
| Ridge investors | `build_ridge_forecast.py`, `test_ridge_mfd.py` | `results/ridge_mfd/` |
| Paper-style tables | `paper_tables.py --variant forest\|ridge` | `results/paper_tables/` |
| Economic profile | `economic_profile.py`, `correlation_check.py`, `frac_up_fmb.py`, `test_structured.py` | `results/economic/`, `results/correlation_check/`, `results/frac_up/`, `results/structured/` |
| Figures | `make_figures.py`, `plot_results.py`, `plot_paper_overview.py` | `results/figures/` |

## Reproducing

**Data is not included.** JKP, CRSP and Compustat are licensed through WRDS and can't be redistributed. You need a WRDS account.

1. Create the environment. Versions match the authors' `uv.lock`.
   ```bash
   conda env create -f environment.yml
   conda run -n mfd pip install --no-deps wrds==3.5.0
   ```
2. Download the data:
   ```bash
   python scripts/download_jkp_usa.py        # JKP characteristics from WRDS -> data/raw/
   python scripts/download_replication.py    # authors' package from Harvard Dataverse (doi:10.7910/DVN/BO8SQ8)
   python programs/00_preparation/run_preparation.py
   ```
3. Build MFD with 100 investors × 2,000 trees. This is the expensive step: each yearly refit block is independent, and the runners split the blocks across processes.
   ```bash
   scripts/run_full_mfd.sh 8 4               # 8 shards × 4 threads
   ```
4. Run any test script from the project root, for example `python scripts/test_miller.py`. Each one writes to its own folder under `results/`.

## Credits

The MFD measure and the code in `programs/` are by Turan G. Bali, Bryan T. Kelly, Mathis Mörke and Jamil Rahman, from their public replication package (doi:10.7910/DVN/BO8SQ8). The extensions in `scripts/`, the results and the slides are my own work.

Jatin Lamba, MIT
