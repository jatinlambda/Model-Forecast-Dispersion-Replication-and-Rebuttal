# Authors' replication code

This folder holds the replication code for Bali, Kelly, Mörke and Rahman, *Machine Forecast Disagreement*, from their public package on Harvard Dataverse (doi:10.7910/DVN/BO8SQ8). It is kept here so the MFD construction is the same one the paper used.

| Folder | Contents |
|---|---|
| `00_preparation/` | Data downloads (CRSP, IBES links, short interest, earnings dates) and characteristic processing; entry point is `run_preparation.py` |
| `01_run_mfd/` | MFD construction: `construct_mfd.py` trains the 100 random-forest investors; entry point is `run_mfd_construction.py` |
| `02_analysis/` | The paper's tables: portfolio sorts, Fama-MacBeth, SDF and PCA tests; entry point is `run_analysis.py` |

My own tests are in [`../scripts/`](../scripts/) and reuse settings from `02_analysis/helper.py`.
