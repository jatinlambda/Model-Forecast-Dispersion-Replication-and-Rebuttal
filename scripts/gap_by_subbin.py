"""Per-sub-bin mean-forecast gap (MFD Q5 - Q1, bp/month) inside mean quintile 1,
using exactly the sorting rules of scripts/test_within_q1.py. Writes
results/within_q1/gap_by_subbin.csv."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("wq", str(ROOT / "scripts/test_within_q1.py"))
wq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wq)
tm = wq.tm

p = tm.load_panel(ROOT / "data/mfd_full/mfd_quantiles.pq")
p = p[p.month <= tm.PAPER_END].dropna(subset=["mean", "std", "ret_exc_lead1m", "me"])
p["mq_all"] = p.groupby("month")["mean"].transform(lambda s: wq.qrank(s, 5))
q1 = p[p.mq_all == 1].copy()

rows = []
for K in (1, 2, 4, 5):
    d = q1.copy()
    d["sub"] = d.groupby("month")["mean"].transform(lambda s: wq.qrank(s, K)) if K > 1 else 1
    d["mq"] = d.groupby(["month", "sub"])["std"].transform(lambda s: wq.qrank(s, 5))
    mg = d.groupby(["month", "sub", "mq"])["mean"].mean().unstack("mq")
    gap = ((mg[5] - mg[1]) * 1e4).groupby("sub").mean()   # time-average per sub-bin
    avg = ((mg[5] - mg[1]) * 1e4).groupby("month").mean().mean()  # matches finer_sorts.csv
    for s, g in gap.items():
        rows.append(dict(K=K, sub=int(s), gap_bp=g, avg_gap_bp=avg))
    print(K, [round(v, 1) for v in gap.values], "avg", round(avg, 1))

out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/within_q1/gap_by_subbin.csv", index=False)
