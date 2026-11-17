#!/usr/bin/env bash
# After the 5 running shards exit, finish the remaining blocks with 4 x 8 threads,
# launched one at a time (32 GB limit). Done blocks are skipped automatically.
cd "$(dirname "$0")/.."
while ps -eo args | grep -v grep | grep -q "python -I -u programs/01_run_mfd/construct_mfd.py"; do sleep 15; done
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
i=0
for range in "2004-08 2007-07" "2007-08 2010-07" "2010-08 2014-07" "2014-08 2018-07"; do
  set -- $range
  log=logs/mfd_finish_$i.log
  conda run --no-capture-output -n mfd python -I -u programs/01_run_mfd/construct_mfd.py \
    --n-investors 100 --n-estimators 2000 --n-jobs 8 --out data/mfd_full \
    --test-start "$1" --test-end "$2" > "$log" 2>&1 &
  until grep -qE "fitting block|Traceback|Killed" "$log" 2>/dev/null; do sleep 5; done
  i=$((i+1))
done
wait
echo "REMAINING DONE: $(ls data/mfd_full/month=*_disp.pq | wc -l) of 592 months"
