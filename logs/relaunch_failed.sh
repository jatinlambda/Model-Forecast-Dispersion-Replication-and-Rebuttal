#!/usr/bin/env bash
# Relaunch the two OOM-killed shards one at a time (32 GB job limit).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
run() {  # $1 shard id, $2 start, $3 end
  conda run --no-capture-output -n mfd python -I -u programs/01_run_mfd/construct_mfd.py \
    --n-investors 100 --n-estimators 2000 --n-jobs 4 --out data/mfd_full \
    --test-start "$2" --test-end "$3" > "logs/mfd_shard_$1.log" 2>&1 &
  # wait until data is loaded (first 'fitting block' line) before the next load
  until grep -qE "fitting block|Traceback|Killed" "logs/mfd_shard_$1.log" 2>/dev/null; do sleep 5; done
}
run 0 1976-08 1983-07
run 4 2004-08 2011-07
wait
echo RELAUNCH DONE
