#!/usr/bin/env bash
# Start shard 4 (2004-08..2011-07) once shard 7 (2025-08..end) has exited, to stay under 32 GB.
cd "$(dirname "$0")/.."
while ps -eo args | grep -v grep | grep -q "construct_mfd.py.*--test-start 2025-08"; do sleep 10; done
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
conda run --no-capture-output -n mfd python -I -u programs/01_run_mfd/construct_mfd.py \
  --n-investors 100 --n-estimators 2000 --n-jobs 4 --out data/mfd_full \
  --test-start 2004-08 --test-end 2011-07 > logs/mfd_shard_4.log 2>&1
echo "SHARD4 EXIT $?"
