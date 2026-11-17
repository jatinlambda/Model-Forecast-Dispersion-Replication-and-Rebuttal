#!/usr/bin/env bash
# Full-sample MFD construction, split across parallel processes.
#
# Each refit block (12 forecast months, Aug..Jul) is independent: it trains on
# the 120 months before it, and each forest's seed depends only on
# (seed, investor, block start month). So contiguous year ranges can run as
# separate processes and give exactly the same numbers as one sequential run.
# Each shard writes month=YYYYMM_*.pq files into the same folder; a final pass
# (all blocks already done, so it only aggregates) writes mfd.csv.
#
# Usage: scripts/run_full_mfd.sh [N_SHARDS] [THREADS_PER_SHARD]
#   default 8 shards x 4 threads = 32 cores
set -euo pipefail
cd "$(dirname "$0")/.."

SHARDS=${1:-8}
THREADS=${2:-4}
FIRST_YEAR=1976        # first block starts Aug 1976 (paper's first forecast month)
LAST_YEAR=2025         # last block starts Aug 2025, runs to last month with a lead return
OUT=data/mfd_full
COMMON="--n-investors 100 --n-estimators 2000 --n-jobs $THREADS --out $OUT"
mkdir -p logs "$OUT"

# avoid oversubscription from BLAS/OpenMP inside each process
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS

NBLOCKS=$((LAST_YEAR - FIRST_YEAR + 1))
PER=$(( (NBLOCKS + SHARDS - 1) / SHARDS ))
echo "$NBLOCKS blocks, $SHARDS shards x $THREADS threads, up to $PER blocks each"

pids=()
for ((s = 0; s < SHARDS; s++)); do
  y0=$((FIRST_YEAR + s * PER))
  y1=$((y0 + PER))                       # shard ends Jul of y1
  (( y0 > LAST_YEAR )) && break
  if (( y1 > LAST_YEAR )); then END_ARG=""; else END_ARG="--test-end ${y1}-07"; fi
  echo "shard $s: ${y0}-08 .. ${END_ARG:-end of data}"
  conda run --no-capture-output -n mfd python -I -u programs/01_run_mfd/construct_mfd.py \
      $COMMON --test-start "${y0}-08" $END_ARG > "logs/mfd_shard_$s.log" 2>&1 &
  pids+=($!)
  # Stagger: loading the data peaks at ~5-8 GB per process; with a 32 GB job
  # limit, 8 simultaneous loads get OOM-killed. Wait until this shard has
  # loaded (first 'fitting block' / 'skipping' line) before starting the next.
  until grep -qE "fitting block|skipping|Traceback|Killed" "logs/mfd_shard_$s.log" 2>/dev/null; do sleep 5; done
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if (( fail )); then echo "a shard failed; see logs/mfd_shard_*.log (rerun this script to resume)"; exit 1; fi

echo "all shards done; aggregating"
conda run --no-capture-output -n mfd python -I -u programs/01_run_mfd/construct_mfd.py \
    $COMMON --n-jobs 1 --test-start "${FIRST_YEAR}-08" > logs/mfd_aggregate.log 2>&1
tail -1 logs/mfd_aggregate.log
echo FINISHED
