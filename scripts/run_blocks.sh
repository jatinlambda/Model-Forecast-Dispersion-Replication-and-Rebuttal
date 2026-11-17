#!/usr/bin/env bash
# Full-sample MFD construction: one short-lived process per refit block.
#
# Why per block: memory in a long-running process grows by ~1 GB per refit
# (not returned by the RF library / allocator), so multi-block processes get
# OOM-killed under a 32 GB Slurm limit. A process per block loads only the
# 132 months it needs, fits 100 forests, predicts 12 months and exits.
#
# Results are identical to a single sequential run: each forest's seed depends
# only on (seed, investor, block start month), and blocks are Aug..Jul as in
# the paper. Blocks whose 12 monthly files already exist are skipped, so this
# script is safe to rerun after any interruption.
#
# Usage: scripts/run_blocks.sh [WORKERS] [THREADS_PER_WORKER]   (default 4 x 8)
set -uo pipefail
cd "$(dirname "$0")/.."

WORKERS=${1:-4}   # ~5.5 GB per worker on 2000s cross-sections; 4 fit in 32 GB
THREADS=${2:-8}
FIRST_YEAR=1976
LAST_YEAR=2025
OUT=data/mfd_full
mkdir -p logs/blocks "$OUT"
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS

# list blocks that still have missing months (last block may be partial: Aug..Nov 2025)
todo=()
for ((y = FIRST_YEAR; y <= LAST_YEAR; y++)); do
  n=0
  for m in 08 09 10 11 12; do [ -f "$OUT/month=${y}${m}_disp.pq" ] && n=$((n+1)); done
  for m in 01 02 03 04 05 06 07; do [ -f "$OUT/month=$((y+1))${m}_disp.pq" ] && n=$((n+1)); done
  expected=12; (( y == LAST_YEAR )) && expected=4
  (( n < expected )) && todo+=("$y")
done
echo "$(date +%H:%M:%S) ${#todo[@]} blocks to run with $WORKERS workers x $THREADS threads: ${todo[*]}"

run_block() {
  y=$1
  log=logs/blocks/block_$y.log
  end="--test-end $((y+1))-07"; (( y == LAST_YEAR )) && end=""
  conda run --no-capture-output -n mfd python -I -u programs/01_run_mfd/construct_mfd.py \
    --n-investors 100 --n-estimators 2000 --n-jobs "$THREADS" --out "$OUT" \
    --test-start "${y}-08" $end > "$log" 2>&1
  rc=$?
  echo "$(date +%H:%M:%S) block $y exit $rc"
  return $rc
}
export -f run_block
export THREADS OUT LAST_YEAR

printf "%s\n" "${todo[@]}" | xargs -P "$WORKERS" -I{} bash -c 'run_block {}'
echo "$(date +%H:%M:%S) blocks finished; months on disk: $(ls $OUT/month=*_disp.pq | wc -l) of 592"

# aggregate all months into mfd.csv / mfd_quantiles.pq (no fitting: all blocks done)
conda run --no-capture-output -n mfd python -I -u programs/01_run_mfd/construct_mfd.py \
  --n-investors 100 --n-estimators 2000 --n-jobs 1 --out "$OUT" --test-start "${FIRST_YEAR}-08" \
  > logs/mfd_aggregate.log 2>&1
grep -E "Saved|fitting|Traceback" logs/mfd_aggregate.log | tail -2
echo FINISHED
