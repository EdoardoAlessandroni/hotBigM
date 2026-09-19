#!/bin/bash
# DATASET B -- multi-start: 4 inits per (cell, vseed, arm), selected afterwards by LOWEST TRAINED
# COST. Dataset A (one init, seed 442) is a subset, so B measures what multi-start buys.
#
#   ./_launch_B.sh <first_worker> <last_worker>
#
# The shard is ALWAYS 20-way regardless of how many workers are started, so waves can be added as
# cores free up without any worker recomputing another's configs:
#   ./_launch_B.sh 0 5      while n=15 p=3 still occupies ~14 cores
#   ./_launch_B.sh 6 19     once it finishes
#
# Cells are ordered cheap-first so partial results appear early rather than after the n=12 tail.
# n=15 is deliberately NOT here: p=3 is already running under _launch_A.sh, and p=1/p=2 come after
# n<=12 finishes (see _launch_B15.sh).
#
# Anything already carrying circuit_params is skipped, so re-running this is safe and idempotent.
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs_B

W0=${1:-0}
W1=${2:-19}
CELLS_B="6:1,9:1,6:2,6:3,9:2,12:1,9:3,12:2,12:3"

for W in $(seq $W0 $W1); do
  CELLS="$CELLS_B" WORKER=$W NWORKERS=20 \
    /home/edo/anaconda3/envs/qubo_env/bin/python test_multistart_cell.py \
    > logs_B/w$W.log 2>&1 &
done
wait
echo "DATASET B WORKERS $W0-$W1 DONE"
