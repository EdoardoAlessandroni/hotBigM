#!/bin/bash
# DATASET A completion, in priority order, one pool so the cores stay saturated:
#   15:3  all 4 inits, both arms  -> the multi-start pick for the last unresolved cell (40 runs)
#   6:3, 9:3, 12:3  seed 442, M* only -> backfill circuit_params for the runs imported from
#                   _span_expanded, which never stored them (15 runs)
# Cheap cells first so the backfill lands early; n=15 dominates the tail either way.
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs_A

for W in $(seq 0 5); do
  CELLS="6:3,9:3,12:3" SEEDS=442 ARMS=star WORKER=$W NWORKERS=6 \
    /home/edo/anaconda3/envs/qubo_env/bin/python test_multistart_cell.py \
    > logs_A/backfill_w$W.log 2>&1 &
done
for W in $(seq 0 13); do
  CELLS="15:3" WORKER=$W NWORKERS=14 \
    /home/edo/anaconda3/envs/qubo_env/bin/python test_multistart_cell.py \
    > logs_A/n15p3_w$W.log 2>&1 &
done
wait
echo "DATASET A JOBS DONE"
