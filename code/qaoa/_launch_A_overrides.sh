#!/bin/bash
# DATASET A, second gap: the two OVERRIDE cells carry no circuit_params.
#
# _launch_A.sh backfilled seed 442 only. But dataset A does not use seed 442 at (9,2) or (12,3) --
# it uses the AR-targeted overrides 31337 and 7, and those multi-start records predate the
# circuit_params/QUBO change, so they are unusable for hardware compilation. 20 runs, BOTH arms
# (the notebook/figure override block substitutes both).
#
# Separate pools because the seed differs per cell and test_multistart_cell.py applies one SEEDS
# list to every cell in CELLS.
#
# These should reproduce their existing AR exactly: unlike the seed-442 backfill (whose sweep
# records came from _span_expanded under the OLD stopping rule), these files were written by
# test_multistart_cell.py itself at patience=30, so the code path is identical. Verified after.
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs_A

for W in $(seq 0 2); do
  CELLS="9:2" SEEDS=31337 ARMS=star,L1 WORKER=$W NWORKERS=3 \
    /home/edo/anaconda3/envs/qubo_env/bin/python test_multistart_cell.py \
    > logs_A/ov_n9p2_w$W.log 2>&1 &
done
for W in $(seq 0 2); do
  CELLS="12:3" SEEDS=7 ARMS=star,L1 WORKER=$W NWORKERS=3 \
    /home/edo/anaconda3/envs/qubo_env/bin/python test_multistart_cell.py \
    > logs_A/ov_n12p3_w$W.log 2>&1 &
done
wait
echo "DATASET A OVERRIDE BACKFILL DONE"
