#!/bin/bash
# DATASET B, n=15 -- depths 1 and 2 only. p=3 is already covered: _launch_A.sh ran all four inits
# at (15,3) to choose dataset A's seed, and those runs are protocol-identical to B, so they ARE
# B's (15,3) data. Nothing is recomputed.
#
# Runs last because it is the most expensive block (~37 CPU-h estimated, extrapolated from n=12 --
# this is the least certain number in the dataset B schedule).
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs_B15

for W in $(seq 0 19); do
  CELLS="15:1,15:2" WORKER=$W NWORKERS=20 \
    /home/edo/anaconda3/envs/qubo_env/bin/python test_multistart_cell.py \
    > logs_B15/w$W.log 2>&1 &
done
wait
echo "DATASET B n=15 DONE"
