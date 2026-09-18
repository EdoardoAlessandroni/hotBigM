#!/bin/bash
# Dataset B: 4 fresh inits x 5 vseeds x both arms, every cell at n <= 12.
# Cells not yet run come FIRST so dataset B becomes plottable sooner; the three already-measured
# cells come last, where the only missing piece is circuit_params (the script recomputes exactly
# those and leaves complete files alone).
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs_B
CELLS="6:1,6:2,9:1,9:3,12:1,12:2,6:3,9:2,12:3"
for W in $(seq 0 13); do
  CELLS=$CELLS WORKER=$W NWORKERS=14 /home/edo/anaconda3/envs/qubo_env/bin/python \
      test_multistart_cell.py > logs_B/w$W.log 2>&1 &
done
wait
echo "DATASET B (n<=12) DONE: $(ls -1 data_qaoa/_multistart_n*p*/ 2>/dev/null | grep -c pkl) pkl files"
