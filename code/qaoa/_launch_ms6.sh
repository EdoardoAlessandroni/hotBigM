#!/bin/bash
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs_ms6
for W in $(seq 0 13); do
  N_BITS=6 LAYERS=3 WORKER=$W NWORKERS=14 /home/edo/anaconda3/envs/qubo_env/bin/python \
      test_multistart_cell.py > logs_ms6/w$W.log 2>&1 &
done
wait
echo "MULTISTART n6p3 DONE: $(ls -1 data_qaoa/_multistart_n6p3/ | wc -l)/40 files"
