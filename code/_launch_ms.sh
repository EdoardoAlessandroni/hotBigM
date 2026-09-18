#!/bin/bash
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs_ms
for W in $(seq 0 13); do
  WORKER=$W NWORKERS=14 /home/edo/anaconda3/envs/qubo_env/bin/python \
      test_multistart_n12p3.py > logs_ms/w$W.log 2>&1 &
done
wait
echo "MULTISTART n12p3 DONE: $(ls -1 data_qaoa/_multistart_n12p3/ | wc -l)/40 files"
