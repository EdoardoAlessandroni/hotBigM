#!/bin/bash
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs_probe
for W in 0 1 2 3 4 5 6 7 8 9; do
  WORKER=$W NWORKERS=10 /home/edo/anaconda3/envs/qubo_env/bin/python \
      probe_n9p2.py > logs_probe/w$W.log 2>&1 &
done
wait
echo "PROBE DONE: $(ls -1 data_qaoa/_probe_n9p2/ | wc -l)/40 files"
