#!/bin/bash
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
mkdir -p logs_refix
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for V in 342 442; do
  /home/edo/anaconda3/envs/qubo_env/bin/python _refix_n9p2.py $V star > logs_refix/star_v$V.log 2>&1 &
done
wait
echo "REFIX n9p2 DONE"
