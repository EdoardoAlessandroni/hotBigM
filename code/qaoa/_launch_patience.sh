#!/bin/bash
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
mkdir -p logs_patience
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for PAT in 30 100; do
  /home/edo/anaconda3/envs/qubo_env/bin/python _test_patience.py $PAT 342 \
      > logs_patience/pat${PAT}_v342.log 2>&1 &
done
wait
echo "PATIENCE TEST DONE"
