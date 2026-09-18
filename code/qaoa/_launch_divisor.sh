#!/bin/bash
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
mkdir -p logs_divisor
# pin BLAS threads: the box was at load 33 on 20 cores, which is thread oversubscription,
# not useful parallelism -- each worker is already a whole process
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
NW=6
for W in $(seq 0 $((NW-1))); do
  WORKER=$W NWORKERS=$NW /home/edo/anaconda3/envs/qubo_env/bin/python \
      test_span_divisor.py > logs_divisor/w$W.log 2>&1 &
done
wait
echo "SPAN DIVISOR TEST DONE"
