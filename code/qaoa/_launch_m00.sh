#!/bin/bash
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
mkdir -p logs_m00
NW=10
for W in $(seq 0 $((NW-1))); do
  WORKER=$W NWORKERS=$NW /home/edo/anaconda3/envs/qubo_env/bin/python \
      run_PO_optimality_interp_M00.py > logs_m00/w$W.log 2>&1 &
  echo "launched worker $W pid $!"
done
wait
echo "M00 GRID DONE"
