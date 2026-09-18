#!/bin/bash
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
mkdir -p logs_ladder
NW=5
for W in $(seq 0 $((NW-1))); do
  WORKER=$W NWORKERS=$NW /home/edo/anaconda3/envs/qubo_env/bin/python \
      run_depth_ladder.py > logs_ladder/w$W.log 2>&1 &
done
wait
echo "DEPTH LADDER DONE"
