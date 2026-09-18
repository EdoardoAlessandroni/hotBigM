#!/bin/bash
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
mkdir -p logs_fill
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# exactly the three configs that are QUEUED (not in flight) behind busy workers
for SPEC in "L1:12:3:442" "L1:9:3:442" "L1:15:3:242"; do
  TAG=$(echo $SPEC | tr ':' '_')
  ONLY=$SPEC /home/edo/anaconda3/envs/qubo_env/bin/python \
      run_PO_optimality_interp_M00.py > logs_fill/$TAG.log 2>&1 &
done
wait
echo "FILL RUNS DONE"
