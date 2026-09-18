#!/bin/bash
# Top up n=15 from 3 to 5 vseeds: 12 runs = {star,L1} x p in {1,2,3} x vseed in {342,442}.
# One process per config via ONLY mode, so nothing can collide with the two grid workers still
# finishing n=15 p=3 L1 at v=42/v=142. p=3 launched first -- it is the long pole (~1.5-3 h).
# BLAS threads pinned: without this the box hit load 33 on 20 cores from thread contention and
# every run got SLOWER.
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs_n15

for P in 3 2 1; do
  for V in 342 442; do
    for ARM in star L1; do
      SPEC="$ARM:15:$P:$V"
      TAG="${ARM}_15_${P}_${V}"
      ONLY=$SPEC /home/edo/anaconda3/envs/qubo_env/bin/python \
          run_PO_optimality_interp_M00.py > logs_n15/$TAG.log 2>&1 &
    done
  done
done
wait
echo "N15 TOPUP DONE: $(ls -1 data_qaoa/PO_optimality_maqaoa_percentile_interp_M00/ | grep -c 'bits_15') star + $(ls -1 data_qaoa/PO_optimality_maqaoa_percentile_interp_L1_M00/ | grep -c 'bits_15') L1 files at n=15"
