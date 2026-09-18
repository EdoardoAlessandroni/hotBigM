#!/bin/bash
# exit as soon as every n<=12 p=2 cell exists in both arms (30 files), or the workers are gone
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
A=data_qaoa/PO_optimality_maqaoa_percentile_interp_M00
B=data_qaoa/PO_optimality_maqaoa_percentile_interp_L1_M00
while true; do
  N=$(ls $A/*_layers_2_* $B/*_layers_2_* 2>/dev/null | grep -cE "bits_(6|9|12)_")
  ALIVE=$(ps -eo args | grep -c "[r]un_PO_optimality_interp_M00")
  if [ "$N" -ge 30 ] || [ "$ALIVE" -eq 0 ]; then
    echo "p=2 watcher firing: $N/30 n<=12 p=2 files present, $ALIVE workers alive"
    exit 0
  fi
  sleep 120
done
