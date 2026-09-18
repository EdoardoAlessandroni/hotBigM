#!/bin/bash
# fire when M_L1 p=3 is complete for n<=12 (15 files; the M* side is already in via the import),
# or when the workers are gone
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
B=data_qaoa/PO_optimality_maqaoa_percentile_interp_L1_M00
while true; do
  N=$(ls $B/*_layers_3_* 2>/dev/null | grep -cE "bits_(6|9|12)_")
  ALIVE=$(ps -eo args | grep -c "[r]un_PO_optimality_interp_M00")
  if [ "$N" -ge 15 ] || [ "$ALIVE" -eq 0 ]; then
    echo "p=3 watcher firing: $N/15 M_L1 p=3 files for n<=12, $ALIVE workers alive"
    exit 0
  fi
  sleep 180
done
