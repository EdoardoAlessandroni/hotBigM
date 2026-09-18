#!/bin/bash
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
D=data_qaoa/PO_optimality_maqaoa_percentile_interp_L1_M00
until [ "$(ls -1 $D | grep -c 'bits_15.*layers_3')" -ge 3 ] || \
      [ "$(pgrep -fc run_PO_optimality_interp_M00)" -eq 0 ]; do sleep 120; done
echo "n=15 p=3 L1: $(ls -1 $D | grep -c 'bits_15.*layers_3')/3 files, $(pgrep -fc run_PO_optimality_interp_M00) workers alive"
