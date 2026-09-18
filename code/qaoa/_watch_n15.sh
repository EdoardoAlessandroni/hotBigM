#!/bin/bash
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
S=data_qaoa/PO_optimality_maqaoa_percentile_interp_M00
L=data_qaoa/PO_optimality_maqaoa_percentile_interp_L1_M00
until { [ "$(ls -1 $S | grep -c 'bits_15')" -ge 15 ] && [ "$(ls -1 $L | grep -c 'bits_15')" -ge 15 ]; } \
   || [ "$(ps -eo args | grep -c '[r]un_PO_optimality_interp_M00.py')" -eq 0 ]; do sleep 180; done
echo "n=15 now: $(ls -1 $S | grep -c 'bits_15')/15 star, $(ls -1 $L | grep -c 'bits_15')/15 L1; workers alive: $(ps -eo args | grep -c '[r]un_PO_optimality_interp_M00.py')"
