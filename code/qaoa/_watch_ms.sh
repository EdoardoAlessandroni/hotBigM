#!/bin/bash
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
D=data_qaoa/_multistart_n12p3
until [ "$(ls -1 $D 2>/dev/null | wc -l)" -ge 14 ] \
   || [ "$(ps -eo args | grep -c '[t]est_multistart_n12p3.py')" -eq 0 ]; do sleep 120; done
echo "first wave in: $(ls -1 $D 2>/dev/null | wc -l)/40 files, $(ps -eo args | grep -c '[t]est_multistart_n12p3.py') workers alive"
