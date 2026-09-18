#!/bin/bash
# Wait for the percentile sweep to exit, then repair collapsed cells by multi-restart
# with lowest-final-cost selection.
while kill -0 438727 2>/dev/null; do sleep 20; done
echo "=== sweep finished, starting repair pass ==="
exec /home/edo/anaconda3/envs/qubo_env/bin/python -u repair_collapses.py
