#!/bin/bash
# Auto-handoff for dataset B so cores are never left idle between stages:
#
#   1. wait for n=15 p=3 (dataset A's seed pick, ~16 cores) to finish
#   2. start B workers 6-19  -- wave 1 (workers 0-5) is already running; the shard is 20-way, so
#      the two waves partition the configs and nothing is computed twice
#   3. wait for ALL of n<=12 B to be complete, then start n=15 p=1/p=2
#
# Polls on completed-with-parameters counts rather than on process liveness, so a crashed worker
# cannot make this advance early.
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
PY=/home/edo/anaconda3/envs/qubo_env/bin/python

count() {   # count(cells...) -> runs carrying circuit_params
  $PY - "$@" <<'EOF'
import sys, os, pickle
V=(42,142,242,342,442); S=(442,7,2024,31337); A=("star","L1")
n_ok=0
for spec in sys.argv[1:]:
    n,p=(int(x) for x in spec.split(":"))
    for v in V:
        for qs in S:
            for mc in A:
                f=f"data_qaoa/_multistart_n{n}p{p}/{mc}_n{n}_p{p}_v{v}_s{qs}.pkl"
                try:
                    n_ok += os.path.exists(f) and pickle.load(open(f,"rb")).get("circuit_params") is not None
                except Exception:
                    pass
print(n_ok)
EOF
}

echo "[chain] waiting for n=15 p=3 to finish (need 40)"
until [ "$(count 15:3)" -ge 40 ]; do sleep 120; done
echo "[chain] n=15 p=3 complete at $(date '+%H:%M') -- starting B workers 6-19"
bash _launch_B.sh 6 19

CELLS_B="6:1 6:2 6:3 9:1 9:2 9:3 12:1 12:2 12:3"
echo "[chain] waiting for all of n<=12 B (need 360)"
until [ "$(count $CELLS_B)" -ge 360 ]; do sleep 120; done
echo "[chain] n<=12 dataset B complete at $(date '+%H:%M') -- starting n=15 p=1/p=2"
bash _launch_B15.sh
echo "[chain] DATASET B FULLY COMPLETE at $(date '+%H:%M')"
