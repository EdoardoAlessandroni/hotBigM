#!/bin/bash
# DATASET A TOP-UP, ONE SIZE-BATCH AT A TIME.  Same 150 runs as _launch_A_v10.sh, but scoped so a
# batch of sizes can be finished and inspected (AR + eta_eff, both arms) before cores are committed
# to the next batch. Order requested by the user: {6,15} first, then {9,12}.
#
#   bash _launch_A_v10_size.sh 6,15
#   bash _launch_A_v10_size.sh 9,12
#
# Pool 1 -- the seed-442 sweep (BITS hook), both arms, into the PUBLISHED sweep folders, which is
# where make_fig_paired.py / export_dataset_A.py / the notebook's CELL_LOAD glob for their base
# table, so new instances are picked up with no downstream code change.
# Pool 2 -- only the override cells belonging to the requested sizes, at their own circuit init,
# into _multistart_n{N}p{P}/ where the vseed 42..442 overrides already live:
#     (9,2)@31337   (12,3)@7   (15,3)@2024
# n=6 has no override cell, so a "6,15" batch runs one override cell and a "9,12" batch runs two.
#
# Both runners skip completed work, so re-running after an interruption is safe.
set -u
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
PY=/home/edo/anaconda3/envs/qubo_env/bin/python
NEW_V="542,642,742,842,942"
SIZES="${1:?usage: bash _launch_A_v10_size.sh <comma-separated sizes, e.g. 6,15>}"
TAG="${SIZES//,/_}"
# NW1/NW2 size the two pools. The 14/6 default saturates a 20-core box from cold; pass smaller
# numbers to start a batch on the cores a previous batch has already freed, e.g.
#   NW1=7 NW2=3 bash _launch_A_v10_size.sh 9,12
# while an n=15 p=3 tail is still draining. Oversubscribing costs the in-flight long runs, so match
# NW1+NW2 to the ACTUAL idle core count (nproc minus the running worker count), not to nproc.
NW1="${NW1:-14}"
NW2="${NW2:-6}"

# One BLAS thread per process. Without this every worker spawns 20 threads and the box thrashes.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs_A10

echo "pool 1: seed-442 sweep, BITS=$SIZES x 3 depths x 5 new vseeds x 2 arms, $NW1 workers"
for W in $(seq 0 $((NW1 - 1))); do
  BITS="$SIZES" VSEEDS_EXTRA="$NEW_V" WORKER=$W NWORKERS=$NW1 \
    $PY run_PO_optimality_interp_M00.py > logs_A10/sweep_${TAG}_w$W.log 2>&1 &
done

# the override cells that belong to the sizes in this batch
OVR=""
case ",$SIZES," in *,9,*)  OVR="$OVR 9:2:31337" ;; esac
case ",$SIZES," in *,12,*) OVR="$OVR 12:3:7"    ;; esac
case ",$SIZES," in *,15,*) OVR="$OVR 15:3:2024" ;; esac

if [ -n "$OVR" ]; then
  echo "pool 2: override cells$OVR at their own init, $NW2 workers"
  for W in $(seq 0 $((NW2 - 1))); do
    (
      for spec in $OVR; do
        N=${spec%%:*}; rest=${spec#*:}; P=${rest%%:*}; S=${rest#*:}
        CELLS="$N:$P" VSEEDS="$NEW_V" SEEDS="$S" ARMS=star,L1 \
          WORKER=$W NWORKERS=$NW2 $PY test_multistart_cell.py
      done
    ) > logs_A10/ovr_${TAG}_w$W.log 2>&1 &
  done
fi

wait
echo "SIZE BATCH $SIZES DONE at $(date '+%F %H:%M')"
