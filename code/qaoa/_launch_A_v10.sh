#!/bin/bash
# DATASET A STATISTICS TOP-UP: 5 -> 10 instances per size.
#
# Adds vseeds 542,642,742,842,942 to the existing 42,142,242,342,442. The .lp files for all of
# them already exist under ../../data/PO_small/{6,9,12,15}/ (100 instances per size are shipped),
# so nothing has to be generated.
#
# 150 runs total = 120 at the base init + 30 at the three override inits. Two pools run
# CONCURRENTLY, because they need DIFFERENT circuit inits and running them in sequence would leave
# most cores idle in each tail:
#
#   pool 1, 14 workers, ~43 CPU-h.  The seed-442 sweep, all 12 cells, both arms, written as .txt
#       into the PUBLISHED sweep folders. That is deliberate: make_fig_paired.py, the notebook's
#       CELL_LOAD and export_dataset_A.py all build their base table by globbing those folders, so
#       instances landing there are picked up with no code change anywhere downstream.
#
#   pool 2, 6 workers, ~20 CPU-h.   The three cells dataset A does NOT run at seed 442 --
#       (9,2)->31337, (12,3)->7, (15,3)->2024 -- at their own init, into _multistart_n{N}p{P}/.
#       This is exactly where the existing overrides for vseeds 42..442 live, so the substitution
#       loops in those same three scripts pick them up automatically too.
#
# Expected wall time ~3.5 h on 20 cores. Both pools skip runs that already carry their circuit, so
# this script is safe to re-run after an interruption or a partial failure.
#
# Read HANDOFF.md before touching any of this, especially the section on what the override seeds
# mean -- two of the three were chosen by looking at AR on the ORIGINAL five instances, which has a
# direct consequence for how the ten-instance cells may be reported.
set -u
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
PY=/home/edo/anaconda3/envs/qubo_env/bin/python
NEW_V="542,642,742,842,942"

# One BLAS thread per process. Without this every worker spawns 20 threads and the box thrashes.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs_A10

echo "pool 1: seed-442 sweep, 12 cells x 5 new vseeds x 2 arms = 120 runs, 14 workers"
for W in $(seq 0 13); do
  VSEEDS_EXTRA="$NEW_V" WORKER=$W NWORKERS=14 \
    $PY run_PO_optimality_interp_M00.py > logs_A10/sweep_w$W.log 2>&1 &
done

echo "pool 2: override cells at their own init, 30 runs, 6 workers"
for W in $(seq 0 5); do
  (
    CELLS="9:2"  VSEEDS="$NEW_V" SEEDS=31337 ARMS=star,L1 WORKER=$W NWORKERS=6 $PY test_multistart_cell.py
    CELLS="12:3" VSEEDS="$NEW_V" SEEDS=7     ARMS=star,L1 WORKER=$W NWORKERS=6 $PY test_multistart_cell.py
    CELLS="15:3" VSEEDS="$NEW_V" SEEDS=2024  ARMS=star,L1 WORKER=$W NWORKERS=6 $PY test_multistart_cell.py
  ) > logs_A10/ovr_w$W.log 2>&1 &
done

wait
echo "DATASET A TOP-UP DONE at $(date '+%F %H:%M')"
echo "Now do the post-run checklist in HANDOFF.md -- in particular export_dataset_A.py:VSEEDS"
echo "still says 5 instances, so the exported folder will silently stay at 120 records until it is"
echo "widened to all ten."
