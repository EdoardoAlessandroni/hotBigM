"""Recover the ORIGINAL circuit parameters for the two `_span_expanded` runs that re-run differently.

The problem. 18 dataset-A records (M*, p=3) were trained 2026-09-17 under a trainer that no longer
exists: the stopping-rule fix is bundled in commit 7683e400, the Sep-17 source was never committed,
no .pyc or stdout log survives, and `d4c89267` predates INTERP entirely (no `warm_start` parameter),
so there is no git ancestor of the ladder. Re-running under today's rule reproduces 4 exactly and 8
to <2e-3, but two land far away:

    n=6  p=3 v=242    AR 0.878250 -> 0.895403   (+1.7e-2)
    n=9  p=3 v=142    AR 0.813920 -> 0.659617   (-1.5e-1)

Reverting just the stopping block (`qaoa_sampling_legacy`) does not reproduce them either -- at
n=6 v=242 it reaches cost -0.26535201 against the stored -0.26523962965007863, i.e. it trains PAST
where the original stopped.

The idea. The original's `best_cost` survives at full double precision in the pickle, and within a
depth `best_cost` is MONOTONE decreasing. So if the INTERP ladder agreed through depth p-1 and the
two runs differ only in where the final depth stopped, that exact double must appear somewhere in
the final depth's best_cost trajectory. Recording every step of the final depth therefore either

  * HITS the target bit-for-bit -> the ladder is confirmed identical, the stopping step is
    identified, and the parameters at that step ARE the original circuit; or
  * MISSES -> the divergence is upstream (depth 1 or 2), the original ladder is unrecoverable, and
    the run must be replaced rather than reconstructed.

Either way the answer is definitive, which a blind search over trainer variants would not be.

Method: run the ladder with the legacy rule at depths < p, and at depth p disable early stopping
(epsilon = 0 can never satisfy `|diff| < epsilon`) while snapshotting (params, cost) at every Adam
step via an instrumented optimiser. Then scan for the target.

    TARGET="6:3:242" python recover_original.py

Writes data_qaoa/_recovered/. Reads the sweep folder; never writes to it.
"""

import os
import pickle
import random
from timeit import default_timer as timer

import numpy as np
import pennylane as qml

import analyse_M_L1_vs_star as cmpML1
import qaoa_sampling_legacy as L

SWEEP = "data_qaoa/PO_optimality_maqaoa_percentile_interp_M00"
OUT = "data_qaoa/_recovered"
assert SWEEP not in OUT
os.makedirs(OUT, exist_ok=True)

ETA_REQ, M_SEED, M0_REF = 0.5, 12345, 0.0
STEPS, TOL, QAOA_SEED = 3000, 1e-8, 442

N_BITS, LAYERS, VSEED = (int(x) for x in os.environ.get("TARGET", "6:3:242").split(":"))

old = pickle.load(open(f"{SWEEP}/test_PO_bits_{N_BITS}_vseed_{VSEED}"
                       f"_layers_{LAYERS}_etareq_0.5.txt", "rb"))
TARGET_COST = float(old["best_cost"])
print(f"target n={N_BITS} p={LAYERS} v={VSEED}   best_cost = {TARGET_COST!r}", flush=True)

# ---- instrument: snapshot every Adam step, and never early-stop at the final depth -------------
# Two bugs bit the first attempt and are guarded against here:
#   * PennyLane returns a LIST, not a tuple, when stepping two parameter groups, so an
#     `isinstance(params, tuple)` test silently recorded nothing.
#   * Patching `run_ma_qaoa_with_solution` did not reach the final depth. The ladder is therefore
#     reimplemented below rather than patched -- it is eight lines, and this way the per-depth
#     tolerance is unambiguously under our control. Each depth reports its own best_cost, so an
#     upstream divergence is visible instead of being inferred.
REC = []
DEPTH = {"cur": 0}


class RecordingAdam(qml.AdamOptimizer):
    def step_and_cost(self, *a, **k):
        out = super().step_and_cost(*a, **k)
        params, cost = out[0], out[1]
        if isinstance(params, (list, tuple)) and len(params) == 2:
            g, b = params
            REC.append((DEPTH["cur"], np.array(g), np.array(b), float(cost)))
        return out


_orig_single = L.run_ma_qaoa_with_solution


def my_interp(qubo_dict, num_qubits, p, verbose=False, **kw):
    """L.run_ma_qaoa_interp, but with epsilon forced to 0 at the final depth so it never breaks."""
    init = None
    for p_cur in range(1, p + 1):
        DEPTH["cur"] = p_cur
        kw2 = dict(kw)
        if p_cur == p:
            kw2["tolerance_opt"] = 0.0       # |diff| < 0 is never true -> runs the full STEPS
        n0 = len(REC)
        out = _orig_single(qubo_dict, num_qubits, p=p_cur, init_params=init,
                           verbose=verbose, **kw2)
        (g, b), bc = out[0], out[-1]
        print(f"[ladder] depth {p_cur}/{p}  tol={kw2.get('tolerance_opt')!r}  "
              f"steps={len(REC) - n0}  best_cost={bc!r}", flush=True)
        if p_cur < p:
            init = (L.interp_extend_layers(g), L.interp_extend_layers(b))
    return out


L.run_ma_qaoa_interp = my_interp
L.qml.AdamOptimizer = RecordingAdam

N_idx = int(np.where(L.N_stocks == N_BITS // L.w)[0][0])
t0 = timer()
np.random.seed(M_SEED)
random.seed(M_SEED)
res = L.maqaoa_sampling_oneshot(
    "PO", N_idx, ("PO_small", VSEED), "optimality", ETA_REQ, LAYERS,
    qaoa_samples=1000, qaoa_steps=STEPS, tolerance_opt=TOL,
    Ef_from_percentile=True, Ef_percentile=75, warm_start=True,
    M_choice="star", M0_ref=M0_REF, qaoa_seed=QAOA_SEED,
    print_time=False,
)
print(f"ladder done in {timer() - t0:.0f}s; {len(REC)} recorded steps", flush=True)

# ---- final depth only, monotone best_cost trajectory -------------------------------------------
final = [(g, b, c) for d, g, b, c in REC if d == LAYERS]
print(f"final-depth steps: {len(final)}")
assert final, "no final-depth steps recorded -- instrumentation is not firing"

best, traj = np.inf, []
for i, (g, b, c) in enumerate(final):
    if c < best:
        best = c
        traj.append((i, best, g, b))
print(f"best_cost improved {len(traj)} times; "
      f"range [{traj[0][1]:.12f} .. {traj[-1][1]:.12f}]")

hits = [t for t in traj if t[1] == TARGET_COST]
near = min(traj, key=lambda t: abs(t[1] - TARGET_COST))
print(f"\nEXACT hits: {len(hits)}")
print(f"closest: step {near[0]}  best_cost {near[1]!r}  |delta| {abs(near[1] - TARGET_COST):.3e}")

if hits:
    step, cost, g, b = hits[0]
    print(f"\nRECOVERED: the original stopped at final-depth step {step}")
    # NOTE: probs are NOT stored here -- `res` holds the full-3000-step circuit, not the one at
    # `step`. Confirm by re-running the pipeline with qaoa_steps capped at step+1, which drives the
    # normal code path and yields probs/mean_feas to check against the stored mean_feas.
    rec = dict(arm="star", bits=N_BITS, layers=LAYERS, vseed=VSEED, qaoa_seed=QAOA_SEED,
               circuit_params=(g, b), QUBO=res["QUBO"], M_used=float(res["M_used"]),
               best_cost=cost, best_cost_target=TARGET_COST, stop_step=step,
               stopping_rule="recovered_by_trajectory_search",
               steps=STEPS, tol=TOL, M0_ref=M0_REF, M_seed=M_SEED,
               eta_req=ETA_REQ, Ef_percentile=75, Ef_from_percentile=True, warm_start=True,
               mean_feas_target=float(old["mean_feas"]), secs=timer() - t0)
    pickle.dump(rec, open(f"{OUT}/star_n{N_BITS}_p{LAYERS}_v{VSEED}_s{QAOA_SEED}.pkl", "wb"))
    print(f"wrote {OUT}/star_n{N_BITS}_p{LAYERS}_v{VSEED}_s{QAOA_SEED}.pkl")
else:
    print("\nNOT RECOVERABLE from the final depth: the ladder diverged at depth 1 or 2.")
    print("The original circuit cannot be reconstructed; this run must be replaced, not reproduced.")

np.save(f"{OUT}/traj_n{N_BITS}_p{LAYERS}_v{VSEED}.npy",
        np.array([(i, c) for i, c, _, _ in traj]))
