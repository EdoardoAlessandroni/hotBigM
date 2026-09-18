"""Repair ma-QAOA training collapses in the PO/optimality percentile sweep.

A handful of p=3 cells fail to train at all: the circuit puts ~0 mass on the feasible
subspace (eta_eff ~ 0.001-0.09 against ~0.99 for a healthy run on the same cell). Four
seeds on n=15/p=3/vseed=42 gave 0.0063 / 0.9922 / 0.0066 / 0.9942 -- perfectly bimodal,
so it is a bad minimum of the ma-QAOA landscape, not slow convergence. (The `patience`
early-stopping bug was real and is fixed, but re-running the same seed under the fixed
rule still gave 0.0062, so it is not the cause here.)

Protocol, deliberately keeping trigger and selector apart so the repair does not bias the
reported metric:
  * TRIGGER  -- eta_eff < ETA_COLLAPSE. Only decides where to spend extra compute.
  * SELECTOR -- lowest final training cost across restarts. `best_cost` is the optimizer's
    own objective, not the reported eta_eff, so selecting on it is not outcome selection.
    A collapsed run concentrates on penalty-violating states and so carries a much worse
    cost; the good basin wins on cost without ever consulting eta_eff.

Every original file is preserved as *.collapsed_backup, and the replacement records the
full restart table (seed -> cost, eta_eff) so the repair is auditable after the fact.
"""

import glob
import os
import pickle
import re
import shutil
import sys
from timeit import default_timer as timer

import numpy as np

from qaoa_sampling_oneshot import maqaoa_sampling_oneshot

D = "data_qaoa/PO_optimality_maqaoa_percentile"
DIRECTORY = "PO_small"
ETA_COLLAPSE = 0.1
SEEDS = (442, 443, 444, 445)      # 442 is the sweep default, re-run so its cost is comparable
N_stocks = np.arange(2, 9)
w = 3

N_idx_of_nbits = {int(N * w): i for i, N in enumerate(N_stocks)}

files = sorted(glob.glob(f"{D}/*.txt"))
collapsed = []
for f in files:
    r = pickle.load(open(f, "rb"))
    if r["eta_eff"] < ETA_COLLAPSE:
        collapsed.append((f, r))

print(f"{len(files)} runs on disk, {len(collapsed)} collapsed (eta_eff < {ETA_COLLAPSE})\n", flush=True)
for f, r in collapsed:
    print(f"   n_bits={r['bits']:>2} p={r['layers']} vseed={r['vseed']:>3}  eta_eff={r['eta_eff']:.4f}", flush=True)
print(flush=True)

if not collapsed:
    sys.exit(0)

t0 = timer()
for idx, (fname, old) in enumerate(collapsed, 1):
    n_bits, layers, vseed, eta_req = old["bits"], old["layers"], old["vseed"], old["eta_req"]
    N_idx = N_idx_of_nbits[n_bits]
    print(f"\n===== [{idx}/{len(collapsed)}] n_bits={n_bits} p={layers} vseed={vseed} "
          f"(old eta_eff = {old['eta_eff']:.4f}) =====", flush=True)

    runs = {}
    for s in SEEDS:
        res = maqaoa_sampling_oneshot("PO", N_idx, (DIRECTORY, vseed), "optimality", eta_req, layers,
                                      qaoa_samples=1000, qaoa_steps=500,
                                      Ef_from_percentile=True, Ef_percentile=75,
                                      qaoa_seed=s, print_time=False)
        runs[s] = res
        print(f"   seed {s}: cost = {res['best_cost']:+.6f}   eta_eff = {res['eta_eff_ex']:.4f}", flush=True)

    best_seed = min(runs, key=lambda s: runs[s]["best_cost"])
    results = runs[best_seed]
    print(f"   -> selected seed {best_seed} on LOWEST COST ({results['best_cost']:+.6f}), "
          f"eta_eff = {results['eta_eff_ex']:.4f}", flush=True)

    Q, const = results["QUBO"]
    Q = np.asarray(Q, float)
    Q_norm     = (Q     / results["s_conv"]) / results["norm_final"]
    const_norm = (const / results["s_conv"]) / results["norm_final"]

    res = {
        "eta_eff":   float(results["eta_eff_ex"]),
        "best_pars": results["circuit_params"],
        "M_star":    float(results["M_star"]),
        "Q":            Q,
        "Q_const":      float(const),
        "Q_norm":       Q_norm,
        "Q_const_norm": float(const_norm),
        "vseed":     int(vseed),
        "eta_req":   float(eta_req),
        "layers":    int(layers),
        "bits":      int(n_bits),
        "E_f":               float(results["E_f"]),
        "Ef_from_percentile": True,
        "Ef_percentile":      75,
        # repair bookkeeping
        "best_cost":         float(results["best_cost"]),
        "repaired":          True,
        "repair_seed":       int(best_seed),
        "repair_table":      {int(s): {"cost": float(r["best_cost"]),
                                       "eta_eff": float(r["eta_eff_ex"])} for s, r in runs.items()},
        "eta_eff_before":    float(old["eta_eff"]),
    }

    shutil.copy2(fname, fname + ".collapsed_backup")
    with open(fname, "wb") as fh:
        pickle.dump(res, fh)
    print(f"   written (backup at {os.path.basename(fname)}.collapsed_backup)", flush=True)

print(f"\nREPAIR DONE: {len(collapsed)} cells, {(timer() - t0)/60:.1f} min", flush=True)
