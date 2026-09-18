"""Naive-penalty baseline: identical to run_PO_optimality_interp.py but deploying M_L1 instead of M*.

M_L1 = L1_norm_hot(Q_obj, const_obj, n_bits, temperature_unnormalized, eta_req) is the closed-form
Eq.-12 penalty, evaluated at the SAME beta = 1/temperature_unnormalized that feeds M_method_opt.
That is the only fair choice: the bracket of the formula is an energy in raw QUBO units, so the
temperature added to it must be in those units too -- temperature_normalized is the dimensionless
(unit-span) temperature and would give M_L1 ~ 1e2 instead of ~1e6, i.e. the cold L1 norm.

Everything else is frozen at the M* arm's values so the two folders differ in the single variable M:
same grid, same vseeds, same E_f = P75, warm_start=True, qaoa_seed=442, 500 steps, 1000 samples, and
the same s_conv / beta (s_conv only sets the energy scale for the temperature; the trained
Hamiltonian is Q/max|Q| either way, so freezing it costs nothing and keeps beta comparable).

Extra vs. the M* arm's payload: the exact output distribution `probs`, so the objective-energy study
does not have to be reconstructed from shots. M_star is still computed and stored for the pairing.
"""

import os
import pickle
from timeit import default_timer as timer

import numpy as np

from qaoa_sampling_oneshot import maqaoa_sampling_oneshot

Ns = np.arange(3, 9)              # for NPP
P = 3                             # for NPP
N_city_circle = np.arange(2, 6)   # for TSP_circle
N_city_rand = np.arange(2, 6)     # for TSP_rand
N_stocks = np.arange(2, 9)        # for PO
w = 3                             # for PO
directory = "PO_small"
qaoa_samples = 1000

problem_type = "PO"
M_strategy = "optimality"
N_idx_list = np.array([0, 1, 2, 3])        # 1,2 for TSP   0,1,2 for NPP    0,1,2,3 for PO
layers_list = np.array([1, 2, 3])
eta_req_list = np.array([.5])
vseed_list = np.arange(42, 1042, 100)      # 42-52 (TSP_rand), 52-62 (NPP), 42-942::100 (PO)

# Optional worker split: VSEEDS="42,142" restricts this process to those instances, so the sweep can
# be run as several processes over disjoint vseed ranges (it is embarrassingly parallel, and the
# skip-if-exists check makes overlapping ranges merely wasteful rather than wrong). Safe because the
# deployed penalty M_L1 is a deterministic closed form and the circuit init is seeded explicitly by
# qaoa_seed, so a run's result does not depend on which process executes it.
if os.environ.get("VSEEDS"):
    vseed_list = np.array([int(v) for v in os.environ["VSEEDS"].split(",")])
    print(f"worker restricted to vseeds {vseed_list.tolist()}", flush=True)

Ef_percentile = 75
outdir = f"./data_qaoa/{problem_type}_{M_strategy}_maqaoa_percentile_interp_L1"
os.makedirs(outdir, exist_ok=True)

if problem_type == "PO":
    n_bits_list = w * N_stocks[N_idx_list]
elif problem_type == "NPP":
    n_bits_list = P * Ns[N_idx_list]
elif problem_type == "TSP_rand":
    n_bits_list = N_city_rand[N_idx_list]**2

t_start = timer()
n_done = 0
n_total = len(vseed_list) * len(N_idx_list) * len(layers_list) * len(eta_req_list)

# one file per (N_idx, vseed, layers, eta_req) combination
for vseed in vseed_list:
    t_vseed = timer()
    for i, N_idx in enumerate(N_idx_list):
        n_bits = int(n_bits_list[i])
        for j, layers in enumerate(layers_list):
            for k, eta_req in enumerate(eta_req_list):
                # check if file already exists
                filename = f"{outdir}/test_{problem_type}_bits_{n_bits}_vseed_{vseed}_layers_{layers}_etareq_{eta_req}.txt"
                if os.path.exists(filename):
                    print(filename, "  already exists, skipping", flush=True)
                    n_done += 1
                    continue

                info_instance = vseed
                if problem_type == "PO":
                    info_instance = (directory, vseed)

                results = maqaoa_sampling_oneshot(
                    problem_type, N_idx, info_instance, M_strategy, eta_req, layers,
                    qaoa_samples=qaoa_samples, qaoa_steps=500,
                    Ef_from_percentile=True, Ef_percentile=Ef_percentile,
                    warm_start=True,
                    M_choice="L1",
                )

                Q, const = results["QUBO"]
                Q = np.asarray(Q, float)
                assert Q.shape == (n_bits, n_bits), \
                    f"layers={layers}, eta_req={eta_req}: QUBO is {Q.shape}, expected {(n_bits, n_bits)}"

                # normalized QUBO actually trained by the circuit (pairs with best_pars);
                # reproduces the module's two-step scaling exactly (== Q/max(|Q|))
                Q_norm     = (Q     / results["s_conv"]) / results["norm_final"]
                const_norm = (const / results["s_conv"]) / results["norm_final"]

                res = {
                    "eta_eff":   float(results["eta_eff_ex"]),   # scalar
                    "best_pars": results["circuit_params"],      # single params array
                    "M_star":    float(results["M_star"]),       # our algorithm's M (not deployed)
                    "M_L1":      float(results["M_L1"]),         # naive Eq.-12 M
                    "M_used":    float(results["M_used"]),       # == M_L1 here
                    "M_choice":  results["M_choice"],            # "L1"
                    "s_conv":    float(results["s_conv"]),
                    "probs":     np.asarray(results["probability_states"], float),
                    "Q":            Q,                           # raw (n_bits, n_bits), built at M_L1
                    "Q_const":      float(const),                # raw scalar
                    "Q_norm":       Q_norm,                      # normalized (== Q/max(|Q|)), trained by circuit
                    "Q_const_norm": float(const_norm),           # normalized scalar
                    "vseed":     int(vseed),
                    "eta_req":   float(eta_req),
                    "layers":    int(layers),
                    "bits":      n_bits,
                    # cutoff bookkeeping, specific to this percentile run
                    "E_f":               float(results["E_f"]),
                    "Ef_from_percentile": True,
                    "Ef_percentile":      Ef_percentile,
                    "warm_start":         True,
                }

                file = open(filename, "wb")
                pickle.dump(res, file)
                file.close()
                n_done += 1
                print(filename, f"  written   [E_f = {res['E_f']:.5f}, M_L1 = {res['M_L1']:.4g} "
                                f"(M* = {res['M_star']:.4g}, ratio {res['M_L1']/res['M_star']:.2f}), "
                                f"eta_eff = {res['eta_eff']:.3f}]   {n_done}/{n_total}", flush=True)

    print(f"\n=== VSEED {vseed} DONE ({timer() - t_vseed:.0f}s, {n_done}/{n_total} total, "
          f"elapsed {(timer() - t_start)/60:.1f} min) ===\n", flush=True)

print(f"ALL DONE: {n_done}/{n_total} files in {outdir}, total {(timer() - t_start)/60:.1f} min", flush=True)
