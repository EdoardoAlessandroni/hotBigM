"""Train ma-QAOA on PO / optimality with the instance-adaptive energy cutoff
E_f = percentile(feasible objective spectrum, 75), instead of select_Ef's 1e-1 * n_bits**2.

Mirrors the training loop of qaoa_playground.ipynb one-for-one (same grids, same
qaoa_samples/qaoa_steps, same pickled payload); the only difference is
Ef_from_percentile=True and the output folder. vseed is the outermost loop, so a
complete (all sizes x all layers) slice is on disk after each instance.
"""

import os
import pickle
import sys
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

Ef_percentile = 75
outdir = f"./data_qaoa/{problem_type}_{M_strategy}_maqaoa_percentile"
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
                    "M_star":    float(results["M_star"]),       # scalar
                    "Q":            Q,                           # raw (n_bits, n_bits)
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
                }

                file = open(filename, "wb")
                pickle.dump(res, file)
                file.close()
                n_done += 1
                print(filename, f"  written   [E_f = {res['E_f']:.5f}, M* = {res['M_star']:.4g}, "
                                f"eta_eff = {res['eta_eff']:.3f}]   {n_done}/{n_total}", flush=True)

    print(f"\n=== VSEED {vseed} DONE ({timer() - t_vseed:.0f}s, {n_done}/{n_total} total, "
          f"elapsed {(timer() - t_start)/60:.1f} min) ===\n", flush=True)

print(f"ALL DONE: {n_done}/{n_total} files in {outdir}, total {(timer() - t_start)/60:.1f} min", flush=True)
