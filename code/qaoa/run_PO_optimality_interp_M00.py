"""The M0=0 sweep: M* vs M_L1 on the SAME footing, in the regime where the objective is visible.

The original percentile_interp sweeps (and their L1 twin) evaluate the span proxy at
M0 = L1_norm_hot(..., T_norm, eta) ~ 1e2, which makes the penalty term dominate s_conv by ~3 decades
and inflates both penalties to ~1e6. There the objective is ~1e-7 of the trained Hamiltonian and
both arms are blind feasibility projectors -- the 120-run comparison could not discriminate.

Setting M0 = 0 (span proxy = objective UB - LB only, M-free by construction) cuts M by 3-5 decades
and lifts the objective's share of the trained Hamiltonian to ~1e-2. Combined with a tight
convergence tolerance -- BOTH are required, neither alone does anything -- the circuit stops being a
blind projector and starts ranking feasible states by objective. This sweep asks the paper's actual
question in that regime: does M* buy objective quality *over the naive M_L1 baseline*?

Grid: n_bits in {6,9,12,15} x p in {1,2,3} x 5 vseeds x {M*, M_L1}.
p = 2 is included because the earlier span-variant runs only covered p in {1,3} and {3,5}.
n=15 originally ran 3 vseeds (it is ~4x the cost of n=12 and the effect is weakest there); it was
topped up to 5 on 2026-09-18 so every cell averages over the same instance count.

Settings differing from the original sweep, and why:
  M0_ref = 0.0     the point of the experiment
  3000 steps       the p=3 INTERP schedule needs room; 500 early-stops
  tol = 1e-8       at 1e-4 the optimizer stops before the objective term matters. Control already
                   run: 3000 steps at tol 1e-4 reproduces the 500-step numbers EXACTLY.
Everything else is the established sweep: E_f = P75, eta_req = 0.5, INTERP warm start, seed 442,
1000 samples.

M is PINNED (np.random.seed(M_SEED) before each call). `M_method_opt` samples the feasible spectrum
off the global RNG, so nominally identical calls drift ~3% and the M*/M_L1 pairs are then not
matched on the same instance. The circuit init is seeded separately by qaoa_seed, so this does not
touch the optimization.

Run as several workers:  WORKER=0 NWORKERS=10 python run_PO_optimality_interp_M00.py
"""

import os
import pickle
import random
from timeit import default_timer as timer

import numpy as np

import analyse_M_L1_vs_star as cmpML1
from qaoa_sampling_oneshot import maqaoa_sampling_oneshot, N_stocks, w

STEPS, TOL, ETA_REQ = 3000, 1e-8, 0.5
M0_REF = 0.0
M_SEED = 12345
QAOA_SAMPLES = 1000
EF_PERCENTILE = 75
DIRECTORY = "PO_small"

OUTDIRS = {
    "star": "./data_qaoa/PO_optimality_maqaoa_percentile_interp_M00",
    "L1":   "./data_qaoa/PO_optimality_maqaoa_percentile_interp_L1_M00",
}
for d in OUTDIRS.values():
    os.makedirs(d, exist_ok=True)

VSEEDS = {6: (42, 142, 242, 342, 442), 9: (42, 142, 242, 342, 442),
          12: (42, 142, 242, 342, 442), 15: (42, 142, 242, 342, 442)}
BITS = (6, 9, 12, 15)
LAYERS = (1, 2, 3)
ARMS = tuple(os.environ.get("ARMS", "star,L1").split(","))

configs = [(n, p, v, mc)
           for n in BITS for p in LAYERS for v in VSEEDS[n] for mc in ARMS]

# ONLY="L1:12:3:442,L1:9:3:442" runs exactly these cells and nothing else. Used to pull specific
# configs off a busy worker's queue onto idle cores. NOTE: a process started later loads the
# current module, so its training protocol is whatever the code says NOW -- if the stopping rule
# has changed since the sweep began, these cells differ from their neighbours. The payload records
# `patience`, and files written before the rule change have no such key, so the split is auditable.
if os.environ.get("ONLY"):
    picked = []
    for spec in os.environ["ONLY"].split(","):
        mc, n, p, v = spec.split(":")
        picked.append((int(n), int(p), int(v), mc))
    configs = picked
    print(f"ONLY mode: {configs}", flush=True)
# INTERP trains depths 1..p, so cost ~ p(p+1)/2; cheapest-first then interleaved across workers
# means every worker returns usable rows early instead of all of them starting on n=15.
configs.sort(key=lambda c: (c[0] * c[1] * (c[1] + 1), c[0]))
nw = int(os.environ.get("NWORKERS", 1))
wk = int(os.environ.get("WORKER", 0))
configs = configs[wk::nw]
print(f"worker {wk}/{nw}: {len(configs)} configs", flush=True)

t_start = timer()
for n_bits, layers, vseed, mc in configs:
    outdir = OUTDIRS[mc]
    fname = (f"{outdir}/test_PO_bits_{n_bits}_vseed_{vseed}_layers_{layers}"
             f"_etareq_{ETA_REQ}.txt")
    if os.path.exists(fname):
        print(f"{os.path.basename(fname)} [{mc}] exists, skipping", flush=True)
        continue

    N_idx = int(np.where(N_stocks == n_bits // w)[0][0])
    t0 = timer()
    np.random.seed(M_SEED)      # pin M_method_opt's feasible-spectrum sampling
    random.seed(M_SEED)
    results = maqaoa_sampling_oneshot(
        "PO", N_idx, (DIRECTORY, vseed), "optimality", ETA_REQ, layers,
        qaoa_samples=QAOA_SAMPLES, qaoa_steps=STEPS, tolerance_opt=TOL,
        Ef_from_percentile=True, Ef_percentile=EF_PERCENTILE,
        warm_start=True, M_choice=mc, M0_ref=M0_REF,
        print_time=False,
    )

    Q, const = results["QUBO"]
    Q = np.asarray(Q, float)
    assert Q.shape == (n_bits, n_bits), f"QUBO is {Q.shape}, expected {(n_bits, n_bits)}"
    Q_norm = (Q / results["s_conv"]) / results["norm_final"]          # == Q/max|Q|
    const_norm = (const / results["s_conv"]) / results["norm_final"]

    # scoring on the exact output distribution, same definitions as the M_L1-vs-M* study
    E_obj, E_pen = cmpML1.spectra(n_bits, vseed)
    feas = np.isclose(E_pen, 0)
    probs = np.asarray(results["probability_states"], float)
    P_feas = probs[feas].sum()
    if P_feas > 1e-12:
        wf = probs[feas] / P_feas
        mean_feas = float(wf @ E_obj[feas])
        E_min_f, E_mean_f = E_obj[feas].min(), E_obj[feas].mean()
        gap_feas = float((mean_feas - E_min_f) / (E_mean_f - E_min_f))
        med_feas = cmpML1.weighted_quantile(E_obj[feas], wf, 0.5)
    else:
        mean_feas, gap_feas, med_feas = np.nan, np.nan, np.nan
    share_entry, share_feas = cmpML1.objective_share(n_bits, vseed, results["M_used"])

    res = {
        "eta_eff":   float(results["eta_eff_ex"]),
        "best_pars": results["circuit_params"],
        "M_star":    float(results["M_star"]),
        "M_L1":      float(results["M_L1"]),
        "M_used":    float(results["M_used"]),
        "M_choice":  results["M_choice"],
        "s_conv":    float(results["s_conv"]),
        "probs":     probs,
        "Q": Q, "Q_const": float(const),
        "Q_norm": Q_norm, "Q_const_norm": float(const_norm),
        "vseed": int(vseed), "eta_req": float(ETA_REQ),
        "layers": int(layers), "bits": n_bits,
        "E_f": float(results["E_f"]),
        "Ef_from_percentile": True, "Ef_percentile": EF_PERCENTILE,
        "warm_start": True,
        # what makes this sweep different
        "M0_ref": M0_REF, "steps": STEPS, "tol": TOL, "M_seed": M_SEED,
        # training protocol. The stopping rule changed on 2026-09-18 (best-cost tracking +
        # window-total improvement, default patience 3 -> 30). Runs are only comparable to each
        # other if this matches; files written before the change have no "patience" key at all.
        "patience": results.get("patience"),
        # scoring
        "P_feas": float(P_feas), "gap_feas": gap_feas,
        "mean_all": float(probs @ E_obj), "mean_feas": mean_feas, "med_feas": med_feas,
        "share_feas": float(share_feas), "share_entry": float(share_entry),
        "best_cost": float(results["best_cost"]),
        "secs": timer() - t0,
    }
    pickle.dump(res, open(fname, "wb"))
    print(f"{mc:4s} n={n_bits:2d} p={layers} v={vseed}:  M={res['M_used']:9.4g}  "
          f"share={res['share_feas']:.2e}  P_feas={res['P_feas']:.4f}  "
          f"eta={res['eta_eff']:.4f}  gap={res['gap_feas']:.4f}  [{res['secs']:.0f}s]", flush=True)

print(f"worker {wk} done, {(timer() - t_start)/60:.1f} min", flush=True)
