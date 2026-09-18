"""Push the depth until the objective becomes visible at n = 12.

n = 6 is too small to carry the paper's claim, but it is the only size where M* currently separates
from M_L1 cleanly (AR_f 0.117 vs 0.289 at p=3). The controlling variable is the objective's share of
the trained Hamiltonian, share = objspan / (M * max|Q_pen|), and with M ~ A*T_norm*c_n,
T_norm ~ n^{3/2}/p, c_n ~ n, that scales as

    share ~ p / n^{5/2}

The shares that did and did not produce separation:
    n=6  p=2   7.3e-3   separates
    n=6  p=3   1.06e-2  separates strongly
    n=6  p=1   3.9e-3   blind
    n=12 p=3   2.0e-3   marginal
    n=12 p=5   3.6e-3   (M* only so far)
so n=12 needs p ~ 11 to match n=6 p=2 outright, or p ~ 7 combined with the 1.7x available from
restricting the span proxy to the feasible subspace (measured, not assumed). p=7 is the cheapest
shot with a real chance of landing, and it also fills in the depth trend.

Priority ordered: the n=12 cells the argument needs come first, the small-n ladder completion last.
Settings are identical to the M00 sweep (M0=0, 3000 steps, tol 1e-8, E_f=P75, eta_req 0.5, INTERP,
qaoa_seed 442, M pinned at seed 12345) and results land in the same two folders, so the notebook
picks them up with no change.

Run as several workers:  WORKER=0 NWORKERS=5 python run_depth_ladder.py
"""

import os
import pickle
import random
from timeit import default_timer as timer

import numpy as np

import analyse_M_L1_vs_star as cmpML1
from qaoa_sampling_oneshot import maqaoa_sampling_oneshot, N_stocks, w

STEPS, TOL, ETA_REQ, M_SEED = 3000, 1e-8, 0.5, 12345
M0_REF = 0.0
OUTDIRS = {
    "star": "./data_qaoa/PO_optimality_maqaoa_percentile_interp_M00",
    "L1":   "./data_qaoa/PO_optimality_maqaoa_percentile_interp_L1_M00",
}

# (n_bits, layers, vseeds, arms) in priority order
PLAN = [
    (12, 5, (42, 142, 242), ("L1",)),            # M* already imported from _span_expanded
    (12, 7, (42, 142, 242), ("star", "L1")),     # the decisive cell
    (9,  5, (42, 142, 242, 342, 442), ("L1",)),  # completes the small-n ladder
    (6,  5, (42, 142, 242, 342, 442), ("L1",)),
]

configs = [(n, p, v, mc) for n, p, vs, arms in PLAN for v in vs for mc in arms]
nw = int(os.environ.get("NWORKERS", 1))
wk = int(os.environ.get("WORKER", 0))
configs = configs[wk::nw]
print(f"worker {wk}/{nw}: {len(configs)} configs", flush=True)

for n_bits, layers, vseed, mc in configs:
    fname = (f"{OUTDIRS[mc]}/test_PO_bits_{n_bits}_vseed_{vseed}_layers_{layers}"
             f"_etareq_{ETA_REQ}.txt")
    if os.path.exists(fname):
        print(f"{os.path.basename(fname)} [{mc}] exists, skipping", flush=True)
        continue

    N_idx = int(np.where(N_stocks == n_bits // w)[0][0])
    t0 = timer()
    np.random.seed(M_SEED)
    random.seed(M_SEED)
    results = maqaoa_sampling_oneshot(
        "PO", N_idx, ("PO_small", vseed), "optimality", ETA_REQ, layers,
        qaoa_samples=1000, qaoa_steps=STEPS, tolerance_opt=TOL,
        Ef_from_percentile=True, Ef_percentile=75,
        warm_start=True, M_choice=mc, M0_ref=M0_REF, print_time=False,
    )

    Q, const = results["QUBO"]
    Q = np.asarray(Q, float)
    Q_norm = (Q / results["s_conv"]) / results["norm_final"]
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
        AR_f = float((mean_feas - E_min_f) / (E_obj[feas].max() - E_min_f))
    else:
        mean_feas = gap_feas = med_feas = AR_f = np.nan
    share_entry, share_feas = cmpML1.objective_share(n_bits, vseed, results["M_used"])

    res = {
        "eta_eff": float(results["eta_eff_ex"]), "best_pars": results["circuit_params"],
        "M_star": float(results["M_star"]), "M_L1": float(results["M_L1"]),
        "M_used": float(results["M_used"]), "M_choice": results["M_choice"],
        "s_conv": float(results["s_conv"]), "probs": probs,
        "Q": Q, "Q_const": float(const), "Q_norm": Q_norm,
        "Q_const_norm": float((const / results["s_conv"]) / results["norm_final"]),
        "vseed": int(vseed), "eta_req": float(ETA_REQ), "layers": int(layers), "bits": n_bits,
        "E_f": float(results["E_f"]), "Ef_from_percentile": True, "Ef_percentile": 75,
        "warm_start": True, "M0_ref": M0_REF, "steps": STEPS, "tol": TOL, "M_seed": M_SEED,
        "P_feas": float(P_feas), "gap_feas": gap_feas, "AR_f": AR_f,
        "mean_all": float(probs @ E_obj), "mean_feas": mean_feas, "med_feas": med_feas,
        "share_feas": float(share_feas), "share_entry": float(share_entry),
        "best_cost": float(results["best_cost"]), "secs": timer() - t0,
    }
    pickle.dump(res, open(fname, "wb"))
    print(f"{mc:4s} n={n_bits:2d} p={layers} v={vseed}:  M={res['M_used']:9.4g}  "
          f"share={res['share_feas']:.2e}  P_feas={res['P_feas']:.4f}  "
          f"eta={res['eta_eff']:.4f}  gap={res['gap_feas']:.4f}  AR_f={res['AR_f']:.4f}  "
          f"[{res['secs']:.0f}s]", flush=True)

print(f"worker {wk} done", flush=True)
