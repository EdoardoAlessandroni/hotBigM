"""Does the n=9 result generalise? Span proxy x size x depth, with M pinned and tight convergence.

At n=9, p=3, M0=0 the circuit stops being a blind feasibility projector once it is allowed to
converge (gap_feas 1.10 -> 0.65, eta_eff -> 1.0000, P_feas -> 1.0000), and BOTH ingredients are
needed: 3000 steps at tol 1e-4 reproduces the 500-step numbers exactly, and tol 1e-8 at the
pipeline's M converges to the blind projector instead. n=12 did not replicate. This grid settles it.

Two fixes over `run_span_variants.py`:
  * M is pinned. `M_method_opt` samples the feasible spectrum with the global RNG, so nominally
    identical calls returned M values ~3% apart and the "matched pairs" were not matched. Seeding
    numpy+random before each run makes M exactly reproducible (verified: 51.484375 every time).
  * More vseeds, plus n = 6/15 and depth p = 5, so the size and depth trends are visible.

Everything is trained at 3000 steps / tol 1e-8 (the setting under which the effect appears),
ma-QAOA + INTERP, qaoa_seed 442, E_f = P75, eta_req 0.5.

Run as several workers:  WORKER=0 NWORKERS=12 python run_span_expanded.py
"""

import os
import pickle
import random
from timeit import default_timer as timer

import numpy as np

import analyse_M_L1_vs_star as cmpML1
from qaoa_sampling_oneshot import maqaoa_sampling_oneshot, N_stocks, w

STEPS, TOL, ETA_REQ = 3000, 1e-8, 0.5
M_SEED = 12345          # pins M_method_opt's feasible-spectrum sampling
OUT = "data_qaoa/_span_expanded"
os.makedirs(OUT, exist_ok=True)

# (n_bits, layers) -> vseeds. n=15 p=3 and n=12 p=5 are the expensive corners, so fewer instances.
PLAN = {
    (6, 3): (42, 142, 242, 342, 442),
    (9, 3): (42, 142, 242, 342, 442),
    (12, 3): (42, 142, 242, 342, 442),
    (15, 3): (42, 142, 242),
    (6, 5): (42, 142, 242, 342, 442),
    (9, 5): (42, 142, 242, 342, 442),
    (12, 5): (42, 142, 242),
}
M0_REFS = (0.0, 1.0)

configs = [(m0, n, p, v)
           for (n, p), vs in PLAN.items() for v in vs for m0 in M0_REFS]
# cheapest-first so every worker returns something early, then interleave across workers
configs.sort(key=lambda c: (c[1], c[2]))
nw = int(os.environ.get("NWORKERS", 1))
wk = int(os.environ.get("WORKER", 0))
configs = configs[wk::nw]
print(f"worker {wk}/{nw}: {len(configs)} configs", flush=True)

for m0, n_bits, layers, vseed in configs:
    fname = f"{OUT}/m0{m0:g}_n{n_bits}_p{layers}_v{vseed}.pkl"
    if os.path.exists(fname):
        print(f"{os.path.basename(fname)} exists, skipping", flush=True)
        continue

    N_idx = int(np.where(N_stocks == n_bits // w)[0][0])
    t0 = timer()
    np.random.seed(M_SEED)      # pin M_method_opt; circuit init is seeded separately by qaoa_seed
    random.seed(M_SEED)
    res = maqaoa_sampling_oneshot(
        "PO", N_idx, ("PO_small", vseed), "optimality", ETA_REQ, layers,
        qaoa_samples=1000, qaoa_steps=STEPS, tolerance_opt=TOL,
        Ef_from_percentile=True, Ef_percentile=75,
        warm_start=True, M_choice="star", M0_ref=m0,
        print_time=False,
    )

    E_obj, E_pen = cmpML1.spectra(n_bits, vseed)
    feas = np.isclose(E_pen, 0)
    probs = np.asarray(res["probability_states"], float)
    P_feas = probs[feas].sum()
    if P_feas > 1e-12:
        wf = probs[feas] / P_feas
        mean_feas = float(wf @ E_obj[feas])
        E_min_f, E_mean_f = E_obj[feas].min(), E_obj[feas].mean()
        gap_feas = float((mean_feas - E_min_f) / (E_mean_f - E_min_f))
        med_feas = cmpML1.weighted_quantile(E_obj[feas], wf, 0.5)
    else:
        mean_feas, gap_feas, med_feas = np.nan, np.nan, np.nan
    share_entry, share_feas = cmpML1.objective_share(n_bits, vseed, res["M_used"])

    rec = dict(M0_ref=m0, bits=n_bits, layers=layers, vseed=vseed, steps=STEPS, tol=TOL,
               M_used=float(res["M_used"]), s_conv=float(res["s_conv"]),
               eta_eff=float(res["eta_eff_ex"]), P_feas=float(P_feas),
               mean_all=float(probs @ E_obj), mean_feas=mean_feas, med_feas=med_feas,
               gap_feas=gap_feas, share_feas=float(share_feas), share_entry=float(share_entry),
               best_cost=float(res["best_cost"]), E_f=float(res["E_f"]), secs=timer() - t0)
    pickle.dump(rec, open(fname, "wb"))
    print(f"M0={m0:g} n={n_bits} p={layers} v={vseed}:  M={rec['M_used']:9.4g}  "
          f"share={rec['share_feas']:.2e}  P_feas={rec['P_feas']:.4f}  "
          f"eta={rec['eta_eff']:.4f}  gap={rec['gap_feas']:.4f}  [{rec['secs']:.0f}s]", flush=True)

print("worker done", flush=True)
