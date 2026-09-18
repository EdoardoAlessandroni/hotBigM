"""Does tightening the Adam tolerance let the circuit see the objective at M*?

Claim under test (previously ASSERTED, never measured): the objective's share of the trained
Hamiltonian sits below tolerance_opt = 1e-4, so training converges before the objective can move a
parameter, and no amount of extra optimization helps.

The claim is not obvious. Once the penalty part of the cost is at its optimum its gradient vanishes,
so the residual gradient is dominated by the objective term -- and Adam rescales per-parameter by
the running gradient RMS, so a systematically tiny-but-consistent gradient can still drive motion.
The objective's contribution to the cost (~1e-7 at n=9, ~1e-8 at n=12) is also far above float64
resolution (~1e-15 on a cost of order 1), so it is numerically there to be found.

Sweeps tolerance_opt over {1e-4, 1e-8, 1e-12} at a raised step budget, holding the instance, the
penalty (M*) and the circuit seed fixed. Reports eta_eff and gap_feas -- the scale-free energy
metric, 1 = uniform over the feasible set (blind), 0 = sits on the best feasible state.

Run as several workers:  CONFIGS="0,1,2" python test_tolerance_objective.py
"""

import os
import pickle
import sys
from timeit import default_timer as timer

import numpy as np

import analyse_M_L1_vs_star as cmpML1
from qaoa_sampling_oneshot import maqaoa_sampling_oneshot, N_stocks, w

TOLS = (1e-4, 1e-8, 1e-12)
CELLS = [(9, 42), (9, 142), (9, 242), (12, 42), (12, 142)]
STEPS = 3000
LAYERS = 3
OUT = "data_qaoa/_tolerance_objective"
os.makedirs(OUT, exist_ok=True)

configs = [(n, v, t) for (n, v) in CELLS for t in TOLS]
if os.environ.get("CONFIGS"):
    picked = [int(i) for i in os.environ["CONFIGS"].split(",")]
    configs = [configs[i] for i in picked]
    print(f"worker handling configs {picked}", flush=True)

for n_bits, vseed, tol in configs:
    fname = f"{OUT}/n{n_bits}_v{vseed}_tol{tol:.0e}.pkl"
    if os.path.exists(fname):
        print(f"{fname} exists, skipping", flush=True)
        continue

    N_idx = int(np.where(N_stocks == n_bits // w)[0][0])
    t0 = timer()
    res = maqaoa_sampling_oneshot(
        "PO", N_idx, ("PO_small", vseed), "optimality", 0.5, LAYERS,
        qaoa_samples=1000, qaoa_steps=STEPS,
        Ef_from_percentile=True, Ef_percentile=75,
        warm_start=True, M_choice="star",
        tolerance_opt=tol, print_time=False,
    )

    # score on the exact distribution, same definitions as the M_L1 vs M* study
    E_obj, E_pen = cmpML1.spectra(n_bits, vseed)
    feas = np.isclose(E_pen, 0)
    probs = np.asarray(res["probability_states"], float)
    P_feas = probs[feas].sum()
    wf = probs[feas] / P_feas
    mean_feas = float(wf @ E_obj[feas])
    E_min_f, E_mean_f = E_obj[feas].min(), E_obj[feas].mean()
    gap_feas = (mean_feas - E_min_f) / (E_mean_f - E_min_f)

    rec = dict(n_bits=n_bits, vseed=vseed, tol=tol, steps=STEPS, layers=LAYERS,
               eta_eff=float(res["eta_eff_ex"]), P_feas=float(P_feas),
               mean_feas=mean_feas, gap_feas=float(gap_feas),
               best_cost=float(res["best_cost"]), M_star=float(res["M_star"]),
               secs=timer() - t0)
    pickle.dump(rec, open(fname, "wb"))
    print(f"n={n_bits} v={vseed} tol={tol:.0e}:  eta={rec['eta_eff']:.4f}  "
          f"P_feas={rec['P_feas']:.4f}  gap_feas={rec['gap_feas']:.4f}  "
          f"cost={rec['best_cost']:.12f}  [{rec['secs']:.0f}s]", flush=True)

print("worker done", flush=True)
