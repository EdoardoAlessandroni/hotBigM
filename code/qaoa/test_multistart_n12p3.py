"""Decisive test: does multi-start over the circuit init rescue the WEAKEST cell, n=12 p=3?

Background. At n=9 p=2 (`probe_n9p2.py`) the circuit init `qaoa_seed` turned out to dominate every
other hyperparameter: 6000 steps at tol 1e-12 reproduced 3000 at 1e-8 to 8 decimals (so the result
is a converged local optimum, not under-training), while changing the seed moved mean AR from 0.56
to 0.86. Crucially, picking the seed with the LOWEST TRAINED COST -- no oracle knowledge -- selected
the same run the AR-oracle would have on 5/5 instances, lifting mean AR 0.5643 -> 0.9089.

n=12 p=3 is the decisive case because it is where the whole study is weakest: M* 0.6436 vs M_L1
0.5974 against a blind reference of 0.5788, i.e. both arms are barely above a blind feasibility
projector, and the objective's share (1.6-2.5e-3) sits below the ~5e-3 needed for the circuit to
rank feasible states. If multi-start lifts THIS cell, the study's absolute numbers are limited by
the init rather than by share, and the whole grid should be re-run that way.

Both arms are included because the paper's question is the M* - M_L1 gap, not absolute quality: if
multi-start lifts both arms equally the headline claim is untouched, and if it lifts M_L1 more the
claim is in trouble. That has to be measured, not assumed.

Grid: {star, L1} x 5 vseeds x qaoa_seed in {442, 7, 2024, 31337}, n=12, p=3. 40 runs.
Ordered vseed-major so complete instances land early and can be reported before the rest finish.

Protocol otherwise identical to the M00 sweep: M0=0, 3000 steps, tol 1e-8, patience 30 (the fixed
rule), E_f = P75, eta_req 0.5, INTERP warm start, M pinned with np.random.seed(12345).

WRITES ONLY TO data_qaoa/_multistart_n12p3/. The finished sweep folders are never opened.
"""

import os
import pickle
import random
from timeit import default_timer as timer

import numpy as np

import analyse_M_L1_vs_star as cmpML1
from qaoa_sampling_oneshot import maqaoa_sampling_oneshot, N_stocks, w

N_BITS, LAYERS, ETA_REQ, M_SEED, M0_REF = 12, 3, 0.5, 12345, 0.0
STEPS, TOL, PATIENCE = 3000, 1e-8, 30
VSEEDS = (42, 142, 242, 342, 442)
QAOA_SEEDS = (442, 7, 2024, 31337)
ARMS = ("star", "L1")

OUT = "data_qaoa/_multistart_n12p3"
# Hard guard: this script must never be able to clobber the finished paper grid.
PROTECTED = ("PO_optimality_maqaoa_percentile_interp_M00",
             "PO_optimality_maqaoa_percentile_interp_L1_M00")
assert not any(p in OUT for p in PROTECTED), f"refusing to write into a sweep folder: {OUT}"
os.makedirs(OUT, exist_ok=True)

configs = [(v, qs, mc) for v in VSEEDS for qs in QAOA_SEEDS for mc in ARMS]
nw = int(os.environ.get("NWORKERS", 1))
wk = int(os.environ.get("WORKER", 0))
configs = configs[wk::nw]
print(f"worker {wk}/{nw}: {len(configs)} configs", flush=True)

for vseed, qaoa_seed, mc in configs:
    fname = f"{OUT}/{mc}_n{N_BITS}_p{LAYERS}_v{vseed}_s{qaoa_seed}.pkl"
    if os.path.exists(fname):
        print(f"{os.path.basename(fname)} exists, skipping", flush=True)
        continue

    N_idx = int(np.where(N_stocks == N_BITS // w)[0][0])
    t0 = timer()
    np.random.seed(M_SEED)          # pin M_method_opt's feasible-spectrum sampling
    random.seed(M_SEED)
    r = maqaoa_sampling_oneshot(
        "PO", N_idx, ("PO_small", vseed), "optimality", ETA_REQ, LAYERS,
        qaoa_samples=1000, qaoa_steps=STEPS, tolerance_opt=TOL,
        Ef_from_percentile=True, Ef_percentile=75, warm_start=True,
        M_choice=mc, M0_ref=M0_REF, qaoa_seed=qaoa_seed, patience=PATIENCE,
        print_time=False,
    )

    E_obj, E_pen = cmpML1.spectra(N_BITS, vseed)
    feas = np.isclose(E_pen, 0)
    probs = np.asarray(r["probability_states"], float)
    P_feas = probs[feas].sum()
    lo, hi, mu = E_obj[feas].min(), E_obj[feas].max(), E_obj[feas].mean()
    if P_feas > 1e-12:
        wf = probs[feas] / P_feas
        mean_feas = float(wf @ E_obj[feas])
        AR = float((mean_feas - hi) / (lo - hi))       # 1 = feasible optimum, higher is better
    else:
        mean_feas, AR = np.nan, np.nan
    share_entry, share_feas = cmpML1.objective_share(N_BITS, vseed, r["M_used"])

    rec = dict(arm=mc, bits=N_BITS, layers=LAYERS, vseed=vseed, qaoa_seed=qaoa_seed,
               steps=STEPS, tol=TOL, patience=PATIENCE, M0_ref=M0_REF, M_seed=M_SEED,
               M_used=float(r["M_used"]), M_star=float(r["M_star"]), M_L1=float(r["M_L1"]),
               s_conv=float(r["s_conv"]), eta_eff=float(r["eta_eff_ex"]),
               P_feas=float(P_feas), mean_feas=mean_feas, mean_all=float(probs @ E_obj),
               AR=AR, AR_unif=float((mu - hi) / (lo - hi)),
               share_feas=float(share_feas), share_entry=float(share_entry),
               best_cost=float(r["best_cost"]), E_f=float(r["E_f"]),
               probs=probs, secs=timer() - t0)
    pickle.dump(rec, open(fname, "wb"))
    print(f"{mc:4s} v={vseed:3d} s={qaoa_seed:5d}:  AR={AR:.4f} (blind {rec['AR_unif']:.4f})  "
          f"cost={rec['best_cost']:+.8f}  P_feas={P_feas:.4f}  eta={rec['eta_eff']:.4f}  "
          f"[{rec['secs']:.0f}s]", flush=True)

print(f"worker {wk} done", flush=True)
