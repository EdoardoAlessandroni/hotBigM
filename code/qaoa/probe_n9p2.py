"""Why does INTERP p=1 -> p=2 destroy objective quality at n=9, and can a hyperparameter fix it?

The defect, from the finished M0=0 grid (M* arm, AR = (E - E_max)/(E_min - E_max), 1 = optimal):

    vseed:      42       142      242      342      442
    p=1      0.4293   0.6492   0.4022   0.9978   0.9893
    p=2      0.4293   0.6596   0.4022   0.7501   0.5805     <- collapses on 342 and 442 only
    p=3      0.6029   0.8139   0.6100   1.0000   1.0000     <- fully recovers

So it is NOT an average degradation: it hits exactly the two instances where p=1 had already found a
near-perfect solution, and p=3 undoes it. The whole grid ran under the OLD stopping rule (patience
effectively 3, comparing the oscillating per-step cost between adjacent checkpoints), which stopped
the n=9 p=2 run at step 369/3000. Separately, n=15 is monotone in depth under the new rule and
non-monotone under the old one. Hypothesis: the p=2 stage is simply under-trained.

Arms, all on the M* penalty, M0=0, 5 vseeds, p in {1, 2} (INTERP trains 1..p, so p=2 re-trains p=1):
  base      new stopping rule (patience 30), everything else as the grid   -- tests the hypothesis
  seed7     + qaoa_seed 7      -- is the p=2 basin an artifact of the circuit init?
  seed2024  + qaoa_seed 2024
  tol12     + tolerance_opt 1e-12, 6000 steps  -- can it be driven out by sheer training?

Writes to data_qaoa/_probe_n9p2/, NEVER to the sweep folders: the paper grid must not be touched.
"""

import os
import pickle
import random
from timeit import default_timer as timer

import numpy as np

import analyse_M_L1_vs_star as cmpML1
from qaoa_sampling_oneshot import maqaoa_sampling_oneshot, N_stocks, w

N_BITS, ETA_REQ, M_SEED, M0_REF = 9, 0.5, 12345, 0.0
VSEEDS = (42, 142, 242, 342, 442)
LAYERS = (1, 2)
OUT = "data_qaoa/_probe_n9p2"
os.makedirs(OUT, exist_ok=True)

#     name        qaoa_seed  steps  tol     patience
VARIANTS = {
    "base":     dict(qaoa_seed=442,  steps=3000, tol=1e-8,  patience=30),
    "seed7":    dict(qaoa_seed=7,    steps=3000, tol=1e-8,  patience=30),
    "seed2024": dict(qaoa_seed=2024, steps=3000, tol=1e-8,  patience=30),
    "tol12":    dict(qaoa_seed=442,  steps=6000, tol=1e-12, patience=50),
}

configs = [(name, v, p) for name in VARIANTS for v in VSEEDS for p in LAYERS]
nw = int(os.environ.get("NWORKERS", 1))
wk = int(os.environ.get("WORKER", 0))
configs = configs[wk::nw]
print(f"worker {wk}/{nw}: {len(configs)} configs", flush=True)

for name, vseed, layers in configs:
    cfg = VARIANTS[name]
    fname = f"{OUT}/{name}_n{N_BITS}_p{layers}_v{vseed}.pkl"
    if os.path.exists(fname):
        print(f"{os.path.basename(fname)} exists, skipping", flush=True)
        continue

    N_idx = int(np.where(N_stocks == N_BITS // w)[0][0])
    t0 = timer()
    np.random.seed(M_SEED)          # pin M_method_opt's feasible-spectrum sampling, as in the grid
    random.seed(M_SEED)
    r = maqaoa_sampling_oneshot(
        "PO", N_idx, ("PO_small", vseed), "optimality", ETA_REQ, layers,
        qaoa_samples=1000, qaoa_steps=cfg["steps"], tolerance_opt=cfg["tol"],
        Ef_from_percentile=True, Ef_percentile=75, warm_start=True,
        M_choice="star", M0_ref=M0_REF, qaoa_seed=cfg["qaoa_seed"],
        patience=cfg["patience"], print_time=False,
    )

    E_obj, E_pen = cmpML1.spectra(N_BITS, vseed)
    feas = np.isclose(E_pen, 0)
    probs = np.asarray(r["probability_states"], float)
    P_feas = probs[feas].sum()
    lo, hi, mu = E_obj[feas].min(), E_obj[feas].max(), E_obj[feas].mean()
    if P_feas > 1e-12:
        wf = probs[feas] / P_feas
        mean_feas = float(wf @ E_obj[feas])
        AR = float((mean_feas - hi) / (lo - hi))          # 1 = optimal, 0 = worst feasible
    else:
        mean_feas, AR = np.nan, np.nan

    rec = dict(variant=name, bits=N_BITS, layers=layers, vseed=vseed, **cfg,
               M_used=float(r["M_used"]), s_conv=float(r["s_conv"]),
               eta_eff=float(r["eta_eff_ex"]), P_feas=float(P_feas),
               mean_feas=mean_feas, AR=AR, AR_unif=float((mu - hi) / (lo - hi)),
               best_cost=float(r["best_cost"]), probs=probs, secs=timer() - t0)
    pickle.dump(rec, open(fname, "wb"))
    print(f"{name:9s} p={layers} v={vseed}:  AR={AR:.4f}  (blind {rec['AR_unif']:.4f})  "
          f"P_feas={P_feas:.4f}  eta={rec['eta_eff']:.4f}  cost={rec['best_cost']:+.6f}  "
          f"[{rec['secs']:.0f}s]", flush=True)

print(f"worker {wk} done", flush=True)
