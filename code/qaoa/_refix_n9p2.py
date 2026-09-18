"""Priority 2: re-run the n=9 p=2 anomaly with the FIXED stopping rule (no monkey-patching).

Uses the pipeline as it now stands -- best-cost tracked, window-total improvement, patience=30
(a 300-step window instead of 30). M pinned and circuit seed 442, exactly as the sweep, so the
only thing that changed is when the optimizer is allowed to give up.

Reference under the old rule: stopped at step 369/3000, <E_obj>_feas = +0.0881, gap 0.7823.
The p=1 run on the same instance reached -0.807 / gap 0.0070, so a far better solution exists.
Also prints wall time, to calibrate how much the fix costs before committing to a full re-run.
"""
import pickle, random, sys, os
import numpy as np
from timeit import default_timer as timer
import analyse_M_L1_vs_star as cmpML1
from qaoa_sampling_oneshot import maqaoa_sampling_oneshot, N_stocks, w

VSEED, ARM = int(sys.argv[1]), sys.argv[2]
N_BITS, LAYERS = 9, 2
N_idx = int(np.where(N_stocks == N_BITS // w)[0][0])

t0 = timer()
np.random.seed(12345); random.seed(12345)
r = maqaoa_sampling_oneshot("PO", N_idx, ("PO_small", VSEED), "optimality", 0.5, LAYERS,
                            qaoa_samples=1000, qaoa_steps=3000, tolerance_opt=1e-8,
                            Ef_from_percentile=True, Ef_percentile=75, warm_start=True,
                            M_choice=ARM, M0_ref=0.0, qaoa_seed=442, print_time=False)
secs = timer() - t0

Eo, Ep = cmpML1.spectra(N_BITS, VSEED)
f = np.isclose(Ep, 0)
pr = np.asarray(r["probability_states"], float)
wf = pr[f] / pr[f].sum()
mean_f = float(wf @ Eo[f])
lo, hi, mu = Eo[f].min(), Eo[f].max(), Eo[f].mean()
rec = dict(vseed=VSEED, arm=ARM, patience=r["patience"], M=float(r["M_used"]), mean_feas=mean_f,
           gap_feas=float((mean_f - lo) / (mu - lo)), AR_f=float((mean_f - lo) / (hi - lo)),
           best_cost=float(r["best_cost"]), eta=float(r["eta_eff_ex"]),
           P_feas=float(pr[f].sum()), secs=secs)
os.makedirs("data_qaoa/_refix_n9p2", exist_ok=True)
pickle.dump(rec, open(f"data_qaoa/_refix_n9p2/{ARM}_v{VSEED}.pkl", "wb"))
print(f"\nFIXED n=9 p=2 {ARM} v={VSEED}:  M={rec['M']:.3f}  <E_obj>_feas={mean_f:+.5f}  "
      f"gap={rec['gap_feas']:.4f}  AR_f={rec['AR_f']:.4f}  eta={rec['eta']:.4f}  "
      f"P_feas={rec['P_feas']:.4f}  [{secs:.0f}s]", flush=True)
