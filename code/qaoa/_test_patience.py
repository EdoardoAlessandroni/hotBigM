"""Is the n=9 p=2 plateau escapable with a longer horizon, or a true local minimum?

TEMPORARY TEST -- monkey-patches run_ma_qaoa_with_solution to force `patience`, which the pipeline
never forwards (it is frozen at 3 in the signature, i.e. a 30-step window). Nothing is written to
qaoa_sampling_oneshot.py.

M is pinned, the Hamiltonian is identical to the sweep's, and the circuit seed is the sweep's 442.
The only variable is how long the optimizer is allowed to sit on a flat region before giving up.

Reference: the sweep (patience=3) stopped at step 369 with <E_obj>_feas = +0.0881, gap 0.7823.
p=1 on the same instance reached -0.807 / gap 0.0070, so there IS a far better solution to find.
"""
import os, pickle, random, sys
import numpy as np
import analyse_M_L1_vs_star as cmpML1
import qaoa_sampling_oneshot as qso
from qaoa_sampling_oneshot import maqaoa_sampling_oneshot, N_stocks, w

PATIENCE = int(sys.argv[1])
VSEED = int(sys.argv[2])
N_BITS, LAYERS = 9, 2

_orig = qso.run_ma_qaoa_with_solution
def patched(*a, **k):
    k["patience"] = PATIENCE
    return _orig(*a, **k)
qso.run_ma_qaoa_with_solution = patched

N_idx = int(np.where(N_stocks == N_BITS // w)[0][0])
np.random.seed(12345); random.seed(12345)
r = maqaoa_sampling_oneshot("PO", N_idx, ("PO_small", VSEED), "optimality", 0.5, LAYERS,
                            qaoa_samples=1000, qaoa_steps=3000, tolerance_opt=1e-8,
                            Ef_from_percentile=True, Ef_percentile=75, warm_start=True,
                            M_choice="star", M0_ref=0.0, qaoa_seed=442, print_time=False)

Eo, Ep = cmpML1.spectra(N_BITS, VSEED)
f = np.isclose(Ep, 0)
p_ = np.asarray(r["probability_states"], float)
wf = p_[f] / p_[f].sum()
mean_f = float(wf @ Eo[f])
lo, hi, mu = Eo[f].min(), Eo[f].max(), Eo[f].mean()
rec = dict(patience=PATIENCE, vseed=VSEED, M=float(r["M_used"]), mean_feas=mean_f,
           gap_feas=float((mean_f - lo) / (mu - lo)), AR_f=float((mean_f - lo) / (hi - lo)),
           best_cost=float(r["best_cost"]), eta=float(r["eta_eff_ex"]), P_feas=float(p_[f].sum()))
os.makedirs("data_qaoa/_patience", exist_ok=True)
pickle.dump(rec, open(f"data_qaoa/_patience/pat{PATIENCE}_v{VSEED}.pkl", "wb"))
print(f"\nRESULT patience={PATIENCE} v={VSEED}:  M={rec['M']:.3f}  <E_obj>_feas={mean_f:+.5f}  "
      f"gap={rec['gap_feas']:.4f}  AR_f={rec['AR_f']:.4f}  best_cost={rec['best_cost']:.6f}  "
      f"eta={rec['eta']:.4f}", flush=True)
