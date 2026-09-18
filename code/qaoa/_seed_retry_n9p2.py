"""Is the n=9, p=2 M* degradation a local minimum (seed-dependent) or systematic?

At v=342/442 the p=2 INTERP run converges to a POSITIVE normalized cost while the p=1 run reached a
negative one, with P_feas = 1 in both, i.e. strictly worse on the quantity being minimized. If that
is a bad basin, other circuit seeds should recover the p=1-quality result. M stays pinned so the
Hamiltonian is unchanged; only the initialization moves.
"""
import pickle, random
import numpy as np
import analyse_M_L1_vs_star as cmpML1
from qaoa_sampling_oneshot import maqaoa_sampling_oneshot, N_stocks, w

N_idx = int(np.where(N_stocks == 9 // w)[0][0])
out = {}
for v in (342, 442):
    Eo, Ep = cmpML1.spectra(9, v)
    feas = np.isclose(Ep, 0)
    for seed in (442, 7, 2024):
        np.random.seed(12345); random.seed(12345)     # pin M exactly as the sweep does
        r = maqaoa_sampling_oneshot("PO", N_idx, ("PO_small", v), "optimality", 0.5, 2,
                                    qaoa_samples=1000, qaoa_steps=3000, tolerance_opt=1e-8,
                                    Ef_from_percentile=True, Ef_percentile=75, warm_start=True,
                                    M_choice="star", M0_ref=0.0, qaoa_seed=seed, print_time=False)
        p = np.asarray(r["probability_states"], float)
        wf = p[feas] / p[feas].sum()
        mean_f = float(wf @ Eo[feas])
        gap = (mean_f - Eo[feas].min()) / (Eo[feas].mean() - Eo[feas].min())
        out[(v, seed)] = dict(M=float(r["M_used"]), gap=float(gap), mean_feas=mean_f,
                              best_cost=float(r["best_cost"]), eta=float(r["eta_eff_ex"]))
        print(f"v={v} seed={seed:5d}  M={r['M_used']:7.3f}  <E_obj>_feas={mean_f:+.5f}  "
              f"gap={gap:.4f}  best_cost={r['best_cost']:.3e}  eta={r['eta_eff_ex']:.4f}", flush=True)
pickle.dump(out, open("data_qaoa/_seed_retry_n9p2.pkl", "wb"))
print("\nreference: p=1 reached gap 0.0070 (v=342) and 0.0311 (v=442); "
      "the sweep's p=2 seed-442 runs gave 0.7823 and 1.2226")
