"""TEMPORARY TEST -- does artificially shrinking the span proxy buy objective quality at n=12?

NOT a pipeline change. `qaoa_sampling_oneshot.py` is untouched; this script monkey-patches
`span_proxy` in the imported module for the duration of the run, so the production code path is
exactly what it was. Delete this file and nothing else is affected.

Rationale. s_conv enters the pipeline only through temperature_unnormalized = s_conv * T_norm,
which sets beta, which sets M. So dividing s_conv by D divides M by ~D and multiplies the
objective's share of the trained Hamiltonian by ~D. Two things motivate D > 1:

  * measured slack: span_proxy bounds the objective span over the whole Boolean cube, but the
    sampler only occupies the feasible subspace, which is 1.69x narrower at n=12 (2.24x at n=15).
    So D ~ 1.7 is "free" in the sense that a constrained bound would deliver it honestly.
  * calibration: separation at n=6 needed share >~ 7e-3. n=12 p=3 sits at 2.0e-3, so D = 4 lands
    at 8e-3 and D = 8 at 1.6e-2, comparable to n=6 p=3 (1.06e-2) which separated strongly.

Testing at p=3 rather than p=7 because share responds to D and to p the same way, and p=3 costs
4.7x less (INTERP cost ~ p(p+1)/2). If the effect is real, D=8 at p=3 must show it.

Everything else matches the M00 sweep: M0=0, 3000 steps, tol 1e-8, E_f=P75, eta_req 0.5, INTERP
warm start, qaoa_seed 442, M pinned at seed 12345. D=1 is not re-run -- it is the existing sweep.

Run as several workers:  WORKER=0 NWORKERS=5 python test_span_divisor.py
"""

import os
import pickle
import random
from timeit import default_timer as timer

import numpy as np

import analyse_M_L1_vs_star as cmpML1
import qaoa_sampling_oneshot as qso
from qaoa_sampling_oneshot import maqaoa_sampling_oneshot, N_stocks, w

STEPS, TOL, ETA_REQ, M_SEED = 3000, 1e-8, 0.5, 12345
N_BITS, LAYERS = 12, 3
VSEEDS = (42, 142, 242)
DIVISORS = (2.0, 4.0, 8.0)
ARMS = ("star", "L1")
OUT = "data_qaoa/_span_divisor"
os.makedirs(OUT, exist_ok=True)

_orig_span_proxy = qso.span_proxy


def patch_span(divisor):
    """Divide s_conv by `divisor`, leaving the reported bounds untouched."""
    def patched(*args, **kwargs):
        s_conv, UB_obj, UB_pen, method = _orig_span_proxy(*args, **kwargs)
        return s_conv / divisor, UB_obj, UB_pen, method
    qso.span_proxy = patched


configs = [(d, v, mc) for d in DIVISORS for v in VSEEDS for mc in ARMS]
nw = int(os.environ.get("NWORKERS", 1))
wk = int(os.environ.get("WORKER", 0))
configs = configs[wk::nw]
print(f"worker {wk}/{nw}: {len(configs)} configs", flush=True)

for divisor, vseed, mc in configs:
    fname = f"{OUT}/d{divisor:g}_{mc}_n{N_BITS}_p{LAYERS}_v{vseed}.pkl"
    if os.path.exists(fname):
        print(f"{os.path.basename(fname)} exists, skipping", flush=True)
        continue

    N_idx = int(np.where(N_stocks == N_BITS // w)[0][0])
    patch_span(divisor)
    try:
        t0 = timer()
        np.random.seed(M_SEED)
        random.seed(M_SEED)
        r = maqaoa_sampling_oneshot(
            "PO", N_idx, ("PO_small", vseed), "optimality", ETA_REQ, LAYERS,
            qaoa_samples=1000, qaoa_steps=STEPS, tolerance_opt=TOL,
            Ef_from_percentile=True, Ef_percentile=75,
            warm_start=True, M_choice=mc, M0_ref=0.0, print_time=False,
        )
    finally:
        qso.span_proxy = _orig_span_proxy      # never leave the module patched

    E_obj, E_pen = cmpML1.spectra(N_BITS, vseed)
    feas = np.isclose(E_pen, 0)
    probs = np.asarray(r["probability_states"], float)
    P_feas = probs[feas].sum()
    lo, hi = E_obj[feas].min(), E_obj[feas].max()
    if P_feas > 1e-12:
        wf = probs[feas] / P_feas
        mean_feas = float(wf @ E_obj[feas])
        gap_feas = float((mean_feas - lo) / (E_obj[feas].mean() - lo))
        AR_f = float((mean_feas - lo) / (hi - lo))
    else:
        mean_feas = gap_feas = AR_f = np.nan
    share_entry, share_feas = cmpML1.objective_share(N_BITS, vseed, r["M_used"])

    rec = dict(divisor=divisor, arm=mc, bits=N_BITS, layers=LAYERS, vseed=vseed,
               steps=STEPS, tol=TOL, M0_ref=0.0, M_seed=M_SEED,
               M_used=float(r["M_used"]), s_conv=float(r["s_conv"]),
               eta_eff=float(r["eta_eff_ex"]), P_feas=float(P_feas),
               mean_all=float(probs @ E_obj), mean_feas=mean_feas,
               gap_feas=gap_feas, AR_f=AR_f,
               share_feas=float(share_feas), share_entry=float(share_entry),
               best_cost=float(r["best_cost"]), E_f=float(r["E_f"]),
               probs=probs, secs=timer() - t0)
    pickle.dump(rec, open(fname, "wb"))
    print(f"D={divisor:g} {mc:4s} v={vseed}:  M={rec['M_used']:8.3f}  share={rec['share_feas']:.2e}  "
          f"P_feas={rec['P_feas']:.4f}  eta={rec['eta_eff']:.4f}  AR_f={rec['AR_f']:.4f}  "
          f"[{rec['secs']:.0f}s]", flush=True)

print(f"worker {wk} done", flush=True)
