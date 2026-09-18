"""Does a smaller span proxy restore the circuit's sensitivity to the objective?

The span proxy is  s_conv(M0) = (UB_obj - E_LB_obj) + M0 * UB_pen,  and the pipeline evaluates it at
M0 = L1_norm_hot(..., T_norm, eta) ~ 1e2, which makes the penalty term dominate by ~3 decades. Since
temperature_unnormalized = s_conv * T_norm and M ~ T * c_n, that inflates BOTH M* and M_L1 to ~1e6,
where the objective is ~1e-7 of the trained Hamiltonian and therefore invisible (see
fixed_point_M.py: the self-consistent version of this loop has no positive root, loop gain ~1e4).

Three span variants, crossed with both penalty rules:

    M0 = "L1"  the pipeline's own choice                      (already on disk, not re-run here)
    M0 = 1     objective span + penalty span at trivial weight -- s_conv = A + B
    M0 = 0     objective span only                             -- s_conv = A, M-free by construction

Everything else is held at the existing sweep's settings (E_f = P75, INTERP warm start, seed 442,
500 steps, tol 1e-4, eta_req 0.5) so the new cells drop straight into the same comparison as the
120-run M*/M_L1 study.

Run as several workers:  CONFIGS="0,1,2" python run_span_variants.py
"""

import os
import pickle
from timeit import default_timer as timer

import numpy as np

import analyse_M_L1_vs_star as cmpML1
from qaoa_sampling_oneshot import maqaoa_sampling_oneshot, N_stocks, w

M0_REFS = tuple(float(x) for x in os.environ.get("M0_REFS", "0,1").split(","))
BITS = tuple(int(x) for x in os.environ.get("BITS", "9,12").split(","))
LAYERS = tuple(int(x) for x in os.environ.get("LAYERS", "1,3").split(","))
VSEEDS = (42, 142, 242)
M_CHOICES = tuple(os.environ.get("M_CHOICES", "star,L1").split(","))
# Training budget. The default reproduces the existing sweep; raising it is how we separate "M is too
# large for the objective to matter" from "the circuit was simply not trained long enough to exploit it".
STEPS = int(os.environ.get("STEPS", 500))
TOL = float(os.environ.get("TOL", 1e-4))
ETA_REQ = 0.5
OUT = "data_qaoa/_span_variants"
os.makedirs(OUT, exist_ok=True)

configs = [(m0, n, p, v, mc)
           for m0 in M0_REFS for n in BITS for p in LAYERS for v in VSEEDS for mc in M_CHOICES]
if os.environ.get("CONFIGS"):
    picked = [int(i) for i in os.environ["CONFIGS"].split(",")]
    configs = [configs[i] for i in picked]
    print(f"worker handling {len(picked)} configs", flush=True)

for m0, n_bits, layers, vseed, mc in configs:
    tag = "" if (STEPS, TOL) == (500, 1e-4) else f"_s{STEPS}_t{TOL:.0e}"
    fname = f"{OUT}/m0{m0:g}_{mc}_n{n_bits}_p{layers}_v{vseed}{tag}.pkl"
    if os.path.exists(fname):
        print(f"{os.path.basename(fname)} exists, skipping", flush=True)
        continue

    N_idx = int(np.where(N_stocks == n_bits // w)[0][0])
    t0 = timer()
    res = maqaoa_sampling_oneshot(
        "PO", N_idx, ("PO_small", vseed), "optimality", ETA_REQ, layers,
        qaoa_samples=1000, qaoa_steps=STEPS, tolerance_opt=TOL,
        Ef_from_percentile=True, Ef_percentile=75,
        warm_start=True, M_choice=mc, M0_ref=m0,
        print_time=False,
    )

    # same scoring as the M_L1 vs M* study, on the exact output distribution
    E_obj, E_pen = cmpML1.spectra(n_bits, vseed)
    feas = np.isclose(E_pen, 0)
    probs = np.asarray(res["probability_states"], float)
    P_feas = probs[feas].sum()
    if P_feas > 1e-12:
        wf = probs[feas] / P_feas
        mean_feas = float(wf @ E_obj[feas])
        E_min_f, E_mean_f = E_obj[feas].min(), E_obj[feas].mean()
        gap_feas = float((mean_feas - E_min_f) / (E_mean_f - E_min_f))
    else:
        mean_feas, gap_feas = np.nan, np.nan
    mean_all = float(probs @ E_obj)
    share_entry, share_feas = cmpML1.objective_share(n_bits, vseed, res["M_used"])

    rec = dict(M0_ref=m0, M_choice=mc, bits=n_bits, layers=layers, vseed=vseed,
               steps=STEPS, tol=TOL,
               M_used=float(res["M_used"]), M_star=float(res["M_star"]),
               M_L1=float(res["M_L1"]), s_conv=float(res["s_conv"]),
               eta_eff=float(res["eta_eff_ex"]), P_feas=float(P_feas),
               mean_all=mean_all, mean_feas=mean_feas, gap_feas=gap_feas,
               share_entry=float(share_entry), share_feas=float(share_feas),
               E_f=float(res["E_f"]), secs=timer() - t0)
    pickle.dump(rec, open(fname, "wb"))
    print(f"M0={m0:g} {mc:4s} n={n_bits} p={layers} v={vseed}:  M={rec['M_used']:10.4g}  "
          f"share={rec['share_feas']:.2e}  P_feas={rec['P_feas']:.4f}  "
          f"eta={rec['eta_eff']:.4f}  gap_feas={rec['gap_feas']:.4f}  [{rec['secs']:.0f}s]",
          flush=True)

print("worker done", flush=True)
