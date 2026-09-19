"""Recover circuit_params for the dataset-A records that have none, under the ORIGINAL stopping rule.

The 18 affected records (M* arm, p=3) were imported from `_span_expanded`, whose recording script
dropped `best_pars`. They must be reproduced, not merely re-run: they were trained before the
2026-09-18 stopping-rule fix, so today's trainer lands on a different local optimum for some of
them. This script uses `qaoa_sampling_legacy` (see make_legacy_module.py) -- today's code with only
the adjacent-difference stopping test restored.

Correctness is CHECKED, not assumed: every re-run is compared against the AR and best_cost already
in the sweep record. A record is only usable for dataset A if it reproduces; anything that does not
is reported so the plot is never silently changed underneath.

    PROBE="6:3:242" python backfill_legacy.py          # single config, for validating the rebuild
    NWORKERS=6 WORKER=$W python backfill_legacy.py     # the whole set

Writes data_qaoa/_legacy_backfill/. Never opens the sweep folders for writing.
"""

import os
import pickle
import random
from timeit import default_timer as timer

import numpy as np

import analyse_M_L1_vs_star as cmpML1
from qaoa_sampling_legacy import maqaoa_sampling_oneshot, N_stocks, w

SWEEP = {"star": "data_qaoa/PO_optimality_maqaoa_percentile_interp_M00",
         "L1":   "data_qaoa/PO_optimality_maqaoa_percentile_interp_L1_M00"}
OUT = "data_qaoa/_legacy_backfill"
PROTECTED = tuple(SWEEP.values())
assert not any(p in OUT for p in PROTECTED), f"refusing to write into a sweep folder: {OUT}"
os.makedirs(OUT, exist_ok=True)

# exactly the protocol `run_span_expanded.py` used; qaoa_seed/patience were left at their defaults
ETA_REQ, M_SEED, M0_REF = 0.5, 12345, 0.0
STEPS, TOL, QAOA_SEED = 3000, 1e-8, 442
ATOL = 1e-6          # what counts as "reproduced"


def sweep_path(arm, n, p, v):
    return f"{SWEEP[arm]}/test_PO_bits_{n}_vseed_{v}_layers_{p}_etareq_0.5.txt"


def find_paramless():
    """Every sweep record with no trained parameters -- derived from the data, not hardcoded."""
    out = []
    for arm in SWEEP:
        for n in (6, 9, 12, 15):
            for p in (1, 2, 3):
                for v in (42, 142, 242, 342, 442):
                    f = sweep_path(arm, n, p, v)
                    if not os.path.exists(f):
                        continue
                    try:
                        r = pickle.load(open(f, "rb"))
                    except Exception:
                        continue
                    if r.get("best_pars") is None:
                        out.append((arm, n, p, v))
    return out


targets = find_paramless()
if os.environ.get("PROBE"):
    n_, p_, v_ = (int(x) for x in os.environ["PROBE"].split(":"))
    targets = [t for t in targets if t[1:] == (n_, p_, v_)]

nw, wk = int(os.environ.get("NWORKERS", 1)), int(os.environ.get("WORKER", 0))
targets = targets[wk::nw]
print(f"worker {wk}/{nw}: {len(targets)} paramless records -> {targets}", flush=True)

for arm, N_BITS, LAYERS, vseed in targets:
    fname = f"{OUT}/{arm}_n{N_BITS}_p{LAYERS}_v{vseed}_s{QAOA_SEED}.pkl"
    if os.path.exists(fname) and pickle.load(open(fname, "rb")).get("circuit_params") is not None:
        print(f"{os.path.basename(fname)} done, skipping", flush=True)
        continue

    old = pickle.load(open(sweep_path(arm, N_BITS, LAYERS, vseed), "rb"))
    N_idx = int(np.where(N_stocks == N_BITS // w)[0][0])
    t0 = timer()
    np.random.seed(M_SEED)
    random.seed(M_SEED)
    r = maqaoa_sampling_oneshot(
        "PO", N_idx, ("PO_small", vseed), "optimality", ETA_REQ, LAYERS,
        qaoa_samples=1000, qaoa_steps=STEPS, tolerance_opt=TOL,
        Ef_from_percentile=True, Ef_percentile=75, warm_start=True,
        M_choice=arm, M0_ref=M0_REF, qaoa_seed=QAOA_SEED,
        print_time=False,
    )

    E_obj, E_pen = cmpML1.spectra(N_BITS, vseed)
    feas = np.isclose(E_pen, 0)
    probs = np.asarray(r["probability_states"], float)
    P_feas = probs[feas].sum()
    lo, hi, mu = E_obj[feas].min(), E_obj[feas].max(), E_obj[feas].mean()
    wf = probs[feas] / P_feas
    mean_feas = float(wf @ E_obj[feas])
    AR = float((mean_feas - hi) / (lo - hi))
    AR_old = float((old["mean_feas"] - hi) / (lo - hi))
    share_entry, share_feas = cmpML1.objective_share(N_BITS, vseed, r["M_used"])

    d_AR = AR - AR_old
    d_M = float(r["M_used"]) - float(old["M_used"])
    ok = abs(d_AR) < ATOL and abs(d_M) < ATOL

    rec = dict(arm=arm, bits=N_BITS, layers=LAYERS, vseed=vseed, qaoa_seed=QAOA_SEED,
               steps=STEPS, tol=TOL, M0_ref=M0_REF, M_seed=M_SEED,
               stopping_rule="legacy_adjacent_diff",     # NOT the current rule -- see module docstring
               provenance="reproduced from _span_expanded settings to recover circuit_params",
               M_used=float(r["M_used"]), M_star=float(r["M_star"]), M_L1=float(r["M_L1"]),
               s_conv=float(r["s_conv"]), eta_eff=float(r["eta_eff_ex"]),
               P_feas=float(P_feas), mean_feas=mean_feas, mean_all=float(probs @ E_obj),
               AR=AR, AR_sweep=AR_old, AR_delta=d_AR, reproduced=bool(ok),
               AR_unif=float((mu - hi) / (lo - hi)),
               share_feas=float(share_feas), share_entry=float(share_entry),
               best_cost=float(r["best_cost"]), best_cost_sweep=old.get("best_cost"),
               E_f=float(r["E_f"]),
               circuit_params=r["circuit_params"], QUBO=r["QUBO"],
               eta_req=ETA_REQ, Ef_percentile=75, Ef_from_percentile=True, warm_start=True,
               qaoa_samples=1000, M_choice=arm,
               probs=probs, secs=timer() - t0)
    pickle.dump(rec, open(fname, "wb"))
    print(f"{'OK  ' if ok else 'DIFF'} {arm:4s} n={N_BITS:2d} p={LAYERS} v={vseed:3d}:  "
          f"AR {AR:.6f} vs sweep {AR_old:.6f} (d={d_AR:+.2e})  "
          f"cost {r['best_cost']:+.8f} vs {old.get('best_cost'):+.8f}  "
          f"dM={d_M:+.2e}  [{rec['secs']:.0f}s]", flush=True)

print(f"worker {wk} done", flush=True)
