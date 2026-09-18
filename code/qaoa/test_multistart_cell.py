"""Multi-start over the circuit init for ONE (n, p) cell, both arms. Generalises
`test_multistart_n12p3.py`, whose docstring has the full rationale.

    N_BITS=6 LAYERS=3 NWORKERS=14 WORKER=0 python test_multistart_cell.py

Why this exists. The n=12 p=3 run settled that (a) picking the init with the lowest trained cost
selects the AR-oracle's choice 10/10, and (b) multi-start lifts the cell well clear of blind. But it
also showed the M* - M_L1 gap SHRINKING from +0.046 to +0.026, and -- the part that matters -- that
at qaoa_seed 2024 alone the gap is -0.007, i.e. M_L1 is marginally ahead. At that cell the entire M*
advantage is inside the init-to-init variation.

n=12 p=3 was always the weakest cell in the study (gap +0.046 against +0.172 at n=6 p=3), so this
may say nothing about where the effect is strong. That is exactly what has to be checked before
committing ~190 CPU-hours to a multi-start re-run of the whole grid: if the gap survives at n=6 p=3
the claim is real and n=12 was simply marginal; if it collapses there too, the headline result is an
artifact of qaoa_seed=442 and the paper needs rewriting.

Writes to data_qaoa/_multistart_n{N}p{P}/ and refuses to touch the finished sweep folders.
"""

import os
import pickle
import random
from timeit import default_timer as timer

import numpy as np

import analyse_M_L1_vs_star as cmpML1
from qaoa_sampling_oneshot import maqaoa_sampling_oneshot, N_stocks, w

# CELLS="6:1,6:2,9:1" runs several (n, p) cells in one process so a worker pool spans the whole
# batch rather than draining one cell at a time. N_BITS/LAYERS remain for the single-cell form.
if os.environ.get("CELLS"):
    CELLS = [tuple(int(x) for x in c.split(":")) for c in os.environ["CELLS"].split(",")]
else:
    CELLS = [(int(os.environ.get("N_BITS", 6)), int(os.environ.get("LAYERS", 3)))]
ETA_REQ, M_SEED, M0_REF = 0.5, 12345, 0.0
STEPS, TOL, PATIENCE = 3000, 1e-8, 30
VSEEDS = (42, 142, 242, 342, 442)
# SEEDS / ARMS are overridable so a targeted backfill (e.g. "just seed 442, just the M* arm, to
# recover circuit_params for the imported p=3 runs") reuses this exact code path.
QAOA_SEEDS = tuple(int(x) for x in os.environ.get("SEEDS", "442,7,2024,31337").split(","))
ARMS = tuple(os.environ.get("ARMS", "star,L1").split(","))

PROTECTED = ("PO_optimality_maqaoa_percentile_interp_M00",
             "PO_optimality_maqaoa_percentile_interp_L1_M00")
for _n, _p in CELLS:
    _o = f"data_qaoa/_multistart_n{_n}p{_p}"
    assert not any(q in _o for q in PROTECTED), f"refusing to write into a sweep folder: {_o}"
    os.makedirs(_o, exist_ok=True)

configs = [(n, p, v, qs, mc) for n, p in CELLS
           for v in VSEEDS for qs in QAOA_SEEDS for mc in ARMS]
nw = int(os.environ.get("NWORKERS", 1))
wk = int(os.environ.get("WORKER", 0))
configs = configs[wk::nw]
print(f"cells {CELLS}  worker {wk}/{nw}: {len(configs)} configs", flush=True)

for N_BITS, LAYERS, vseed, qaoa_seed, mc in configs:
    OUT = f"data_qaoa/_multistart_n{N_BITS}p{LAYERS}"
    fname = f"{OUT}/{mc}_n{N_BITS}_p{LAYERS}_v{vseed}_s{qaoa_seed}.pkl"
    if os.path.exists(fname):
        # earlier batches did not store circuit_params; recompute those so dataset A/B carry the
        # trained parameters. Anything already complete is left alone.
        try:
            _old = pickle.load(open(fname, "rb"))
        except Exception:
            _old = {}
        if _old.get("circuit_params") is not None:
            print(f"{os.path.basename(fname)} complete, skipping", flush=True)
            continue
        print(f"{os.path.basename(fname)} has no circuit_params -- recomputing", flush=True)

    N_idx = int(np.where(N_stocks == N_BITS // w)[0][0])
    t0 = timer()
    np.random.seed(M_SEED)
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
               # full provenance: the trained circuit, the QUBO it was trained on, and every knob
               circuit_params=r["circuit_params"], QUBO=r["QUBO"],
               eta_req=ETA_REQ, Ef_percentile=75, Ef_from_percentile=True, warm_start=True,
               qaoa_samples=1000, M_choice=mc,
               probs=probs, secs=timer() - t0)
    pickle.dump(rec, open(fname, "wb"))
    print(f"{mc:4s} v={vseed:3d} s={qaoa_seed:5d}:  AR={AR:.4f} (blind {rec['AR_unif']:.4f})  "
          f"cost={rec['best_cost']:+.8f}  eta={rec['eta_eff']:.4f}  [{rec['secs']:.0f}s]", flush=True)

print(f"worker {wk} done", flush=True)
