"""Fold the already-computed M*, M0=0, p=3 runs from `_span_expanded` into the M00 sweep folder.

`run_span_expanded.py` and `run_PO_optimality_interp_M00.py` issue the IDENTICAL call for the cells
they share -- M0_ref=0.0, M_choice="star", 3000 steps, tol 1e-8, eta_req 0.5, E_f=P75, INTERP warm
start, qaoa_seed 442, 1000 samples, and M pinned with np.random.seed(12345) -- so re-running them
would reproduce the same numbers at ~10 CPU-hours of cost. Only p=3 is imported: p=5 is outside the
new grid and is left in `_span_expanded`, where the ladder/share panels read it from.

The imported records carry every scoring field the notebook uses, but NOT `probs` / `best_pars` /
`Q` (the expanded grid never stored them). They are set to None and flagged with
`imported_from`, so any cell needing the full output distribution must check rather than assume.
"""

import glob
import os
import pickle

import numpy as np

SRC = "data_qaoa/_span_expanded"
DST = "data_qaoa/PO_optimality_maqaoa_percentile_interp_M00"
os.makedirs(DST, exist_ok=True)

n_new = n_skip = 0
for f in sorted(glob.glob(f"{SRC}/m00_n*_p*_v*.pkl")):
    r = pickle.load(open(f, "rb"))
    if r["layers"] not in (3,):        # p=5 is out of the sweep's scope; M_L1 has no counterpart
        continue
    assert r["M0_ref"] == 0.0 and r["steps"] == 3000 and r["tol"] == 1e-8, \
        f"{f} is not the configuration the M00 sweep expects: {r['M0_ref']=} {r['steps']=} {r['tol']=}"

    out = (f"{DST}/test_PO_bits_{r['bits']}_vseed_{r['vseed']}_layers_{r['layers']}"
           f"_etareq_{r['eta_req'] if 'eta_req' in r else 0.5}.txt")
    if os.path.exists(out):
        n_skip += 1
        continue

    rec = {
        "eta_eff": r["eta_eff"], "M_used": r["M_used"], "M_star": r["M_used"], "M_L1": np.nan,
        "M_choice": "star", "s_conv": r["s_conv"],
        "probs": None, "best_pars": None,                      # not stored by the expanded grid
        "Q": None, "Q_const": None, "Q_norm": None, "Q_const_norm": None,
        "vseed": int(r["vseed"]), "eta_req": 0.5,
        "layers": int(r["layers"]), "bits": int(r["bits"]),
        "E_f": r["E_f"], "Ef_from_percentile": True, "Ef_percentile": 75, "warm_start": True,
        "M0_ref": 0.0, "steps": r["steps"], "tol": r["tol"], "M_seed": 12345,
        "P_feas": r["P_feas"], "gap_feas": r["gap_feas"],
        "mean_all": r["mean_all"], "mean_feas": r["mean_feas"], "med_feas": r["med_feas"],
        "share_feas": r["share_feas"], "share_entry": r["share_entry"],
        "best_cost": r["best_cost"], "secs": r["secs"],
        "imported_from": os.path.basename(f),
    }
    pickle.dump(rec, open(out, "wb"))
    n_new += 1
    print(f"imported n={r['bits']:2d} p={r['layers']} v={r['vseed']}  M={r['M_used']:8.4g}  "
          f"gap={r['gap_feas']:.4f}  (saved {r['secs']:.0f}s)")

print(f"\n{n_new} imported, {n_skip} already present in {DST}")
