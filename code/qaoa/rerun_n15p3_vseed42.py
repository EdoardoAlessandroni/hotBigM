"""Re-run the one cell that collapsed in the percentile sweep, after the `patience` fix.

n_bits=15, layers=3, vseed=42 scored eta_eff = 0.006 under the old stopping rule, which
broke out of Adam on the FIRST flat 10-step window (step 139; every other run in the sweep
ran 259-500). `patience` was declared but never read; it now requires 3 CONSECUTIVE flat
windows.

The optimizer seed is deliberately left at the sweep default (442), so the stopping rule is
the only thing that changed w.r.t. the run that produced the bad file. The old file is kept
as *.old_patience_bug, and is only replaced if the new run actually clears eta_req = 0.5.
"""

import os
import pickle
import shutil

import numpy as np

from qaoa_sampling_oneshot import maqaoa_sampling_oneshot

OUT = "data_qaoa/PO_optimality_maqaoa_percentile/test_PO_bits_15_vseed_42_layers_3_etareq_0.5.txt"
ETA_REQ = 0.5
n_bits, vseed, layers = 15, 42, 3

old = pickle.load(open(OUT, "rb"))
print(f"old file: eta_eff = {old['eta_eff']:.4f}, M* = {old['M_star']:.5g}, E_f = {old['E_f']:.5f}\n",
      flush=True)

results = maqaoa_sampling_oneshot("PO", 3, ("PO_small", vseed), "optimality", ETA_REQ, layers,
                                  qaoa_samples=1000, qaoa_steps=500,
                                  Ef_from_percentile=True, Ef_percentile=75,
                                  qaoa_seed=442)   # sweep default, unchanged

Q, const = results["QUBO"]
Q = np.asarray(Q, float)
assert Q.shape == (n_bits, n_bits)

Q_norm     = (Q     / results["s_conv"]) / results["norm_final"]
const_norm = (const / results["s_conv"]) / results["norm_final"]

res = {
    "eta_eff":   float(results["eta_eff_ex"]),
    "best_pars": results["circuit_params"],
    "M_star":    float(results["M_star"]),
    "Q":            Q,
    "Q_const":      float(const),
    "Q_norm":       Q_norm,
    "Q_const_norm": float(const_norm),
    "vseed":     int(vseed),
    "eta_req":   float(ETA_REQ),
    "layers":    int(layers),
    "bits":      n_bits,
    "E_f":               float(results["E_f"]),
    "Ef_from_percentile": True,
    "Ef_percentile":      75,
}

print(f"\nnew run: eta_eff = {res['eta_eff']:.4f}  (old {old['eta_eff']:.4f})", flush=True)
print(f"         E_f = {res['E_f']:.5f}  (old {old['E_f']:.5f})   "
      f"M* = {res['M_star']:.5g}  (old {old['M_star']:.5g})", flush=True)

if res["eta_eff"] >= ETA_REQ:
    shutil.copy2(OUT, OUT + ".old_patience_bug")
    with open(OUT, "wb") as f:
        pickle.dump(res, f)
    print(f"\nSUBSTITUTED. old file kept at {OUT}.old_patience_bug", flush=True)
else:
    print(f"\nNOT SUBSTITUTED: new eta_eff {res['eta_eff']:.4f} still below eta_req {ETA_REQ}. "
          f"File left untouched.", flush=True)
