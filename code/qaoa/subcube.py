"""Is the INTERP support a SUBCUBE (product state) or a genuinely correlated subset?

shape_interp.py found the 99.9% support is always a power of two (4 at n=9, 16 at n=12/15).
That is the signature of a product state: k qubits left in superposition, the rest pinned,
giving 2^k equally-likely bitstrings. If so the support is a subcube -- exactly k bit positions
vary and every combination of them appears -- and the circuit is not 'sampling the feasible
set' at all, it just happens to sit inside it. If instead the varying bits are correlated
(not all 2^k combinations present), it is a real structured superposition.
"""
import pickle
import numpy as np
from analyse_spread_interp import ma_qaoa_probs, structure
from qaoa_sampling_oneshot import state_inttobin

D = "data_qaoa/PO_optimality_maqaoa_percentile_interp"
for n_bits, p, vs in [(9,3,42), (12,2,42), (12,3,42), (15,3,42)]:
    r = pickle.load(open(f"{D}/test_PO_bits_{n_bits}_vseed_{vs}_layers_{p}_etareq_0.5.txt","rb"))
    feas, E_obj, _ = structure(n_bits, vs)
    Qn = np.asarray(r["Q_norm"], float)
    qd = {i: v for i, v in np.ndenumerate(Qn) if not np.isclose(v, 0)}
    g, b = r["best_pars"]
    probs = ma_qaoa_probs(qd, n_bits, p, g, b)
    idx = np.argsort(-probs)
    k = int(np.searchsorted(np.cumsum(probs[idx]), 0.999) + 1)
    sup = idx[:k]
    X = np.array([state_inttobin(int(i), n_bits) for i in sup])
    varying = [j for j in range(n_bits) if len(set(X[:, j])) > 1]
    n_feas_sup = int(feas[sup].sum())
    print(f"n={n_bits} p={p}: support={k}  varying bits={len(varying)} at {varying}  "
          f"2^k={2**len(varying)}  -> {'SUBCUBE (product state)' if 2**len(varying)==k else 'CORRELATED subset'}"
          f"   feasible in support: {n_feas_sup}/{k}")
