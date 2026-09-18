"""Does the circuit prefer low-objective feasible states, and does it depend on depth?

Decomposes eta_eff = P(feasible) * P(E<=E_f | feasible) and, crucially, reports where the
sampled feasible mass sits in the exhaustive feasible spectrum (mean quantile: 0.5 =
indifferent, <0.5 = prefers good solutions) plus the effective number of feasible states
the circuit spreads over (exp of Shannon entropy).

A sampler that collapses onto ONE arbitrary feasible state is blind yet still scores
eta_eff ~ 1 whenever that state falls below E_f, so per-run eta_eff is not diagnostic --
mean quantile and effective #states are.
"""
import itertools
import numpy as np
from qaoa_sampling_oneshot import (maqaoa_sampling_oneshot, get_QUBO_PO, get_PO_obj,
                                   evaluate_energy, state_inttobin, N_stocks, w)

N_idx = 2                      # n_bits = 12, fast enough to sweep depths x vseeds
N = N_stocks[N_idx]; n_bits = N * w
VSEEDS = (42, 142, 242)
LAYERS = (1, 2, 3)

b = 2**w - 1
X = np.array([[int(x) for s in c for x in format(s, f"0{w}b")]
              for c in itertools.product(range(b+1), repeat=N) if sum(c) == b], float)

print(f"n_bits={n_bits}, |feasible|={len(X)}\n")
print(f"{'vseed':>6} {'p':>2} {'P(feas)':>9} {'P(<=Ef|f)':>10} {'eta_eff':>9} {'F(E_f)':>8} "
      f"{'mean quant':>11} {'eff #states':>12}")
print("-" * 76)

for vseed in VSEEDS:
    Q_pen, c_pen = get_QUBO_PO(N, w, 1, "seed", ("PO_small", vseed), penalization=True,  objective=False)
    Q_obj, c_obj = get_QUBO_PO(N, w, 1, "seed", ("PO_small", vseed), penalization=False, objective=True)
    Qo, Lo = get_PO_obj(n_bits, ("PO_small", vseed))
    E_feas = np.sort(np.einsum("ij,jk,ik->i", X, Qo + np.diag(Lo), X))
    feas_mask = np.zeros(2**n_bits, dtype=bool); E_obj_all = np.zeros(2**n_bits)
    for i in range(2**n_bits):
        x = state_inttobin(i, n_bits)
        feas_mask[i] = np.isclose(evaluate_energy(x, Q_pen, c_pen), 0)
        E_obj_all[i] = evaluate_energy(x, Q_obj, c_obj)
    for p in LAYERS:
        r = maqaoa_sampling_oneshot("PO", N_idx, ("PO_small", vseed), "optimality", 0.5, p,
                                    qaoa_samples=1000, qaoa_steps=500,
                                    Ef_from_percentile=True, Ef_percentile=75,
                                    qaoa_seed=442, print_time=False)
        probs = np.asarray(r["probability_states"], float); E_f = r["E_f"]
        p_feas = probs[feas_mask].sum()
        cond = probs[feas_mask & (E_obj_all <= E_f)].sum() / p_feas
        pf = probs[feas_mask] / p_feas
        q = np.searchsorted(E_feas, E_obj_all[feas_mask], side="right") / len(E_feas)
        mean_q = float((pf * q).sum())
        eff = float(np.exp(-(pf[pf > 0] * np.log(pf[pf > 0])).sum()))
        print(f"{vseed:>6} {p:>2} {p_feas:>9.4f} {cond:>10.4f} {r['eta_eff_ex']:>9.4f} "
              f"{float(np.mean(E_feas <= E_f)):>8.3f} {mean_q:>11.3f} {eff:>12.1f}", flush=True)
