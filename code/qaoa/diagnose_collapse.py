"""Is a 'collapse' a training failure, or a converged run that sits above E_f?

eta_eff = P(feasible) * P(E_obj <= E_f | feasible). A collapsed cell (eta_eff ~ 0.006) is
consistent with either
  (a) P(feasible) ~ 0.006  -- the circuit never reached the feasible subspace, or
  (b) P(feasible) ~ 1 but the conditional ~ 0 -- it converged onto feasible states that
      happen to lie ABOVE the cutoff.
(a) is broken training that a cost-selected restart legitimately repairs; (b) is not broken,
and 'repairing' it would amount to selecting on the reported metric.

Runs the collapsing seed and a healthy seed on the same cell and prints the decomposition.
"""

import itertools

import numpy as np

from qaoa_sampling_oneshot import (maqaoa_sampling_oneshot, get_QUBO_PO, get_PO_obj,
                                   evaluate_energy, state_inttobin, N_stocks, w)

N_idx, vseed, layers = 3, 42, 3          # n_bits = 15, the cell that collapsed on seed 442
N = N_stocks[N_idx]; n_bits = N * w

Q_pen, c_pen = get_QUBO_PO(N, w, 1, "seed", ("PO_small", vseed), penalization=True,  objective=False)
Q_obj, c_obj = get_QUBO_PO(N, w, 1, "seed", ("PO_small", vseed), penalization=False, objective=True)

# exhaustive feasible spectrum, for F(E_f) and for ranking sampled states
b = 2**w - 1
X = np.array([[int(x) for s in c for x in format(s, f"0{w}b")]
              for c in itertools.product(range(b+1), repeat=N) if sum(c) == b], float)
Qo, Lo = get_PO_obj(n_bits, ("PO_small", vseed))
E_feas = np.sort(np.einsum("ij,jk,ik->i", X, Qo + np.diag(Lo), X))

# precompute feasibility + objective for every basis state once
feas_mask = np.zeros(2**n_bits, dtype=bool)
E_obj_all = np.zeros(2**n_bits)
for i in range(2**n_bits):
    x = state_inttobin(i, n_bits)
    feas_mask[i] = np.isclose(evaluate_energy(x, Q_pen, c_pen), 0)
    E_obj_all[i] = evaluate_energy(x, Q_obj, c_obj)

print(f"n_bits={n_bits} p={layers} vseed={vseed}   |feasible| = {feas_mask.sum()}\n")
print(f"{'seed':>5} {'cost':>11} {'P(feas)':>9} {'P(E<=Ef|feas)':>14} {'eta_eff':>9} "
      f"{'F(E_f)':>8} {'mean quantile':>14} {'eff. #states':>13}")
print("-" * 92)

for s in (442, 443):
    r = maqaoa_sampling_oneshot("PO", N_idx, ("PO_small", vseed), "optimality", 0.5, layers,
                                qaoa_samples=1000, qaoa_steps=500,
                                Ef_from_percentile=True, Ef_percentile=75,
                                qaoa_seed=s, print_time=False)
    probs = np.asarray(r["probability_states"], float)
    E_f = r["E_f"]
    p_feas = probs[feas_mask].sum()
    cond = probs[feas_mask & (E_obj_all <= E_f)].sum() / p_feas if p_feas > 0 else np.nan
    F = float(np.mean(E_feas <= E_f))
    # where the sampled feasible mass sits in the feasible spectrum (0 = best, 1 = worst)
    pf = probs[feas_mask] / p_feas
    q = np.searchsorted(E_feas, E_obj_all[feas_mask], side="right") / len(E_feas)
    mean_q = float((pf * q).sum())
    eff_states = float(np.exp(-(pf[pf > 0] * np.log(pf[pf > 0])).sum()))
    print(f"{s:>5} {r['best_cost']:>11.4f} {p_feas:>9.4f} {cond:>14.4f} "
          f"{r['eta_eff_ex']:>9.4f} {F:>8.3f} {mean_q:>14.3f} {eff_states:>13.1f}")

print("\nmean quantile 0.5 = indifferent to the objective; < 0.5 = prefers low-energy solutions.")
