"""Does the training cost tell a good p=3 run from a collapsed one?

Every proposed remedy so far (multi-restart, best-of-k) rests on the assumption that the
optimizer's own objective ranks the restarts the way eta_eff does. The earlier diagnostic
printed cost = -3.4124 for BOTH a collapsed run (eta = 0.006) and a healthy one (eta = 0.993)
-- but at 4 decimals, and the objective contributes ~1e-7 of the cost at M* ~ 1e5, so the
tie may be pure print rounding.

This prints the cost at full precision across restarts. Two possible outcomes:
  * costs differ (even at 1e-7) and rank like eta_eff -> best-of-k on cost is a valid,
    non-outcome-selecting fix, and we can just implement it.
  * costs are genuinely tied -> the QUBO cost is flat across feasible states at M*, no
    cost-based selector can work, and the restart idea is dead.
"""

import numpy as np

from qaoa_sampling_oneshot import maqaoa_sampling_oneshot

CELLS = [(2, 12, 142, 3), (2, 12, 242, 3), (3, 15, 42, 3)]   # (N_idx, n_bits, vseed, layers)
SEEDS = (442, 443, 444, 445)

print(f"{'n':>3} {'vseed':>6} {'p':>2} {'seed':>5} {'cost (full precision)':>26} {'eta_eff':>9}")
print("-" * 58, flush=True)

for N_idx, n_bits, vseed, layers in CELLS:
    costs, etas = [], []
    for s in SEEDS:
        r = maqaoa_sampling_oneshot("PO", N_idx, ("PO_small", vseed), "optimality", 0.5, layers,
                                    qaoa_samples=1000, qaoa_steps=500,
                                    Ef_from_percentile=True, Ef_percentile=75,
                                    qaoa_seed=s, print_time=False)
        costs.append(float(r["best_cost"])); etas.append(float(r["eta_eff_ex"]))
        print(f"{n_bits:>3} {vseed:>6} {layers:>2} {s:>5} {costs[-1]:>26.16f} {etas[-1]:>9.4f}",
              flush=True)

    costs, etas = np.array(costs), np.array(etas)
    spread = costs.max() - costs.min()
    picked = int(np.argmin(costs))
    print(f"    cost spread = {spread:.3e}   best-cost seed = {SEEDS[picked]} -> "
          f"eta = {etas[picked]:.4f}   (best possible {etas.max():.4f}, "
          f"default seed 442 {etas[0]:.4f})", flush=True)
    if spread > 0:
        print(f"    rank corr(cost, -eta) = "
              f"{np.corrcoef(costs, -etas)[0,1]:+.3f}", flush=True)
    print(flush=True)
