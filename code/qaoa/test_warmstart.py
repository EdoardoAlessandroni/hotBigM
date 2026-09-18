"""Does INTERP warm-starting cure the p=3 bimodality?

p=3 cold-started from a random init is a lottery: the circuit concentrates on ~1.6 effective
states and eta_eff is essentially the indicator of whether that one state sits below E_f, so
the sweep produced a 0/1 distribution. Warm-starting p=3 from the p=2 optimum should make it
inherit the basin p=2 found, which the sweep shows is reliably a good one.

Compares against the cold p=3 numbers already on disk, same instances, same optimizer seed.
"""

import pickle

import numpy as np

from qaoa_sampling_oneshot import maqaoa_sampling_oneshot

D = "data_qaoa/PO_optimality_maqaoa_percentile"
VSEEDS = range(42, 1000, 100)
N_idx, n_bits, layers = 2, 12, 3

print(f"n_bits={n_bits}, p={layers}, INTERP warm start vs the cold runs on disk\n")
print(f"{'vseed':>6} {'cold eta':>9} {'warm eta':>9} {'delta':>8}")
print("-" * 36, flush=True)

cold, warm = [], []
for vs in VSEEDS:
    old = pickle.load(open(f"{D}/test_PO_bits_{n_bits}_vseed_{vs}_layers_{layers}_etareq_0.5.txt", "rb"))
    r = maqaoa_sampling_oneshot("PO", N_idx, ("PO_small", vs), "optimality", 0.5, layers,
                                qaoa_samples=1000, qaoa_steps=500,
                                Ef_from_percentile=True, Ef_percentile=75,
                                qaoa_seed=442, warm_start=True, print_time=False)
    c, w = float(old["eta_eff"]), float(r["eta_eff_ex"])
    cold.append(c); warm.append(w)
    print(f"{vs:>6} {c:>9.4f} {w:>9.4f} {w - c:>+8.4f}", flush=True)

cold, warm = np.array(cold), np.array(warm)
print("-" * 36)
print(f"{'mean':>6} {cold.mean():>9.4f} {warm.mean():>9.4f} {warm.mean()-cold.mean():>+8.4f}")
print(f"{'sd':>6} {cold.std():>9.4f} {warm.std():>9.4f}")
print(f"\ncollapses (eta < 0.1): cold {(cold < 0.1).sum()}/10  ->  warm {(warm < 0.1).sum()}/10")
print(f"in [0.5, 1.0]:         cold {((cold >= 0.5)).sum()}/10  ->  warm {((warm >= 0.5)).sum()}/10")
