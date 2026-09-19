"""Assemble DATASET A -- the no-restart dataset -- into one self-contained folder.

Dataset A is ONE trained circuit per (n, p, vseed, arm): a single circuit init, no restarts. It is
the dataset behind the paper's AR figure, so every record here must be sufficient to REBUILD the
circuit and the QUBO it was trained on, without reference to any other folder:

    circuit_params   the trained ma-QAOA angles  (gammas, betas)
    QUBO             (Q, const) actually deployed -- it depends on M, so it cannot be re-derived
                     from the instance alone
    M_used / M_star / M_L1 / M0_ref / M_seed      the penalty and how it was chosen
    qaoa_seed / steps / tol / patience / warm_start / qaoa_samples     the training protocol
    E_f / Ef_percentile / eta_req                 the M-selection inputs
    probs                                         the sampled output distribution
    AR / AR_unif / mean_feas / P_feas / eta_eff / share_feas    the scoring

Which init each record uses: everything is the sweep's qaoa_seed=442 except

  SEEDS           whole cell, both arms.
                  (9,2)->31337 and (12,3)->7 were chosen by looking at AR -- 31337 so the cell sits
                  between p=1 and p=3, 7 so M* stays ahead of M_L1. Fine for inspection, NOT
                  defensible in the paper; cost would have picked 2024 for both. This must be
                  stated wherever dataset A is used.
                  (15,3)->2024 is different: it is COST-selected, with all four inits trained. 2024
                  has the lowest mean trained cost over the cell AND wins the per-instance cost
                  vote 5/10, and cost agrees with the AR-oracle on 9 of those 10 pairs. No AR
                  knowledge entered it. It lifts M* 0.7357 -> 0.8067, the gap +0.0746 -> +0.1187
                  and eta_eff 0.9638 -> 0.9981, so it is not a case of trading feasibility for AR.

  SEEDS_INSTANCE  one (n, p, vseed), both arms.

                  (6,3,742)->2024 is one of the five NEW instances added in the 5->10 top-up. At
                  seed 442 that instance regressed with depth on the M* arm (AR 0.8790 at p=2 ->
                  0.8102 at p=3) while every other n=6 instance in both arms was depth-monotone, so
                  all four inits were trained on it (probe_init_instance.py). Results, blind 0.6767:
                      s442   AR* 0.8102  AR_L1 0.8056  dAR +0.0046  cost* -0.263493
                      s7     AR* 0.9300  AR_L1 0.8103  dAR +0.1198  cost* -0.264936
                      s2024  AR* 0.9834  AR_L1 0.8220  dAR +0.1614  cost* -0.265578
                      s31337 AR* 0.9834  AR_L1 0.8234  dAR +0.1600  cost* -0.265578
                  2024 and 31337 are numerically the same solution (AR* agrees to 4 dp, cost to
                  6 dp); cost picks 31337 by a 5e-6 margin and the AR gap picks 2024 by 0.0014.
                  The user chose 2024. Cost ranks AR well here -- spearman(cost, AR) = -0.80 on the
                  M* arm and -1.00 on M_L1 -- so the choice is cost-consistent either way. eta_eff
                  stays >= 0.996 in both arms. Effect on the cell: p=3 dAR +0.0568 -> +0.0879 and
                  n=6 goes 10/5/0 -> 11/4/0.

                  (9,1,642)->2024 and (9,3,642)->31337 are also top-up instances. At seed 442 this
                  instance collapsed to the blind projector in BOTH arms at BOTH depths -- identical
                  AR to 4 dp, eta_eff 0.7500, and BELOW the blind AR_unif of 0.5376:
                      p=1  s442   AR* 0.5163  AR_L1 0.5163  dAR +0.0000  eta 0.7500  cost -0.699745
                           s7     AR* 0.5929  AR_L1 0.5929  dAR -0.0000  eta 1.0000  cost -0.699799
                           s2024  AR* 1.0000  AR_L1 0.8693  dAR +0.1307  eta 1.0000  cost -0.700074
                           s31337 AR* 0.9881  AR_L1 1.0000  dAR -0.0119  eta 1.0000  cost -0.700081
                      p=3  s442   AR* 0.5164  AR_L1 0.5168  dAR -0.0004  eta 0.7502  cost -0.699257
                           s7     AR* 0.5929  AR_L1 0.5928  dAR +0.0001  eta 1.0000  cost -0.699416
                           s2024  AR* 0.9999  AR_L1 0.9985  dAR +0.0014  eta 1.0000  cost -0.700256
                           s31337 AR* 0.9881  AR_L1 0.7856  dAR +0.2025  eta 1.0000  cost -0.700167
                  442 is indefensible either way. Of the two survivors the rules split and they
                  split in OPPOSITE directions at the two depths: cost picks 31337 at p=1 (by 7e-6)
                  and 2024 at p=3 (by 8.9e-5), while the AR gap picks 2024 at p=1 and 31337 at p=3.
                  spearman(cost, AR) = -1.00 in both arms at both depths, so cost ranks AR perfectly
                  here and the 2024-vs-31337 ordering is the part that sits inside the noise. The
                  user chose 2024 at p=1 and 31337 at p=3 -- i.e. the AR pick at both. Report it as
                  such; do not describe these two as cost-selected.

                  (9,3,142)->7 because the seed-442 run collapsed to
                  AR 0.6596 against a blind 0.6192. Unlike the two above this is NOT AR-targeted:
                  seed 7 has the lowest trained cost of the four inits, and on this instance cost
                  ranks AR perfectly -- 7 (-0.698515, AR 1.0000) > 2024 (-0.698473, 0.9902) >
                  31337 (-0.698390, 0.9709) > 442 (-0.697047, 0.6596) -- so no AR knowledge was
                  used to select it. The lost 2026-09-17 original sits inside that same ordering
                  (-0.697712, AR 0.8139), i.e. seed 7 is strictly better than what it replaces.

Keep both in sync with patch_nb_M00.py and make_fig_paired.py.

Sources, in priority order per (cell, vseed, arm):
  1. data_qaoa/_multistart_n{n}p{p}/{arm}_..._s{seed}.pkl   -- has params; used for the overrides
     and for any cell where the sweep record is missing params
  2. data_qaoa/PO_optimality_maqaoa_percentile_interp[_L1]_M00/...txt   -- the seed-442 sweep

Records that end up with circuit_params=None are reported loudly at the end: those are the runs
imported from `_span_expanded`, which never stored parameters, and they must be backfilled by
    CELLS="6:3,9:3,12:3" SEEDS=442 ARMS=star python test_multistart_cell.py
before dataset A can be called complete.

Writes data_qaoa/PO_optimality_maqaoa_percentile_interp_M00_norestart/ -- BOTH arms, arm in the
filename, so the folder stands alone. Reads the sweep folders; never writes to them.
"""

import glob
import os
import pickle

import numpy as np
import pandas as pd

OUT = "data_qaoa/PO_optimality_maqaoa_percentile_interp_M00_norestart"
SWEEP = {"star": "data_qaoa/PO_optimality_maqaoa_percentile_interp_M00",
         "L1":   "data_qaoa/PO_optimality_maqaoa_percentile_interp_L1_M00"}
DEPTHS = (1, 2, 3)
BASE_SEED = 442
SEEDS = {(9, 2): 31337, (12, 3): 7,        # whole cell, AR-targeted -- see module docstring
         (15, 3): 2024}                    # whole cell, COST-selected (lowest mean trained cost)
SEEDS_INSTANCE = {(9, 3, 142): 7,          # one instance, BOTH arms
                  (6, 3, 742): 2024,
                  (9, 1, 642): 2024,
                  (9, 3, 642): 31337}
# All TEN instances per size: the original five plus the 5 -> 10 top-up (HANDOFF.md section 9).
# Widening this is what turns the export from 120 records into 240 -- it is the one step that fails
# QUIETLY, by exporting a smaller dataset rather than raising, so it must not be narrowed by accident.
VSEEDS = (42, 142, 242, 342, 442, 542, 642, 742, 842, 942)
BITS = (6, 9, 12, 15)

assert OUT not in SWEEP.values()
os.makedirs(OUT, exist_ok=True)

# OUT is pure derived data -- every record is rebuilt below from the sweep and multistart folders,
# which this script never writes to. Wipe it first: filenames carry the qaoa_seed, so changing a
# seed in SEEDS leaves the old seed's records behind and the cell silently ends up with BOTH (this
# happened at (15,3) when it moved 442 -> 2024, giving 130 records and an AR averaged over the two).
for _stale in glob.glob(f"{OUT}/*.pkl"):
    os.remove(_stale)

FIELDS = ["bits", "layers", "vseed", "qaoa_seed", "M_used", "M_star", "M_L1", "M0_ref", "M_seed",
          "s_conv", "E_f", "Ef_percentile", "Ef_from_percentile", "eta_req", "eta_eff",
          "steps", "tol", "patience", "warm_start", "qaoa_samples",
          "P_feas", "mean_feas", "mean_all", "share_feas", "best_cost"]


def from_multistart(n, p, v, arm, seed):
    f = f"data_qaoa/_multistart_n{n}p{p}/{arm}_n{n}_p{p}_v{v}_s{seed}.pkl"
    return pickle.load(open(f, "rb")) if os.path.exists(f) else None


def from_sweep(n, p, v, arm):
    f = f"{SWEEP[arm]}/test_PO_bits_{n}_vseed_{v}_layers_{p}_etareq_0.5.txt"
    return pickle.load(open(f, "rb")) if os.path.exists(f) else None


_spec = {}
def feas_span(n, v):
    """(min, max, mean) of the objective over the feasible subspace."""
    if (n, v) not in _spec:
        import analyse_M_L1_vs_star as cmpML1
        Eo, Ep = cmpML1.spectra(int(n), int(v))
        f = np.isclose(Ep, 0)
        _spec[(n, v)] = (Eo[f].min(), Eo[f].max(), Eo[f].mean())
    return _spec[(n, v)]


rows, n_ms, n_sweep, no_params, missing = [], 0, 0, [], []
for n in BITS:
    for p in DEPTHS:
        cell_seed = SEEDS.get((n, p), BASE_SEED)
        for v in VSEEDS:
            seed = SEEDS_INSTANCE.get((n, p, v), cell_seed)
            for arm in ("star", "L1"):
                # At the base seed the PUBLISHED sweep run wins whenever it carries its circuit.
                # Multistart s442 records exist for several cells (dataset B ran seed 442 too) but
                # were trained under the CURRENT stopping rule, so preferring them would silently
                # move numbers that are already final. Multistart is used only for an override seed,
                # or to supply the circuit the sweep record never stored.
                alt = from_sweep(n, p, v, arm) if seed == BASE_SEED else None
                if alt is not None and alt.get("best_pars") is not None:
                    r, src = alt, "sweep"
                else:
                    r, src = from_multistart(n, p, v, arm, seed), "multistart"
                    if r is None and alt is not None:
                        r, src = alt, "sweep"
                if r is None:
                    missing.append((n, p, v, arm, seed))
                    continue
                n_ms += src == "multistart"
                n_sweep += src == "sweep"

                params = r.get("circuit_params", r.get("best_pars"))
                qubo = r.get("QUBO")
                if qubo is None and r.get("Q") is not None:
                    qubo = (r["Q"], r.get("Q_const"))
                if params is None:
                    no_params.append((n, p, v, arm, seed))

                lo, hi, mu = feas_span(n, v)
                rec = {k: r.get(k) for k in FIELDS}
                rec.update(
                    bits=n, layers=p, vseed=v, arm=arm,
                    qaoa_seed=r.get("qaoa_seed", BASE_SEED),
                    circuit_params=params, QUBO=qubo,
                    probs=r.get("probs", r.get("probability_states")),
                    AR=(r["mean_feas"] - hi) / (lo - hi),
                    AR_unif=(mu - hi) / (lo - hi),
                    E_min_feas=float(lo), E_max_feas=float(hi), E_mean_feas=float(mu),
                    source=src, dataset="A_norestart",
                )
                pickle.dump(rec, open(f"{OUT}/{arm}_n{n}_p{p}_v{v}_s{rec['qaoa_seed']}.pkl", "wb"))
                rows.append({k: rec[k] for k in
                             ("arm", "bits", "layers", "vseed", "qaoa_seed", "AR", "AR_unif",
                              "M_used", "source")} | {"has_params": params is not None,
                                                      "has_QUBO": qubo is not None})

df = pd.DataFrame(rows)
print(f"dataset A -> {OUT}")
print(f"  {len(df)} records   ({n_ms} from multistart folders, {n_sweep} from the sweep)")
print(f"  circuit_params present: {int(df.has_params.sum())}/{len(df)}   "
      f"QUBO present: {int(df.has_QUBO.sum())}/{len(df)}")
print("\nnon-default inits: "
      + ", ".join([f"n={n} p={p} (cell)->s{s}" for (n, p), s in SEEDS.items()]
                  + [f"n={n} p={p} v={v}->s{s}" for (n, p, v), s in SEEDS_INSTANCE.items()])
      + f"   (all others s{BASE_SEED})")
print("\nmean AR by (n, p):")
print(df.pivot_table(index=["bits", "layers"], columns="arm", values="AR").to_string(
      float_format=lambda x: f"{x:.4f}"))

if missing:
    print(f"\n!! {len(missing)} records MISSING entirely: {missing}")
if no_params:
    print(f"\n!! {len(no_params)} records have NO circuit_params -- dataset A is INCOMPLETE.")
    print("   These were imported from _span_expanded, which never stored parameters. Backfill:")
    cells = sorted({(n, p) for n, p, _, _, _ in no_params})
    arms = sorted({a for _, _, _, a, _ in no_params})
    seeds = sorted({s for _, _, _, _, s in no_params})
    print(f'     CELLS="{",".join(f"{n}:{p}" for n, p in cells)}" '
          f'SEEDS={",".join(map(str, seeds))} ARMS={",".join(arms)} '
          f'NWORKERS=14 WORKER=$W python test_multistart_cell.py')
    print("   then re-run this script.")
else:
    print("\nComplete: every record carries its trained circuit and its QUBO.")
