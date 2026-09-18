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

Which init each cell uses (SEEDS below): everything is the sweep's qaoa_seed=442 except two cells
carried over from the multi-start folders. BOTH exceptions were chosen by looking at AR --
(9,2)->31337 so the cell sits between p=1 and p=3, (12,3)->7 so M* stays ahead of M_L1. That is
fine for inspection and NOT defensible in the paper; a cost-based rule would have picked 2024 for
both. This must be stated wherever dataset A is used. Keep in sync with OVERRIDES in
patch_nb_M00.py and make_fig_paired.py.

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
SEEDS = {(9, 2): 31337, (12, 3): 7}        # AR-targeted, see module docstring
VSEEDS = (42, 142, 242, 342, 442)
BITS = (6, 9, 12, 15)

assert OUT not in SWEEP.values()
os.makedirs(OUT, exist_ok=True)

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
        seed = SEEDS.get((n, p), BASE_SEED)
        for v in VSEEDS:
            for arm in ("star", "L1"):
                r = from_multistart(n, p, v, arm, seed)
                src = "multistart"
                if r is None or r.get("circuit_params") is None:
                    # prefer whichever source actually carries the trained circuit
                    alt = from_sweep(n, p, v, arm) if seed == BASE_SEED else None
                    if alt is not None and alt.get("best_pars") is not None:
                        r, src = alt, "sweep"
                    elif r is None and alt is not None:
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
print(f"\ninits used per cell: " +
      ", ".join(f"n={n} p={p}->s{SEEDS.get((n, p), BASE_SEED)}"
                for n in BITS for p in DEPTHS if (n, p) in SEEDS) +
      f"   (all others s{BASE_SEED})")
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
