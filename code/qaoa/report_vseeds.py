"""Per-size AR / eta_eff report for a subset of dataset-A instances, on demand.

Built for the 5 -> 10 instance top-up (HANDOFF.md section 9): after each size batch finishes we
have to decide whether the optimization was good enough before spending cores on the next size.
"Good" is, in the user's words and in this order:
  1. M* must come out ahead of M_L1 on AR  -- the primary goal;
  2. eta_eff should be reasonably high (~>0.5-0.6) in BOTH arms -- secondary, but a high AR on a
     circuit that puts its mass outside the feasible set means nothing (HANDOFF.md section 4: AR
     carries no feasibility information at all).
If a cell fails that, the remedy already used on the original five is a different circuit init
(qaoa_seed), which is why qaoa_seed is printed per row.

    python report_vseeds.py --bits 6,15                # the 5 NEW instances (default)
    python report_vseeds.py --bits 6 --vseeds old      # the original 5, for a like-for-like check
    python report_vseeds.py --bits 6 --vseeds all --per-instance

Record resolution is a deliberate COPY of export_dataset_A.py's rule -- sweep record wins at the
base seed whenever it carries its circuit, multistart only for an override seed or a missing
circuit. If that rule changes there, change it here too, or this report stops describing the
dataset it claims to describe.

AR is recomputed here from mean_feas and the feasible span rather than read off the record, because
the sweep payload stores gap_feas (a different, superseded score) and not AR.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

SWEEP = {"star": "data_qaoa/PO_optimality_maqaoa_percentile_interp_M00",
         "L1":   "data_qaoa/PO_optimality_maqaoa_percentile_interp_L1_M00"}
DEPTHS = (1, 2, 3)
BASE_SEED = 442
SEEDS = {(9, 2): 31337, (12, 3): 7, (15, 3): 2024}   # whole cell -- see export_dataset_A.py
SEEDS_INSTANCE = {(9, 3, 142): 7, (6, 3, 742): 2024,  # one instance, both arms
                  (9, 1, 642): 2024, (9, 3, 642): 31337}
V_OLD = (42, 142, 242, 342, 442)
V_NEW = (542, 642, 742, 842, 942)
TIE = 0.01                                           # |dAR| <= TIE counts as a tie

ap = argparse.ArgumentParser()
ap.add_argument("--bits", default="6,9,12,15")
ap.add_argument("--vseeds", default="new", choices=("new", "old", "all"))
ap.add_argument("--per-instance", action="store_true")
args = ap.parse_args()
BITS = tuple(int(x) for x in args.bits.split(","))
VSEEDS = {"new": V_NEW, "old": V_OLD, "all": V_OLD + V_NEW}[args.vseeds]

_spec = {}
def feas_span(n, v):
    if (n, v) not in _spec:
        import analyse_M_L1_vs_star as cmpML1
        Eo, Ep = cmpML1.spectra(int(n), int(v))
        f = np.isclose(Ep, 0)
        _spec[(n, v)] = (Eo[f].min(), Eo[f].max(), Eo[f].mean())
    return _spec[(n, v)]


def load(n, p, v, arm, seed):
    """(record, source) under export_dataset_A.py's resolution rule."""
    sw = f"{SWEEP[arm]}/test_PO_bits_{n}_vseed_{v}_layers_{p}_etareq_0.5.txt"
    alt = pickle.load(open(sw, "rb")) if (seed == BASE_SEED and os.path.exists(sw)) else None
    if alt is not None and alt.get("best_pars") is not None:
        return alt, "sweep"
    ms = f"data_qaoa/_multistart_n{n}p{p}/{arm}_n{n}_p{p}_v{v}_s{seed}.pkl"
    if os.path.exists(ms):
        return pickle.load(open(ms, "rb")), "multistart"
    return (alt, "sweep") if alt is not None else (None, None)


rows, missing = [], []
for n in BITS:
    for p in DEPTHS:
        for v in VSEEDS:
            seed = SEEDS_INSTANCE.get((n, p, v), SEEDS.get((n, p), BASE_SEED))
            for arm in ("star", "L1"):
                r, src = load(n, p, v, arm, seed)
                if r is None:
                    missing.append((arm, n, p, v, seed))
                    continue
                lo, hi, mu = feas_span(n, v)
                rows.append(dict(
                    arm=arm, bits=n, layers=p, vseed=v,
                    qaoa_seed=r.get("qaoa_seed", BASE_SEED), source=src,
                    AR=(r["mean_feas"] - hi) / (lo - hi),
                    AR_unif=(mu - hi) / (lo - hi),
                    eta_eff=r["eta_eff"], P_feas=r["P_feas"], M_used=r["M_used"],
                    share_feas=r["share_feas"], best_cost=r["best_cost"],
                    patience=r.get("patience"),
                    has_params=r.get("circuit_params", r.get("best_pars")) is not None,
                ))

df = pd.DataFrame(rows)
if df.empty:
    print(f"no records yet for bits={BITS} vseeds={VSEEDS}")
    raise SystemExit(0)

print(f"=== bits={BITS}  vseeds={VSEEDS} ({args.vseeds})  {len(df)} records ===")
if missing:
    print(f"!! {len(missing)} records not on disk yet / missing: {missing}")
nop = df[~df.has_params]
if len(nop):
    print(f"!! {len(nop)} records carry NO circuit params -- cannot rebuild the circuit:")
    print(nop[["arm", "bits", "layers", "vseed", "qaoa_seed"]].to_string(index=False))
old_rule = int(df.patience.isna().sum())
print(f"stopping rule: {len(df) - old_rule} current (patience=30), {old_rule} OLD (patience=None)")

wide = df.pivot_table(index=["bits", "layers"], columns="arm",
                      values=["AR", "eta_eff", "P_feas"])
print("\nmean AR / eta_eff / P_feas by (n, p):")
tab = pd.DataFrame({
    "AR_star":  wide[("AR", "star")],    "AR_L1":  wide[("AR", "L1")],
    "AR_unif":  df.groupby(["bits", "layers"]).AR_unif.mean(),
    "dAR":      wide[("AR", "star")] - wide[("AR", "L1")],
    "eta_star": wide[("eta_eff", "star")], "eta_L1": wide[("eta_eff", "L1")],
    "Pf_star":  wide[("P_feas", "star")],  "Pf_L1":  wide[("P_feas", "L1")],
})
print(tab.to_string(float_format=lambda x: f"{x:+.4f}"))

# instance-paired verdict: the two arms are trained on the SAME instance and init, so the paired
# sign count is the statistic that matters, not the difference of the two cell means.
pair = df.pivot_table(index=["bits", "layers", "vseed"], columns="arm", values="AR")
pair["dAR"] = pair["star"] - pair["L1"]
pair = pair.dropna(subset=["dAR"])
print("\npaired verdict (per instance, tie = |dAR| <= 0.01):")
for key, g in pair.groupby(level="bits"):
    w = int((g.dAR > TIE).sum()); l = int((g.dAR < -TIE).sum()); t = len(g) - w - l
    print(f"  n={key:2d}:  {w} win / {t} tie / {l} loss   mean dAR {g.dAR.mean():+.4f}")
w = int((pair.dAR > TIE).sum()); l = int((pair.dAR < -TIE).sum()); t = len(pair) - w - l
print(f"  TOTAL:  {w} win / {t} tie / {l} loss   mean dAR {pair.dAR.mean():+.4f}")

# the flags worth acting on: an arm below its own instance's blind reference, or low eta_eff
bad_ar = df[df.AR < df.AR_unif - 1e-9]
if len(bad_ar):
    print(f"\n!! {len(bad_ar)} runs BELOW their own instance's blind AR_unif "
          f"(candidates for a different qaoa_seed):")
    print(bad_ar[["arm", "bits", "layers", "vseed", "qaoa_seed", "AR", "AR_unif",
                  "eta_eff", "best_cost"]]
          .sort_values(["bits", "layers", "vseed"])
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
low_eta = df[df.eta_eff < 0.5]
if len(low_eta):
    print(f"\n!! {len(low_eta)} runs with eta_eff < 0.50:")
    print(low_eta[["arm", "bits", "layers", "vseed", "qaoa_seed", "eta_eff", "P_feas", "AR"]]
          .sort_values(["bits", "layers", "vseed"])
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if args.per_instance:
    print("\nper-instance:")
    print(df.sort_values(["bits", "layers", "vseed", "arm"])
          [["arm", "bits", "layers", "vseed", "qaoa_seed", "source", "AR", "AR_unif",
            "eta_eff", "P_feas", "share_feas", "M_used", "best_cost"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
