"""Compare all four circuit inits on ONE (n, p, vseed), both arms, and show what each selection
rule would pick.

    python probe_init_instance.py --bits 6 --layers 3 --vseed 742

Reads data_qaoa/_multistart_n{n}p{p}/{arm}_n{n}_p{p}_v{v}_s{seed}.pkl (written by
test_multistart_cell.py) and prints, per init:  AR and eta_eff for both arms, the M* - M_L1 gap, and
the trained cost for both arms.

Both selection rules are printed, neutrally, because the user decides between them:

  COST   lowest mean trained cost. Uses no knowledge of AR. This is how (15,3) and (9,3,142) were
         chosen.
  AR     largest M* - M_L1 gap. This is how (9,2)->31337 and (12,3)->7 were chosen. The user has
         said these are not indefensible and that he has a solution in mind for the paper, so do
         NOT argue against this rule -- just report what each rule picks and what it costs/buys.

Also printed: spearman(cost, AR) over the four inits, which says whether cost ranks AR on this
particular instance at all -- that is the useful input to the choice, and it varies by instance
(perfect at (9,3,142), only ~-0.35 across dataset B as a whole).

A dataset-A cell/instance override applies ONE seed to BOTH arms, so the cost rule is applied to the
mean trained cost over the two arms; the per-arm costs are printed too, since they can disagree.
Cost is comparable across inits only WITHIN one (instance, arm) -- never across instances.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--bits", type=int, required=True)
ap.add_argument("--layers", type=int, required=True)
ap.add_argument("--vseed", type=int, required=True)
ap.add_argument("--seeds", default="442,7,2024,31337")
args = ap.parse_args()
n, p, v = args.bits, args.layers, args.vseed
SEEDS = [int(x) for x in args.seeds.split(",")]

rows = []
for s in SEEDS:
    for arm in ("star", "L1"):
        f = f"data_qaoa/_multistart_n{n}p{p}/{arm}_n{n}_p{p}_v{v}_s{s}.pkl"
        if not os.path.exists(f):
            print(f"missing: {f}")
            continue
        r = pickle.load(open(f, "rb"))
        rows.append(dict(seed=s, arm=arm, AR=r["AR"], AR_unif=r["AR_unif"],
                         eta_eff=r["eta_eff"], P_feas=r["P_feas"], cost=r["best_cost"],
                         M_used=r["M_used"], secs=r["secs"]))
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("nothing to compare yet")

# the published sweep record is the incumbent this probe is testing against
sw = {"star": "data_qaoa/PO_optimality_maqaoa_percentile_interp_M00",
      "L1":   "data_qaoa/PO_optimality_maqaoa_percentile_interp_L1_M00"}
inc = {}
for arm, d in sw.items():
    f = f"{d}/test_PO_bits_{n}_vseed_{v}_layers_{p}_etareq_0.5.txt"
    if os.path.exists(f):
        r = pickle.load(open(f, "rb"))
        import analyse_M_L1_vs_star as cmpML1
        Eo, Ep = cmpML1.spectra(n, v)
        fe = np.isclose(Ep, 0)
        lo, hi = Eo[fe].min(), Eo[fe].max()
        inc[arm] = dict(AR=(r["mean_feas"] - hi) / (lo - hi), eta_eff=r["eta_eff"],
                        cost=r["best_cost"], patience=r.get("patience"))

wide = df.pivot_table(index="seed", columns="arm", values=["AR", "eta_eff", "cost"])
tab = pd.DataFrame({
    "AR_star": wide[("AR", "star")], "AR_L1": wide[("AR", "L1")],
    "dAR": wide[("AR", "star")] - wide[("AR", "L1")],
    "eta_star": wide[("eta_eff", "star")], "eta_L1": wide[("eta_eff", "L1")],
    "cost_star": wide[("cost", "star")], "cost_L1": wide[("cost", "L1")],
}).reindex([s for s in SEEDS if s in wide.index])
tab["cost_mean"] = (tab.cost_star + tab.cost_L1) / 2

blind = float(df.AR_unif.iloc[0])
print(f"=== n={n} p={p} v={v}   blind AR_unif = {blind:.4f} ===")
print(tab.to_string(float_format=lambda x: f"{x:+.6f}"))

if inc:
    print("\nincumbent published sweep record (qaoa_seed 442):")
    for arm, d in inc.items():
        rule = "current" if d["patience"] is not None else "OLD"
        print(f"  {arm:4s}  AR={d['AR']:.4f}  eta={d['eta_eff']:.4f}  "
              f"cost={d['cost']:+.8f}  [{rule} stopping rule]")

cost_pick = tab.cost_mean.idxmin()
ar_pick = tab.dAR.idxmax()
print()
for label, pick in (("COST (lowest mean trained cost)", cost_pick),
                    ("AR   (largest M* - M_L1 gap)   ", ar_pick)):
    print(f"{label} -> seed {pick}: dAR {tab.dAR[pick]:+.4f}  "
          f"AR* {tab.AR_star[pick]:.4f}  AR_L1 {tab.AR_L1[pick]:.4f}  "
          f"eta* {tab.eta_star[pick]:.4f}  eta_L1 {tab.eta_L1[pick]:.4f}")
print(f"  per-arm cost picks: star -> s{tab.cost_star.idxmin()}, L1 -> s{tab.cost_L1.idxmin()}")
print(f"\nthe two rules {'AGREE' if cost_pick == ar_pick else 'DISAGREE'}; the AR pick buys "
      f"{tab.dAR[ar_pick] - tab.dAR[cost_pick]:+.4f} of dAR over the cost pick")
# does cost rank AR at all on this instance? (spearman over the four inits, per arm)
for arm, c, a in (("star", "cost_star", "AR_star"), ("L1", "cost_L1", "AR_L1")):
    if tab[c].notna().sum() >= 3:
        rho = tab[[c, a]].corr(method="spearman").iloc[0, 1]
        print(f"  {arm:4s}: spearman(cost, AR) over inits = {rho:+.2f} "
              f"(-1 = lower cost perfectly predicts higher AR)")
