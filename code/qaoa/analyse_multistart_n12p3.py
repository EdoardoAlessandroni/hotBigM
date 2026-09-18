"""Read `data_qaoa/_multistart_n12p3/` and answer the three questions the test was built to settle.

1. Does multi-start lift the weakest cell?  Reference, from the finished M0=0 grid at n=12 p=3
   (corrected AR, 1 = feasible optimum): M* 0.6436, M_L1 0.5974, blind 0.5788. Both arms sit
   barely above a blind feasibility projector, so this is where the study is weakest.

2. Can the SELECTION be made without an oracle?  At n=9 p=2 picking the init with the lowest
   trained cost chose the same run the AR-oracle would on 5/5 instances. If that holds here, the
   recipe is legitimate; if cost and AR decouple, multi-start is only usable with oracle knowledge
   and is therefore useless in practice.

3. Does it preserve the paper's claim?  If multi-start lifts M_L1 as much as M*, the headline
   "M* buys objective quality" survives only at K=1 and the paper has a problem. This is the
   question that actually matters, and it is why both arms are in the test.

Also reports the K-curve -- expected AR of pick-by-cost over every subset of K inits -- which is how
many restarts to recommend. At n=9 that read K=1 0.70, K=2 0.82, K=3 0.91, but with only 3 inits
K=3 was the oracle by construction; 4 inits here give the first non-tautological look at whether it
has plateaued.
"""

import glob
import pickle
from itertools import combinations

import numpy as np
import pandas as pd

SRC = "data_qaoa/_multistart_n12p3"
GRID = {"M*": 0.6436, "M_L1": 0.5974}       # same cell, single init 442, from the finished sweep
BLIND = 0.5788
ARM = {"star": "M*", "L1": "M_L1"}

rows = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{SRC}/*.pkl"))]
if not rows:
    raise SystemExit(f"no runs in {SRC} yet")
df = pd.DataFrame([{k: r[k] for k in ("arm", "vseed", "qaoa_seed", "AR", "AR_unif",
                                      "best_cost", "P_feas", "eta_eff", "secs")} for r in rows])
df["arm"] = df.arm.map(ARM)
INITS = sorted(df.qaoa_seed.unique())
print(f"{len(df)}/40 runs   inits {INITS}   vseeds {sorted(df.vseed.unique())}\n")

print("AR per (arm, vseed, init)   [1 = feasible optimum, higher better]")
print(df.pivot_table(index=["arm", "vseed"], columns="qaoa_seed", values="AR").to_string(
      float_format=lambda x: f"{x:.4f}"))

# ---- Q2: is the cost a legitimate selector? ------------------------------------------------
print("\n" + "=" * 78)
print("Q2  pick-by-lowest-cost vs the AR-oracle, per instance")
hits = tot = 0
for arm in df.arm.unique():
    for v in sorted(df.vseed.unique()):
        s = df[(df.arm == arm) & (df.vseed == v)]
        if len(s) < 2:
            continue
        by_cost = s.loc[s.best_cost.idxmin()]
        by_AR = s.loc[s.AR.idxmax()]
        ok = by_cost.qaoa_seed == by_AR.qaoa_seed
        hits += ok
        tot += 1
        print(f"  {arm:5s} v={v:3d}:  cost picks s={int(by_cost.qaoa_seed):5d} AR={by_cost.AR:.4f}"
              f"   | oracle s={int(by_AR.qaoa_seed):5d} AR={by_AR.AR:.4f}   "
              f"{'MATCH' if ok else 'MISS'}  (n={len(s)})")
if tot:
    rho = df.groupby("arm").apply(
        lambda g: g.best_cost.corr(g.AR, method="spearman"), include_groups=False)
    print(f"\n  selector matches oracle {hits}/{tot}")
    print("  spearman(cost, AR) per arm (negative = lower cost goes with higher AR):")
    print("   " + rho.to_string(float_format=lambda x: f"{x:+.3f}").replace("\n", "\n   "))

# ---- Q1 + Q3: does it lift the cell, and does the M* - M_L1 gap survive? --------------------
print("\n" + "=" * 78)
print("Q1/Q3  mean AR by arm: single init vs multi-start (selected by cost)")
out = {}
for arm in ("M*", "M_L1"):
    s = df[df.arm == arm]
    if not len(s):
        continue
    per_init = {q: s[s.qaoa_seed == q].AR.mean() for q in INITS}
    sel = s.loc[s.groupby("vseed").best_cost.idxmin()].AR.mean()
    orc = s.groupby("vseed").AR.max().mean()
    out[arm] = dict(grid=GRID[arm], **{f"init{q}": v for q, v in per_init.items()},
                    pick_by_cost=sel, oracle=orc)
res = pd.DataFrame(out).T
print(res.to_string(float_format=lambda x: f"{x:.4f}"))
print(f"\n  blind reference (uniform over feasible) = {BLIND:.4f}")
if {"M*", "M_L1"} <= set(res.index):
    print(f"\n  M* - M_L1 gap:   single init (grid) {GRID['M*'] - GRID['M_L1']:+.4f}"
          f"   ->   multi-start {res.loc['M*','pick_by_cost'] - res.loc['M_L1','pick_by_cost']:+.4f}")
    print("  (if the gap collapses, multi-start rescues quality but dissolves the paper's claim)")

# ---- how many restarts? ---------------------------------------------------------------------
print("\n" + "=" * 78)
print("K-curve: expected AR of pick-by-cost over all subsets of K inits")
for arm in ("M*", "M_L1"):
    s = df[df.arm == arm]
    if not len(s):
        continue
    line = []
    for K in range(1, len(INITS) + 1):
        vals = []
        for sub in combinations(INITS, K):
            t = s[s.qaoa_seed.isin(sub)]
            per = [t[t.vseed == v].loc[t[t.vseed == v].best_cost.idxmin()].AR
                   for v in sorted(t.vseed.unique()) if len(t[t.vseed == v])]
            vals.append(np.mean(per))
        line.append(f"K={K}: {np.mean(vals):.4f}")
    print(f"  {arm:5s}  " + "   ".join(line))
print("\n  NOTE: K = len(inits) equals the oracle by construction -- read the SHAPE, not the endpoint.")
