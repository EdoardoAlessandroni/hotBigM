"""eta_eff vs n, one colour per depth, solid = M*, dashed = M_L1.

The AR figure is conditional on feasibility (infeasible mass is discarded and renormalised), so AR
is only the right figure of merit if the penalty is actually delivering the feasibility it promised.
eta_eff is that check: eta_req = 0.5 was requested from every run, so anything at or above the grey
line met its contract, and the M* vs M_L1 comparison on AR is only fair where the two arms are not
buying AR with different amounts of feasibility.

Reads the EXPORTED dataset A (export_dataset_A.py), not the raw sweep. That folder already has the
single-init resolution rule applied -- base seed 442 except the SEEDS / SEEDS_INSTANCE overrides --
so this figure cannot disagree with the AR figure about which run each point represents. It used to
re-implement the rule from the sweep folders and had drifted: it still listed only the original five
vseeds and an OVERRIDES dict missing (15,3)->2024 and every per-instance override, so it silently
plotted eta_eff for runs the AR figure had already replaced. Run export_dataset_A.py first.
"""

import glob
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = "data_qaoa/PO_optimality_maqaoa_percentile_interp_M00_norestart"
BITS, DEPTHS = (6, 9, 12, 15), (1, 2, 3)
ETA_REQ = 0.5
ARM = {"star": "M*", "L1": "M_L1"}

rows = []
for f in sorted(glob.glob(f"{SRC}/*.pkl")):
    r = pickle.load(open(f, "rb"))
    if r["bits"] not in BITS or r["layers"] not in DEPTHS:
        continue
    rows.append(dict(arm=ARM[r["arm"]], bits=int(r["bits"]), layers=int(r["layers"]),
                     vseed=int(r["vseed"]), qaoa_seed=int(r["qaoa_seed"]),
                     eta=float(r["eta_eff"]), P_feas=float(r["P_feas"])))
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit(f"no records in {SRC} -- run export_dataset_A.py first")
n_ov = int((df.qaoa_seed != 442).sum())
print(f"{len(df)} runs over {df.vseed.nunique()} vseeds   "
      f"({n_ov} at a non-default init)")

g = df.groupby(["arm", "bits", "layers"]).eta.agg(["mean", "std", "min"])
print("\neta_eff by (arm, n, p):")
print(g.to_string(float_format=lambda x: f"{x:.4f}"))
print(f"\nruns below eta_req={ETA_REQ}: {(df.eta < ETA_REQ).sum()}/{len(df)}")
if (df.eta < ETA_REQ).any():
    print(df[df.eta < ETA_REQ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

piv = df.pivot_table(index=["bits", "layers"], columns="arm", values="eta")
piv["M* - M_L1"] = piv["M*"] - piv["M_L1"]
print("\nmean eta_eff, M* vs M_L1:")
print(piv.to_string(float_format=lambda x: f"{x:+.4f}"))

# ---- figure ----------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
cols = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}
for p in DEPTHS:
    for arm, ls, mk in (("M*", "-", "o"), ("M_L1", "--", "s")):
        s = df[(df.arm == arm) & (df.layers == p)].groupby("bits").eta.agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], yerr=s["std"].fillna(0), color=cols[p], ls=ls,
                    marker=mk, ms=5, lw=1.6, capsize=3, alpha=.9,
                    label=f"p={p}, {arm}")

ax.axhline(ETA_REQ, color="grey", lw=1, ls=":", zorder=0)
ax.text(15.25, ETA_REQ, r"$\eta_{req}=0.5$", color="grey", fontsize=8, va="center")
ax.set_xlabel("problem size $n$")
ax.set_ylabel(r"$\eta_{eff}$")
ax.set_title(r"Effective feasibility guarantee: solid $M^*$, dashed $M_{L1}$")
ax.set_xticks(BITS)
ax.set_ylim(0.4, 1.06)
ax.grid(alpha=.25, lw=.5)

h_p = [plt.Line2D([], [], color=cols[p], lw=1.8, label=f"$p={p}$") for p in DEPTHS]
h_a = [plt.Line2D([], [], color="k", ls="-", marker="o", ms=5, lw=1.4, label="$M^*$"),
       plt.Line2D([], [], color="k", ls="--", marker="s", ms=5, lw=1.4, label="$M_{L1}$")]
ax.add_artist(ax.legend(handles=h_p, loc="lower left", fontsize=8, frameon=False, ncol=3))
ax.legend(handles=h_a, loc="lower right", fontsize=8, frameon=False)

fig.tight_layout()
os.makedirs("misc_plots", exist_ok=True)
fig.savefig("misc_plots/eta_eff_vs_n.png", dpi=170)
print("\nwrote misc_plots/eta_eff_vs_n.png")
