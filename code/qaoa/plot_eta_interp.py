"""eta_eff of the percentile-E_f sweep: is it in [0.5, 1] on average, with honest spread?

Left panel  -- per-instance scatter + mean, one curve per depth, against the eta_req = 0.5 floor.
Right panel -- the distribution of eta_eff per depth, which is where the p=3 bimodality shows.
"""

import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np

D = "data_qaoa/PO_optimality_maqaoa_percentile_interp"
BITS = (6, 9, 12, 15)
PS = (1, 2, 3)
COL = {1: "#4C72B0", 2: "#55A868", 3: "#C44E52"}

rows = []
for f in sorted(glob.glob(f"{D}/*.txt")):
    r = pickle.load(open(f, "rb"))
    rows.append((r["bits"], r["layers"], r["vseed"], r["eta_eff"]))
rows = np.array(rows)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2),
                              gridspec_kw={"width_ratios": [1.35, 1]})

ax.axhspan(0.5, 1.0, color="green", alpha=0.06, zorder=0)
ax.axhline(0.5, color="k", ls="--", lw=1, zorder=1)
ax.text(15.35, 0.505, r"$\eta_{\rm req}=0.5$", fontsize=8, va="bottom", ha="right")

for p in PS:
    m = [rows[(rows[:, 1] == p) & (rows[:, 0] == nb)][:, 3] for nb in BITS]
    means = [e.mean() for e in m]
    for nb, e in zip(BITS, m):
        ax.scatter(np.full(len(e), nb) + (p - 2) * 0.28, e, s=14, alpha=0.45,
                   color=COL[p], edgecolors="none", zorder=2)
    ax.plot(BITS, means, "-o", color=COL[p], lw=1.8, ms=6, zorder=3,
            label=f"$p={p}$  (mean over 10 instances)")

ax.set_xticks(BITS)
ax.set_xlabel("$n_{\\rm bits}$")
ax.set_ylabel(r"$\eta_{\rm eff}$")
ax.set_ylim(-0.04, 1.06)
ax.set_title(r"ma-QAOA, PO / optimality, $E_f = P_{75}$, INTERP warm start")
ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

# --- right: distribution, pooled over n >= 9 (n = 6 has only 8 feasible states) ---
for p in PS:
    e = rows[(rows[:, 1] == p) & (rows[:, 0] >= 9)][:, 3]
    ax2.hist(e, bins=np.linspace(0, 1, 21), histtype="step", lw=1.8, color=COL[p],
             label=f"$p={p}$: mean {e.mean():.2f}, sd {e.std():.2f}")
ax2.axvline(0.5, color="k", ls="--", lw=1)
ax2.set_xlabel(r"$\eta_{\rm eff}$")
ax2.set_ylabel("# runs")
ax2.set_title(r"distribution, $n_{\rm bits}\geq 9$  (30 runs/depth)")
ax2.legend(fontsize=8, loc="upper left", framealpha=0.9)

fig.tight_layout()
fig.savefig("misc_plots/PO_optimality_percentile_interp_eta.pdf")
fig.savefig("misc_plots/PO_optimality_percentile_interp_eta.png", dpi=170)
print("saved misc_plots/PO_optimality_percentile_interp_eta.{pdf,png}")
