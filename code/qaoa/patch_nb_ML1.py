"""Append the M_L1 vs M* comparison cells to qaoa_playground.ipynb, after the '## M_L1 M* comparison'
markdown cell. Idempotent: reruns replace the previously inserted block instead of duplicating it.

Cell bodies are plain (non-f) strings so that the notebook code can use braces freely; only the
literal token __TAG__ is substituted.
"""

import json

NB = "qaoa_playground.ipynb"
MARKER = "## M_L1 M* comparison"
TAG = "### [M_L1-vs-Mstar]"   # every inserted cell starts with this, so we can find & replace them


CELL_TABLE = '''__TAG__ build the comparison table (exact output distributions, both arms)
## Both arms were trained identically -- same grid, same vseeds, E_f = P75, INTERP warm start,
## qaoa_seed 442, 500 Adam steps -- so the single variable is the deployed penalty weight:
##     M*    = our algorithm's, from M_method_opt(beta, ...)
##     M_L1  = the naive closed form, L1_norm_hot(Q_obj, c_obj, n, T_unnormalized, eta)
## evaluated at the SAME beta = 1/temperature_unnormalized. T_unnormalized (not T_normalized) is
## the only dimensionally admissible choice: the bracket of the L1 formula is an energy in raw
## QUBO units, so the temperature added to it must be in those units too.
##
## Energies below are ALWAYS the raw objective E_obj, never the total E_obj + M*E_pen: the total
## is not comparable across arms, since M differs by construction.

import analyse_M_L1_vs_star as cmpML1

REBUILD = False    # True re-evaluates all 240 stored circuits (~min); False loads the cached table

if REBUILD:
    df_M = pd.DataFrame(cmpML1.collect(cmpML1.D_STAR, "M*") + cmpML1.collect(cmpML1.D_L1, "M_L1"))
    df_M.to_pickle("data_qaoa/M_L1_vs_star_summary.pkl")
else:
    df_M = pd.read_pickle("data_qaoa/M_L1_vs_star_summary.pkl")

print(f"{len(df_M)} runs: {df_M['arm'].value_counts().to_dict()}")
df_M.groupby(["bits", "layers", "arm"]).agg(
    M_used   = ("M_used",    "mean"),
    P_feas   = ("P_feas",    "mean"),
    eta_eff  = ("eta",       "mean"),
    E_all    = ("mean_all",  "mean"),
    E_feas   = ("mean_feas", "mean"),
    med_feas = ("med_feas",  "mean"),
    gap_all  = ("gap_all",   "mean"),
    gap_feas = ("gap_feas",  "mean"),
    q_feas   = ("q_feas",    "mean"),
).round(3)

## gap_feas is the headline energy metric: (<E_o|feas> - E_min_feas) / (E_uniform_feas - E_min_feas).
##   0  -> the circuit sits on the best feasible state
##   1  -> uniform over the feasible set, i.e. the objective had NO influence at all
## It is scale-free (so it pools across instances, unlike the raw <E_o>) and continuous (so it is
## not quantized by the small feasible sets at n = 6, 9, unlike a percentile rank).
'''


CELL_OVERVIEW = r'''__TAG__ panel 1 -- feasibility, eta_eff and the scale-free energy gap vs size
## Solid = M* (ours), dashed = M_L1 (naive). Colour = depth p.
## gap_feas = (<E_o|feas> - E_min_feas)/(E_uniform_feas - E_min_feas) is the finer probe: eta_eff
## saturates at 0.75 for ANY blind projector onto the feasible set (E_f is the 75th percentile of
## that same spectrum), whereas gap_feas = 1 pins "blind" exactly and 0 pins "optimal".

ARMS = {"M*": dict(ls="-", filled=True), "M_L1": dict(ls="--", filled=False)}
COL  = {1: "#4C72B0", 2: "#55A868", 3: "#C44E52"}
BITS = (6, 9, 12, 15)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
panels = [("P_feas",   r"$P_{\rm feas}$",   "feasibility"),
          ("eta",      r"$\eta_{\rm eff}$", r"$\eta_{\rm eff}$"),
          ("gap_feas", r"$(\langle E_o\rangle_{\rm feas}-E_{\min})/(E_{\rm unif}-E_{\min})$",
                       "objective quality (lower = better)")]

for ax, (col, ylab, title) in zip(axes, panels):
    for arm, st in ARMS.items():
        for p in (1, 2, 3):
            sub = df_M[(df_M["arm"] == arm) & (df_M["layers"] == p)]
            means = [sub[sub["bits"] == nb][col].mean() for nb in BITS]
            ax.plot(BITS, means, st["ls"], marker="o", color=COL[p], lw=1.8, ms=6,
                    markerfacecolor=(COL[p] if st["filled"] else "white"))
    ax.set_xticks(BITS)
    ax.set_xlabel(r"$n_{\rm bits}$")
    ax.set_ylabel(ylab)
    ax.set_title(title)

axes[1].axhline(0.75, color="grey", ls=":", lw=1.2)
axes[1].text(6.1, 0.755, "blind-projector floor (0.75)", fontsize=7.5, color="grey", va="bottom")
axes[2].axhline(1.0, color="grey", ls=":", lw=1.2)
axes[2].set_ylim(top=1.02)                    # headroom so the label clears the axes title
axes[2].text(6.1, 0.996, "blind: uniform over feasible set", fontsize=7.5, color="grey", va="top")

handles = ([Line2D([], [], color=COL[p], marker="o", lw=1.8, label=f"$p={p}$") for p in (1, 2, 3)]
           + [Line2D([], [], color="k", ls=st["ls"], marker="o", lw=1.8,
                     markerfacecolor=("k" if st["filled"] else "white"), label=arm)
              for arm, st in ARMS.items()])
axes[0].legend(handles=handles, fontsize=8, ncol=2, framealpha=0.9)
fig.suptitle(r"PO / optimality, ma-QAOA + INTERP, $E_f=P_{75}$:   $M^*$ (ours) vs $M_{\ell_1}$ (naive)",
             y=1.02)
fig.tight_layout()
fig.savefig("misc_plots/M_L1_vs_star_overview.pdf", bbox_inches="tight")
fig.savefig("misc_plots/M_L1_vs_star_overview.png", dpi=170, bbox_inches="tight")
plt.show()
'''


CELL_DIST = r'''__TAG__ panel 2 -- the sampled-energy DISTRIBUTION for one representative cell
## Left  : unconditional, i.e. every state the circuit returns, infeasible ones included.
## Right : conditioned on feasibility, which separates "does it optimise?" from "is it feasible?".
## Grey = the instance's own spectrum (the uniform-sampling reference); vertical lines are the
## probability-weighted means.

N_SHOW, P_SHOW, V_SHOW = 12, 3, 42     # edit to inspect another cell

runs  = {arm: cmpML1.load_run(folder, N_SHOW, P_SHOW, V_SHOW)
         for arm, folder in [("M*", cmpML1.D_STAR), ("M_L1", cmpML1.D_L1)]}
E_obj = runs["M*"]["E_obj"]
feas  = runs["M*"]["feas"]
E_f   = runs["M*"]["E_f"]
COLA  = {"M*": "#4C72B0", "M_L1": "#C44E52"}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, cond in zip(axes, [False, True]):
    mask = feas if cond else np.ones_like(feas, dtype=bool)
    bins = np.linspace(E_obj[mask].min(), E_obj[mask].max(), 60)

    ax.hist(E_obj[mask], bins=bins, weights=np.full(mask.sum(), 1 / mask.sum()),
            color="lightgrey", label="uniform reference", zorder=0)
    # M* is drawn thick-solid and M_L1 thin-dashed ON TOP: the two distributions coincide to ~1e-8,
    # so equal linewidths would hide one curve entirely and read as a plotting bug.
    styles = {"M*": dict(lw=4.0, ls="-", alpha=0.55), "M_L1": dict(lw=1.6, ls="--", alpha=1.0)}
    for arm, r in runs.items():
        w = r["probs"][mask]
        w = w / w.sum()                     # renormalize: on the right panel this is P(.|feasible)
        mean = np.average(E_obj[mask], weights=w)
        med  = cmpML1.weighted_quantile(E_obj[mask], w, 0.5)
        ax.hist(E_obj[mask], bins=bins, weights=w, histtype="step", color=COLA[arm],
                zorder=2, label=f"{arm}   mean {mean:+.4f},  median {med:+.4f}", **styles[arm])
        ax.axvline(mean, color=COLA[arm], lw=1.2, zorder=3, **{"ls": styles[arm]["ls"]})
    ax.axvline(E_f, color="k", ls="--", lw=1.2, zorder=4)
    ax.text(E_f, ax.get_ylim()[1] * 0.95, r"  $E_f=P_{75}$", fontsize=8, va="top")
    ax.set_xlabel(r"$E_{\rm obj}$")
    ax.set_ylabel("probability")
    ax.set_title(("conditioned on feasibility" if cond else "all returned states")
                 + f"    ($n={N_SHOW}$, $p={P_SHOW}$, vseed {V_SHOW})")
    ax.legend(fontsize=8, framealpha=0.9)

fig.tight_layout()
stem = f"misc_plots/M_L1_vs_star_dist_n{N_SHOW}_p{P_SHOW}_v{V_SHOW}"
fig.savefig(stem + ".pdf")
fig.savefig(stem + ".png", dpi=170)
plt.show()
'''


CELL_PAIRED = r'''__TAG__ panel 3 -- paired per-instance verdict
## The two arms share the instance AND the circuit seed, so every (n, p, vseed) cell is a matched
## pair: the comparison can be read instance by instance, not only in the mean.

key  = ["bits", "layers", "vseed"]
pair = (df_M[df_M.arm == "M*"].set_index(key)
        .join(df_M[df_M.arm == "M_L1"].set_index(key), lsuffix="_star", rsuffix="_L1"))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
specs = [("eta_star",      "eta_L1",      r"$\eta_{\rm eff}$",             "higher"),
         ("gap_feas_star", "gap_feas_L1", r"energy gap (feasible)",        "lower")]
for ax, (a, b, lab, better) in zip(axes, specs):
    for p in (1, 2, 3):
        s = pair.xs(p, level="layers")
        ax.scatter(s[a], s[b], s=34, color=COL[p], alpha=0.75, edgecolors="none", label=f"$p={p}$")
    lo, hi = min(pair[a].min(), pair[b].min()), max(pair[a].max(), pair[b].max())
    pad = 0.04 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(lab + r"   with $M^*$")
    ax.set_ylabel(lab + r"   with $M_{\ell_1}$")
    wins = int((pair[a] > pair[b]).sum() if better == "higher" else (pair[a] < pair[b]).sum())
    ties = int((pair[a] == pair[b]).sum())
    ax.set_title(f"{lab}:  $M^*$ better in {wins}/{len(pair)}  ({ties} ties)")
    ax.legend(fontsize=8, framealpha=0.9)

fig.suptitle(r"matched pairs (same instance, same circuit seed) -- $M^*$ wins below the diagonal "
             r"on the right, above it on the left", y=1.02, fontsize=10)
fig.tight_layout()
fig.savefig("misc_plots/M_L1_vs_star_paired.pdf", bbox_inches="tight")
fig.savefig("misc_plots/M_L1_vs_star_paired.png", dpi=170, bbox_inches="tight")
plt.show()

## compact numerical verdict, pooled over the sizes where the feasible set is big enough to matter
big = pair.reset_index().query("bits >= 9")
print("pooled over n >= 9   (30 matched pairs per depth)\n")
for p in (1, 2, 3):
    s = big[big.layers == p]
    print(f"  p={p}:  P_feas {s.P_feas_star.mean():.3f} -> {s.P_feas_L1.mean():.3f}"
          f"   |  eta_eff {s.eta_star.mean():.3f} -> {s.eta_L1.mean():.3f}"
          f"   |  gap_feas {s.gap_feas_star.mean():.3f} -> {s.gap_feas_L1.mean():.3f}"
          f"   |  gap_all {s.gap_all_star.mean():.3f} -> {s.gap_all_L1.mean():.3f}")
'''


CELL_SHARE = r'''__TAG__ panel 4 -- WHY the two arms coincide: the objective's share of the trained Hamiltonian
## The circuit never sees M's overall scale: the module's two-step scaling reduces exactly to
## Q/max|Q|, so the only thing M controls is how much of the normalized cost Hamiltonian is
## objective rather than penalty. share_feas is the energy spread of E_obj ACROSS THE FEASIBLE SET
## divided by max|Q| -- inside that set the penalty is identically zero, so this is the whole signal
## available to tell two feasible states apart (roughly the phase, in rad, a gamma ~ 1 cost layer
## can impart between them).
##
## Reference line: tolerance_opt = 1e-4, the Adam *cost-convergence* tolerance used in training.
## (It is not a resolution floor -- the optimizer does not have a 1e-4 "blind spot" -- but a share
## this far below it means training stops long before the objective term has moved the cost.) Going
## from M* to M_L1 moves the share down by the same 4-8x as the penalty goes up, which at this
## altitude changes nothing: hence the null result.
##
## THE FIX IS THE NEXT SECTION. The altitude is entirely due to the span proxy being evaluated at
## M0 = L1_hot(T_norm); setting M0 = 0 drops M by 3-5 decades, lifts the share to ~1e-2, and the
## comparison stops being degenerate. See '## M0 = 0: the span-proxy fix' below.

df_S = cmpML1.add_shares(df_M)
TOL  = 1e-4      # tolerance_opt passed to run_ma_qaoa_with_solution

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

for arm, st in ARMS.items():
    for p in (1, 2, 3):
        sub = df_S[(df_S["arm"] == arm) & (df_S["layers"] == p)]
        means = [sub[sub["bits"] == nb]["share_feas"].mean() for nb in BITS]
        ax.plot(BITS, means, st["ls"], marker="o", color=COL[p], lw=1.8, ms=6,
                markerfacecolor=(COL[p] if st["filled"] else "white"))
ax.axhline(TOL, color="k", ls="-.", lw=1.3)
ax.text(6.1, TOL * 1.25, "Adam convergence tolerance ($10^{-4}$)", fontsize=8, va="bottom")
ax.set_yscale("log")
ax.set_xticks(BITS)
ax.set_xlabel(r"$n_{\rm bits}$")
ax.set_ylabel(r"objective share of $Q/\max|Q|$   (feasible-set spread)")
ax.set_title("the objective is invisible at BOTH penalties")
ax.legend(handles=handles, fontsize=8, ncol=2, framealpha=0.9, loc="lower left")

## right: the null result plotted against its cause -- the arms only begin to differ, and then only
## at the 1e-3 level, in the cells where the share is largest (n = 6).
key  = ["bits", "layers", "vseed"]
pr   = (df_S[df_S.arm == "M*"].set_index(key)
        .join(df_S[df_S.arm == "M_L1"].set_index(key), lsuffix="_star", rsuffix="_L1")).reset_index()
for nb in BITS:
    s = pr[pr.bits == nb]
    ax2.scatter(s.share_feas_star, (s.eta_L1 - s.eta_star).abs(), s=30, alpha=0.75,
                edgecolors="none", label=f"$n={nb}$")
ax2.axvline(TOL, color="k", ls="-.", lw=1.3)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"objective share at $M^*$")
ax2.set_ylabel(r"$|\eta_{\rm eff}(M_{\ell_1}) - \eta_{\rm eff}(M^*)|$")
ax2.set_title("paired difference vs. available signal")
ax2.legend(fontsize=8, framealpha=0.9)

fig.tight_layout()
fig.savefig("misc_plots/M_L1_vs_star_share.pdf", bbox_inches="tight")
fig.savefig("misc_plots/M_L1_vs_star_share.png", dpi=170, bbox_inches="tight")
plt.show()

print(df_S.groupby(["bits", "layers", "arm"])[["M_used", "share_entry", "share_feas"]]
      .mean().to_string(float_format=lambda v: f"{v:.3e}"))
'''


CELLS = [CELL_TABLE, CELL_OVERVIEW, CELL_DIST, CELL_PAIRED, CELL_SHARE]


def main():
    nb = json.load(open(NB))
    cells = nb["cells"]

    def find_anchor(cs):
        return next(i for i, c in enumerate(cs)
                    if c["cell_type"] == "markdown" and MARKER in "".join(c["source"]))

    anchor = find_anchor(cells)

    # drop a previously inserted block (and the original '# add here' stub) so reruns are idempotent
    keep, removed = [], 0
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if i > anchor and (src.startswith(TAG) or src.strip() == "# add here"):
            removed += 1
            continue
        keep.append(c)
    cells = keep
    anchor = find_anchor(cells)

    new = []
    for src in CELLS:
        new.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                    "source": src.replace("__TAG__", TAG).splitlines(keepends=True)})

    cells[anchor + 1: anchor + 1] = new
    nb["cells"] = cells
    json.dump(nb, open(NB, "w"), indent=1)
    print(f"anchor '{MARKER}' at cell {anchor}; removed {removed} old cell(s), "
          f"inserted {len(new)}; notebook now has {len(cells)} cells")


if __name__ == "__main__":
    main()
