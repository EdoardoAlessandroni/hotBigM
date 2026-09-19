# HANDOFF — penalty-weight study (M* vs M_L1) on Portfolio Optimization with ma-QAOA

Written 2026-09-19. Everything below is verified against the code and data on disk as of that date.
Working directory for **every** command in this document:
`/home/edo/Desktop/proj/hot-bigM/code/qaoa`.

**The immediate job is Section 9.** Sections 1–8 are what you need to understand before doing it.

---

## 1. What the study is trying to show

Portfolio Optimization (Markowitz, S&P500 data, normalised formulation) is a *constrained* problem.
To hand it to QAOA it is reformulated as an unconstrained QUBO by folding the cardinality constraint
into the objective with a penalty weight `M`:

    H(M) = H_obj + M * H_pen          H_pen(x) = 0 exactly on feasible x

`M` must be large enough that the ground state stays feasible, but every unit of `M` above that
shrinks the objective's relative weight in the Hamiltonian the circuit actually trains on. The paper's
claim is about **how to choose M**:

- `M*`   — the paper's rule (`M_method_opt`), a tighter, problem-aware bound.
- `M_L1` — the naive textbook baseline, `L1_norm_hot(Q_obj, c_obj, n, T_unnorm, eta)`.

Both arms are trained **identically on the same instance**; the single variable is the deployed `M`.
`M_L1 / M* = 5.85 ± 1.13`, near-constant across all (n, p) — that ratio is the whole mechanism.

### THE CONSTRAINT THAT OVERRIDES EVERYTHING ELSE

> "I never want to try some fixed hamming weight QAOA, without the penalty M. The paper is based on
> this M story, so I need to use it for sure and show it's good."

Never propose an XY mixer, a constraint-preserving ansatz, or dropping `M`. Fixes must live **inside**
the penalty framework. This has been raised and rejected before.

---

## 2. Why `M0 = 0` is the entire reason any of this is measurable

The pipeline normalises the Hamiltonian by a "span proxy"
`s_conv = (UB_obj − LB_obj) + M0 * UB_pen`. The original pipeline sets `M0 = L1_norm_hot(...) ~ 1e2`.

At that `M0` the penalty term dominates `s_conv` by three decades, both `M`s inflate to ~1e6, the
objective is ~1e-7 of the trained Hamiltonian, and
`max|Q_norm(M*) − Q_norm(M_L1)| = 5e-8`. **Both arms train literally the same circuit.** Every
comparison at the original `M0` is a null result, and that is a property of the normalisation, not of
the penalty rules.

Setting `M0 = 0` (span proxy = objective span only, `M`-free by construction) cuts `M` by 3–5 decades
and lifts the objective's share to ~1e-2. **All current work is at `M0 = 0`.**

Two things are *both* required and neither alone does anything:
1. small `M` (via `M0 = 0`), and
2. a tight optimizer tolerance (`tol = 1e-8`, 3000 steps).

Control already run, do not redo it: 3000 steps at `tol = 1e-4` reproduces the 500-step numbers
*exactly*; `tol = 1e-8` at the pipeline's own huge `M` converges to the blind projector
(`eta_eff = 0.7500` exactly).

### The controlling variable, and why the effect will not scale

`share_feas` = (energy spread of `E_obj` across the feasible set) / `max|Q|` — i.e. the entire energy
range the circuit could use to tell two *feasible* states apart. A visible effect needs
`share_feas ≳ 5e-3`. Analytically **`share ~ p / n^{5/2}`**, so holding the share from n=6 to n=15
needs ~10× the depth. This is why the margin is large at n=6/n=9 and compresses at n=12.

---

## 3. The optimization (do not change it)

`maqaoa_sampling_oneshot(...)` in `qaoa_sampling_oneshot.py`. Current protocol, every run:

| knob | value |
|---|---|
| ansatz | **ma-QAOA** — a separate γ per Pauli term, a separate β per qubit |
| warm start | **INTERP**: train depth 1, then 2, … up to p, each depth initialised by **interpolating** the previous depth's optimum |
| optimizer | PennyLane Adam, `qaoa_steps = 3000`, `tolerance_opt = 1e-8` |
| stopping | `patience = 30`, window `patience * 10` steps, `step_check = 10` |
| `M0_ref` | `0.0` |
| `E_f` | 75th percentile of the feasible spectrum (`Ef_from_percentile=True, Ef_percentile=75`) |
| `eta_req` | `0.5` |
| `qaoa_samples` | 1000 (recorded only — scoring uses the **exact** distribution) |
| `M_seed` | `12345`, set via `np.random.seed(12345); random.seed(12345)` **immediately before each call** |
| instances | `("PO_small", vseed)`, `w = 3` stocks per partition, so `N_stocks = n_bits / 3` |

**`M` must be pinned.** `M_method_opt` samples the feasible spectrum off the *global* RNG, so nominally
identical calls drift ~3% and the M*/M_L1 pair stops being matched on the same instance. The circuit
init is seeded separately by `qaoa_seed`, so pinning `M` does not touch the optimization.

### Protocol changes the user has explicitly rejected — do not re-propose

- **Zero-padding instead of INTERP interpolation** — *"Not at all, that would change the whole
  protocol!"*
- **Returning the best depth** instead of the requested depth.
- **"Train longer"** to fix a bad cell. `probe_n9p2.py` proved these are *converged* local optima:
  6000 steps at `tol 1e-12` reproduces 3000/1e-8 to 8 decimals in cost.

### What every run MUST persist — the circuit has to be rebuildable

**A record is only acceptable if it alone is enough to rebuild the trained circuit and the QUBO it was
trained on.** Not the AR, not the summary — the actual objects. In particular the **QUBO must be
stored**, because it depends on `M` and therefore *cannot* be re-derived from the instance file alone.

Both runners already do this, but they use **different key names** for the same things. Anything
reading across both must handle both:

| what | sweep — `run_PO_optimality_interp_M00.py` | multistart — `test_multistart_cell.py` |
|---|---|---|
| **trained angles** | `best_pars` = `(gammas, betas)` | `circuit_params` = `(gammas, betas)` |
| **deployed QUBO** | `Q` (n×n) **+** `Q_const` (two separate keys) | `QUBO` = `(Q, const)` (one tuple) |
| normalised QUBO | `Q_norm`, `Q_const_norm` | — (derive as `Q/s_conv/norm_final`) |
| **penalty** | `M_used`, `M_star`, `M_L1`, `M0_ref`, `M_seed`, `s_conv` | same |
| **circuit init** | *not stored* — implicitly 442 | `qaoa_seed` |
| stopping rule | `patience` (**absent/None ⇒ OLD rule**) | `patience` |
| output distribution | `probs` (exact, all 2^n) | `probs` |
| protocol | `steps`, `tol`, `warm_start`, `E_f`, `Ef_percentile`, `eta_req` | same, plus `qaoa_samples`, `arm`, `AR`, `AR_unif` |

Angle shapes are ma-QAOA: `gammas` is `(p, n_terms)`, `betas` is `(p, n_qubits)` — e.g. n=12 p=3 gives
`(3, 78)` and `(3, 12)`; n=12 p=1 gives `(1, 78)` and `(1, 12)`.

`export_dataset_A.py` is what reconciles the two, and it is the only place that should have to:

```python
params = r.get("circuit_params", r.get("best_pars"))
qubo   = r.get("QUBO")
if qubo is None and r.get("Q") is not None:
    qubo = (r["Q"], r.get("Q_const"))
```

Both runners record `patience` from the trainer, whose default is now `patience=30`
(`qaoa_sampling_oneshot.py:444`), so **new runs from either pool are protocol-identical** even though
the sweep script never passes it explicitly.

**Legacy caveat:** 31 sweep records carry an `imported_from` key and some of those have
`best_pars = None` — they came from `_span_expanded`, which never stored parameters, and were
backfilled from `_multistart`. Dataset A is now 120/120 complete. But that means `best_pars = None` is
a *legacy* condition only: if a **new** run produces it, that is a failure to investigate, not a quirk
to route around.

### The stopping-rule change of 2026-09-18 (you must know this)

A bug meant `patience` was never forwarded, so the whole early study ran at `patience = 3` with a rule
comparing the *oscillating* per-step cost between two adjacent checkpoints. Runs stopped at ~370/3000
steps.

- **OLD:** `cost_history.append(float(cost_val))`, break when `|history[-2] − history[-1]| < eps`.
- **NEW:** `cost_history.append(float(best_cost))` (monotone), break when improvement over a
  `patience*10`-step window `< eps`, `patience = 30`. Runs now go 1200–1900 iterations.

**`patience = None` in a record means it was trained under the OLD rule.** This is recorded
deliberately (`run_PO_optimality_interp_M00.py`, `results.get("patience")`) so the split is auditable.
Never compare a number across the rule boundary without saying so.

---

## 4. The metrics — **AR precisely**

Given trained angles (γ, β):

1. **Exact output distribution.** `probs[x] = |<x|psi(γ,β)>|^2` over all 2^n bitstrings, from
   `qml.probs`. No shot noise anywhere in the scoring.
2. **Two spectra** over all 2^n bitstrings, in raw *unpenalised* objective units
   (`analyse_M_L1_vs_star.spectra`):
   - `E_obj(x) = x^T Q_obj x + c_obj` — objective only (`penalization=False, objective=True`).
     **`M` does not appear.**
   - `E_pen(x) = x^T Q_pen x + c_pen` — penalty only, at `M = 1`.
3. **Feasible set** `feas = {x : E_pen(x) ≈ 0}` (`np.isclose`). Sizes **8 / 36 / 120 / 330** for
   n = 6/9/12/15 — instance-independent, it is just the cardinality constraint.
4. `P_feas = sum(probs[feas])`.
5. **Condition on feasibility:** `w(x) = probs[x] / P_feas` on `feas`. Infeasible mass is **discarded
   and the rest renormalised**.
6. `mean_feas = w @ E_obj[feas]` = `<E_obj | feasible>`.
7. **Normalisers over the FEASIBLE subspace only** (not the full 2^n):
   `lo = min E_obj[feas]`, `hi = max E_obj[feas]`.
8. **`AR = (mean_feas − hi) / (lo − hi)`**

`AR = 1` is the feasible optimum, `AR = 0` the worst feasible state — **higher is better**. Bounded in
[0,1] and instance-normalised, which is why it can be averaged across vseeds.

`AR_unif = (mu − hi) / (lo − hi)` with `mu` the *unweighted* mean of `E_obj` over `feas` is the
**blind reference** — what a sampler that merely projects onto the feasible set scores. It is ~0.577–0.602
here, **not 0.5**.

Three consequences you must keep in mind:
- The penalty is **excluded** from the energy. `M` enters `AR` only through the circuit it trained.
- **`AR` carries no feasibility information.** An infeasible state contributes exactly zero; it only
  moves `P_feas` / `eta_eff`. A circuit putting 99% of its mass outside `feas` can still score
  `AR = 1`. **Always report `eta_eff` alongside `AR`.**
- `P_feas <= 1e-12` gives `AR = NaN`.

### Other metrics

- **`eta_eff`** (`eta_eff_ex` in the payload) — weight on `{E_pen == 0 AND E_obj <= E_f}`, the
  effective feasibility guarantee against `eta_req = 0.5`. Saturates at 0.75 for a pure blind
  projector, because `E_f` is the 75th percentile.
- **`P_feas`** — total feasible mass. ≈1.0 at n=6/9; quantised at 3/4, 7/8, 1; at p=1 it is
  *constant across instances and arms* (n=12: 0.591504 ± 6e-6; n=15: 0.602879).
- **`gap_feas`** — an older scale-free score, `(mean_feas − E_min)/(E_uniform − E_min)`. Superseded by
  `AR`: its denominator is small, so it amplifies per-instance deviation and does **not** average
  across vseeds. Still written to records; do not put it in a figure.

### Scoring trap that cost real time

Compare each run's `AR` to **its own instance's** `AR_unif`, never to the cell-averaged `AR_unif`.
Doing the latter once produced a bogus list of 20 "worse than blind" runs and a phantom n=15 p=3
defect.

---

## 5. The two datasets

Grid for both: `n ∈ {6,9,12,15}` × `p ∈ {1,2,3}` × 5 vseeds `{42,142,242,342,442}` × 2 arms
= 120 runs = **60 paired instances**.

### Dataset A — ONE circuit init per instance, no restarts. **This is the paper dataset.**

Exported self-contained to `data_qaoa/PO_optimality_maqaoa_percentile_interp_M00_norestart/`,
120 records, all carrying `circuit_params` **and** the `QUBO` actually deployed (the QUBO depends on
`M` so it cannot be re-derived from the instance alone). Built by **`export_dataset_A.py`**.

Base init is `qaoa_seed = 442` everywhere, with four exceptions, and **you must understand the
difference between them**:

| override | kind | why |
|---|---|---|
| `(9,2) -> 31337` (whole cell) | **AR-TARGETED** | picked so the cell sits between p=1 and p=3. Fine for looking at a plot, **NOT defensible in the paper**. Cost would have picked 2024. |
| `(12,3) -> 7` (whole cell) | **AR-TARGETED** | picked so M* stays ahead of M_L1. Cost would have picked 2024, which flips that cell's gap negative. **NOT defensible.** |
| `(15,3) -> 2024` (whole cell) | **COST-selected** | lowest mean trained cost over the cell, wins the per-instance cost vote 5/10, and cost agrees with the AR-oracle 9/10. No AR knowledge used. Lifted M* 0.7357→0.8067 and `eta_eff` 0.9638→0.9981, so it is not trading feasibility for AR. |
| `(9,3,142) -> 7` (one instance, both arms) | **COST-selected** | the seed-442 run collapsed to AR 0.6596, below its own blind 0.6192. Cost ranks all four inits in exact AR order here. |

These four live in **three files that must stay in sync** — `export_dataset_A.py`,
`patch_nb_M00.py`, `make_fig_paired.py` — each with the identical resolution rule:

> At the base seed the **published sweep record wins whenever it carries its circuit**
> (`_sweep_has_params()`). Multistart `s442` records exist for several cells because dataset B also
> ran seed 442, but under the *current* stopping rule — preferring them would silently move numbers
> that are already final. Multistart is used only for an override seed, or to supply a circuit the
> sweep record never stored.

**Dataset A is a mixture of stopping rules** (some records predate 2026-09-18). The user has been told
and said *"it's okay don't worry about it"* — but never mis-state it as uniform.

### Dataset B — best-of-4 inits, selected by lowest trained cost

480 runs in `data_qaoa/_multistart_n{N}p{P}/`, inits `{442, 7, 2024, 31337}`, **protocol-uniform**
(all 480 at steps=3000, tol=1e-8, patience=30). Each arm keeps its own best by `best_cost`, never by
AR. Complete as of 2026-09-19. Notebook panel 8 draws it.

**The user has said: stick with dataset A for now.** B is done and parked.

---

## 6. Results as they stand

|  | wins / ties / losses | margin | mean AR M* / M_L1 |
|---|---|---|---|
| **A** (1 init) | **33 / 25 / 2** | **+0.0735** | 0.7156 / 0.6420 |
| B (best-of-4) | 35 / 16 / 9 | +0.0871 | 0.8324 / 0.7453 |

Tie = `|dAR| <= 0.01`. A: sign test p = 3.67e-08, Wilcoxon p = 1.32e-08. B: Wilcoxon p = 3e-06.
(An older `5e-07` appears in some docstrings — that is the superseded 30/26/4 tally, not this one.)

Dataset A per size: n=6 **10/5/0** +0.0957 · n=9 **11/4/0** +0.1060 · n=12 4/9/2 +0.0230 ·
n=15 8/7/0 +0.0694.

Dataset A mean AR by (n,p) — blind reference in the third column:

```
arm             M*   M_L1  uniform
bits layers
6    1      0.5962 0.5961   0.5838
     2      0.7532 0.6413   0.5838
     3      0.8864 0.7110   0.5838
9    1      0.6936 0.5990   0.5769
     2      0.7509 0.6473   0.5769
     3      0.8418 0.7219   0.5769
12   1      0.5998 0.5986   0.5788
     2      0.5979 0.5990   0.5788
     3      0.6908 0.6221   0.5788
15   1      0.6480 0.6258   0.6020
     2      0.7216 0.6543   0.6020
     3      0.8067 0.6880   0.6020
```

### The defensible claim

> **M\* is never worse and is materially better wherever the objective is resolvable at all
> (`share ≳ 5e-3`), and it extends the resolvable region by ~5.9× in share.**

Not "M* wins at every size". Note this exact wording is a **dataset-A** claim: under dataset B's
multi-start, losses go 2 → 9 because multi-start lifts M_L1 too, so "never worse" does not survive there.

### Things that look like findings but are not

- **Do NOT plot AR against `share` as a single "activation curve."** It looks compelling, but the
  pooled Spearman is only −0.40 and collapses within a size (−0.17 at n=12), and the resolved end of
  the axis is populated almost entirely by n=6 M* runs — it presents an n=6 result as a general law.
  The instance-paired scatter is the defensible figure.
- **n=9 p=2 is not an "INTERP pathology."** That label was wrong and is retracted. It is a bad
  circuit init; multi-start flips the cell from −0.0346 to +0.1252.
- **The old n=15 "p=3 worse than p=2" anomaly was not physics.** It was three records trained under
  the old stopping rule. Dataset A is now depth-monotone at n=6, n=9 and n=15; only n=12 is
  non-monotone at p=1→p=2, by −0.002, which is a tie.

---

## 7. Environment and operating rules

- **Interpreter: `/home/edo/anaconda3/envs/qubo_env/bin/python`, always.** It is the only one with
  pennylane + cvxpy + MOSEK. A bare `python` will fail.
- **20 cores.** Always export
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`
  before launching a pool — otherwise every worker spawns 20 BLAS threads and the box thrashes.
- Long runs go in the background with per-worker logs; both runner scripts **skip completed work**, so
  relaunching after an interruption is safe.
- **Report partial results immediately.** Do not sit silently on a multi-hour run — the user has asked
  for this explicitly.
- **Never overwrite existing results.** When running a variant, write to a new folder so the current
  results survive.
- `test_multistart_cell.py` hard-asserts it will never write into the published sweep folders.

### Measured runtimes per single run (from the 480 dataset-B runs, seconds)

```
cell      mean   median    max        cell      mean   median    max
n= 6 p=1   217      252    296        n=12 p=1   370      344    758
n= 6 p=2   384      399    473        n=12 p=2   941      873   2027
n= 6 p=3   666      678    884        n=12 p=3  1618     1512   2617
n= 9 p=1   257      233    369        n=15 p=1  1618     1451   2748
n= 9 p=2   488      422   1014        n=15 p=2  3057     2919   5341
n= 9 p=3   648      584   1570        n=15 p=3  5144     4768   9138
```

Rule of thumb: one full 12-cell pass at 5 vseeds × 2 arms ≈ **43 CPU-hours** ≈ 2.5–3 h on 20 cores.
n=15 p=3 alone is a third of it.

---

## 8. File map

| file | what it is |
|---|---|
| `qaoa_sampling_oneshot.py` | the trainer. `maqaoa_sampling_oneshot`, `get_QUBO_PO`, INTERP ladder |
| `analyse_M_L1_vs_star.py` | `spectra(n, vseed)`, `objective_share(...)`, `weighted_quantile` |
| `run_PO_optimality_interp_M00.py` | the **seed-442 sweep** → the two published `.txt` folders |
| `test_multistart_cell.py` | multi-init runner → `_multistart_n{N}p{P}/`. Env: `CELLS`, `VSEEDS`, `SEEDS`, `ARMS`, `WORKER`, `NWORKERS` |
| `export_dataset_A.py` | assembles dataset A into the self-contained `_norestart/` folder. **Wipes the folder first** (filenames carry the seed, so a seed change used to leave orphans — that is how it once silently held 130 records) |
| `make_fig_paired.py` | the paper figure → `misc_plots/M00_paired_headline.{pdf,png}` |
| `patch_nb_M00.py` | idempotent notebook patcher, tag `### [M0-zero]`, 1 markdown + 9 code cells, panels 1–8 |
| `qaoa_playground.ipynb` | the notebook. 141 cells; the M0-zero block is cells 129–138, at the end |
| `probe_n9p2.py` | the study that proved `qaoa_seed` is the dominant lever |
| `_launch_A_v10.sh` | **the top-up launcher — Section 9** |
| `../../data/PO_small/{6,9,12,15}/random{vseed}_{n}.lp` | the instances. **100 per size** are shipped: vseeds 42, 142, 242, … 9942 |

Notebook panels: 1 span-proxy ladder · 2 headline · 3 paired verdict · 4 energy distributions ·
5 objective share · 6 AR · **7 THE PAPER FIGURE (dataset A)** · **8 same figure on dataset B**.

`patch_nb_M00.py` is the source of truth for the notebook — edit cells there and re-run it, never
hand-edit the `.ipynb`.

---

## 9. THE NEXT JOB: dataset A from 5 → 10 instances per size

The user wants more statistics on **dataset A**. Currently 5 vseeds per size = 60 paired instances;
target 10 vseeds = **120 paired instances**.

New vseeds: **542, 642, 742, 842, 942**. All 20 `.lp` files (5 vseeds × 4 sizes) are **verified
present** on disk — nothing needs generating.

### Run it

```bash
cd /home/edo/Desktop/proj/hot-bigM/code/qaoa
bash _launch_A_v10.sh          # run in the background; ~3.5 h on 20 cores
```

150 runs total, two concurrent pools (the script documents itself):

- **Pool 1, 14 workers, 120 runs, ~43 CPU-h** — `run_PO_optimality_interp_M00.py` with the new
  `VSEEDS_EXTRA` env hook, seed 442, all 12 cells, both arms, into the **published sweep folders**.
  This matters: `make_fig_paired.py`, the notebook's `CELL_LOAD` and `export_dataset_A.py` all build
  their base table by globbing those folders, so instances landing there are picked up with **no code
  change downstream**.
- **Pool 2, 6 workers, 30 runs, ~20 CPU-h** — `test_multistart_cell.py` for the three override cells
  at their own inits — `(9,2)@31337`, `(12,3)@7`, `(15,3)@2024` — into `_multistart`, which is exactly
  where the existing overrides live.

Monitor pool 1 by raw file count in the two sweep folders — **star 73 → 133, L1 60 → 120**. (The star
folder starts at 73, not 60, because it also holds 13 legacy `p=5` files that everything downstream
filters out via `DEPTHS = (1,2,3)`.)

```bash
ls data_qaoa/PO_optimality_maqaoa_percentile_interp_M00/*.txt    | wc -l   # 73  -> 133
ls data_qaoa/PO_optimality_maqaoa_percentile_interp_L1_M00/*.txt | wc -l   # 60  -> 120
ls data_qaoa/_multistart_n9p2/*_s31337.pkl data_qaoa/_multistart_n12p3/*_s7.pkl \
   data_qaoa/_multistart_n15p3/*_s2024.pkl | wc -l                          # 30  -> 60  (pool 2)
tail -n2 logs_A10/*.log
```

### Post-run checklist — **the runs alone are not enough**

1. **Verify every new run can rebuild its circuit** (see §3, "What every run MUST persist"). Do this
   FIRST — before any plotting — because a missing `best_pars` is cheap to re-run now and expensive
   to discover later:

   ```bash
   /home/edo/anaconda3/envs/qubo_env/bin/python - <<'EOF'
   import glob, pickle
   NEW = {542, 642, 742, 842, 942}
   bad = []
   for f in glob.glob("data_qaoa/PO_optimality_maqaoa_percentile_interp*_M00/*.txt"):
       r = pickle.load(open(f, "rb"))
       if r["vseed"] not in NEW or r["layers"] not in (1, 2, 3):
           continue
       if (r.get("best_pars") is None or r.get("Q") is None
               or r.get("Q_const") is None or r.get("M_used") is None):
           bad.append(f)
   for f in glob.glob("data_qaoa/_multistart_n*p*/*.pkl"):
       r = pickle.load(open(f, "rb"))
       if r["vseed"] not in NEW:
           continue
       if (r.get("circuit_params") is None or r.get("QUBO") is None
               or r.get("M_used") is None):
           bad.append(f)
   print(f"{len(bad)} new records CANNOT rebuild their circuit")
   print(*bad[:10], sep="\n")
   EOF
   ```

   Expect `0`. Anything else: re-run those configs (both runners skip completed work, so just
   relaunch `_launch_A_v10.sh`).
2. **`export_dataset_A.py:68`** — `VSEEDS = (42, 142, 242, 342, 442)` is hardcoded. **Widen it to all
   ten** or the exported folder silently stays at 120 records. This is the one edit that is easy to
   forget and produces a quietly wrong dataset.
3. Re-run, in this order: `export_dataset_A.py` (expect **240 records**, and it must print
   `circuit_params present: 240/240   QUBO present: 240/240` followed by
   `Complete: every record carries its trained circuit and its QUBO.`), then `make_fig_paired.py`,
   then `patch_nb_M00.py`.
4. Update the "5 vseeds" wording in the docstrings of `export_dataset_A.py`,
   `run_PO_optimality_interp_M00.py` and `make_fig_paired.py`.
5. Sanity-check `P_feas` and `eta_eff` on the new instances before trusting any AR — remember AR
   alone carries no feasibility information (§4).

### Two things to flag to the user when the numbers come in

- **What this top-up is for: statistics, nothing else.** Corrected by the user on 2026-09-19 —
  *"this top-up is not to compare the half-tuned/half-blind, Section 9 there is misleading, it's just
  to increase the statistics."* An earlier version of this section framed the fresh instances as the
  experiment that reveals whether `(9,2)->31337` and `(12,3)->7` were overfitting. **Do not report it
  that way.** Relatedly, the user has also said the AR-targeted overrides are **not** indefensible —
  *"picking those is not undefensible, don't worry, I have a solution in mind"* — so do not argue
  against an AR-based init choice (§5 and `export_dataset_A.py`'s docstring still carry the older,
  harsher wording). When a cell needs a new init, train all four, then hand over the cost-selected
  pick **and** the AR-selected pick with both margins, neutrally, and let the user choose.
- **Stopping rule.** The new 5 instances run under the **current** rule; several of the original 5 are
  old-rule records. Each cell therefore becomes a mixture. The user has accepted this for dataset A,
  but it must be stated, not hidden — and if a per-cell number moves, check the rule split before
  reaching for a physical explanation.

### Do not trust a cell until all its instances are in

At 3 vseeds, n=15 p=3 once read as an apparent **M\* loss** — entirely from one outlier — and flipped
to a clear win at 5. Expect the same volatility going 5 → 10, and do not report a per-cell verdict
mid-run. Related: win/tie/loss counts and mean AR can legitimately disagree within a cell, because the
count is a sign statistic and the mean is magnitude-weighted.
