\Old strategy
    Test 0:
    - Normalize temperature
    - Run 100 QAOA opt steps

    Test 1:
    - Normalize temperature
    - Renormalize Q(M)
    - Run 500 QAOA opt steps (breaking before if eps < tol)

    Test 2:
    - Renormalize Q(M)
    - Run 100 QAOA opt steps

    Actions:
    Normalize temperature = Guess a normalizing factor N (so far, spectral norm of Q_obj + Q_pen + const_obj + const_pen); 
                            given required layers p, compute (normalized) temperature from paper formula and compute the unnormalized temperature = T_norm * N;
                            work with our algo in this unnormalized setting until you have QUBO(M)_unnorm;
                            Normalize the QUBO dividing by N: now in this nromalized setting the normalized temperature is the right one (coming from p)
    Renormalize Q(M) =      Before feeding it to the QAOA normalize the QUBO by dividing it by np.max(np.abs(Q))



\{problem_type}_{M_strategy}
    runs on problem_type (\in {NPP, PO, TSP_rand}) applied to M_strategy (\in {feasibility, optimality}) goal of QUBO solving.

\PO_{M_strategy}_maqaoa_{version}
    same as above, but the quantum circuit is a maQAOA instead of QAOA (layers \in {1, 2, 3} instead of {5, 10, 20}).
    Vrsion indicates, for optimality strategy, in select_Ef in the old one ('old', where it was 3e-2) or the new, current one (just \PO_{M_strategy}_maqaoa, where it is 1e-1)
    


TO REFINE:

PO_optimality_maqaoa_percentile
Ef choosen as 75% percentile; ... p = 3 is pretty sad, mostly because it's a near-deterministic sampler (effective states are abuot 1)

PO_optimality_maqaoa_percentile_interp
PO_optimality_maqaoa_percentile_interp_L1 
comparison bertween M* and Ml1, with tol = 1e-4, steps = 500, M0=Ml1(temp)

PO_optimality_maqaoa_percentile_interp_M00
PO_optimality_maqaoa_percentile_interp_L1_M00
comparison bertween M* and Ml1, with tol = 1e-8, steps = 3000, M0=0

================================================================================
EVERYTHING ABOVE THIS LINE WAS WRITTEN BY EDOARDO.
EVERYTHING BELOW IT WAS WRITTEN BY CLAUDE (2026-09-21) and describes the folders
that the notes above do not cover. Nothing below changes the meaning of anything
above it; it is documentation only.
================================================================================


ARM-PER-FOLDER CONVENTION
-------------------------
The M branch is named by the FOLDER, never only by the file:

    ..._interp_M00        /  ..._interp_L1_M00          the M0=0 sweep, M* / M_L1
    ..._interp            /  ..._interp_L1              the M0=Ml1(temp) sweep

The dataset-A export used to break this rule -- both arms lived in one
`PO_optimality_maqaoa_percentile_interp_M00_norestart/` folder, separated only by
a `star_` / `L1_` filename prefix. On 2026-09-21 it was split into

    PO_optimality_maqaoa_percentile_interp_M00_norestart_Mstar
    PO_optimality_maqaoa_percentile_interp_M00_norestart_Ml1

using the same Mstar / Ml1 arm tags that `optimized_circuits/` already used. The
filenames were NOT changed: a record is still `{star,L1}_n{n}_p{p}_v{vseed}_s{qaoa_seed}.pkl`,
so it names its own arm, and every record also carries `arm` as a field. The
folder is simply authoritative now.


THE DATASET-A EXPORT  (..._M00_norestart_Mstar / ..._M00_norestart_Ml1)
-----------------------------------------------------------------------
240 pickles, 120 per arm = 4 sizes x 3 depths x 10 vseeds x 2 arms. THE PAPER
DATASET, and it is FROZEN. It was built once by `export_dataset_A.py` from the
sweep folders and the _multistart folders; that script was DELETED on 2026-09-21
because dataset A is final and nothing should be able to rebuild (or wipe) these
folders again. They are now the primary artifact, not a cache: back them up,
and do not hand-edit them.

The init seed of every record survives inside the record itself -- the
`qaoa_seed` field and the `_s{seed}` in the filename agree on all 240 -- and so
do the trained angles (`circuit_params`) and the deployed QUBO. So the circuits
are reproducible from these folders alone. What the deleted script took with it
is only the PROSE explaining why each non-442 init was chosen; the overrides
themselves are listed in HANDOFF.md section 5.

How it differs from the sweep folder of the same arm (e.g. _L1_M00 vs
_norestart_Ml1) -- it is NOT just more vseeds, both hold 120 files on the same
4 sizes x 3 depths x 10 vseeds grid:

  * 36 of the 120 are a DIFFERENT RUN of the same instance. The sweep is always
    qaoa_seed 442; the export substitutes a different single init at the cells
    and instances listed in SEEDS / SEEDS_INSTANCE, taking it from _multistart.
    (M* arm: 45 records come from _multistart, 9 of them still at seed 442 --
    those are cases where the sweep .txt never stored its circuit parameters,
    so the run had to be re-taken from the multistart folder. Dataset A is
    therefore a mixture of stopping rules; HANDOFF.md says so and the user has
    accepted it.)
  * the records are ENRICHED and re-keyed, see below.

So "norestart" means ONE init per instance -- not "no multistart data was used",
and not "seed 442". Roughly a third of the folder comes out of _multistart.

Unlike the sweep folders (one .txt per run, no circuit) each record is
self-contained: trained angles (`circuit_params`), the `QUBO` actually deployed
AND its normalized twin `QUBO_norm` (the one the angles belong to), `M_used`,
`E_f`, `AR`, `AR_unif`, `eta_eff`, `P_feas`, the feasible span, and `qaoa_seed`.
"norestart" = ONE circuit init per instance, no multi-start. See HANDOFF.md
section 5 for the handful of non-442 inits and why each was chosen.


WORKING FOLDERS  (underscore prefix = not a sweep, not the paper grid)
----------------------------------------------------------------------
These were produced by one-question probe scripts. None of them is read by the
paper figures except where noted; they are kept as evidence for HANDOFF.md.

_multistart_n{n}p{p}   test_multistart_cell.py -- the SAME cell retrained from
                       several circuit inits (qaoa_seed in 442/7/2024/31337),
                       both arms. This is dataset B. It was also the source of
                       the non-442 dataset-A records (and of any sweep record
                       that never stored its parameters), but that link is
                       historical now -- dataset A is frozen and self-contained.
_span_variants         run_span_variants.py -- span proxy s_conv evaluated at
                       M0 in {"L1", 1, 0}, crossed with both penalty rules. This
                       is where M0=0 was first shown to matter.
_span_expanded         run_span_expanded.py -- the M0=0 result across size and
                       depth, M pinned so the pairs are genuinely matched,
                       3000 steps / tol 1e-8. Source of 18 dataset-A p=3 records
                       (M* arm) that predate the stopping-rule fix; it did NOT
                       store `best_pars`, which is what _legacy_backfill and
                       _recovered exist to repair.
_span_divisor          test_span_divisor.py -- artificially shrinking s_conv by
                       a divisor D to buy objective weight at n=12. EMPTY: the
                       run was not kept.
_tolerance_objective   test_tolerance_objective.py -- tolerance_opt swept over
                       {1e-4, 1e-8, 1e-12} at fixed instance/M/seed, testing
                       whether the objective is simply below tolerance. Read by
                       the notebook's ladder panel, which is no longer drawn.
_probe_n9p2            probe_n9p2.py -- why INTERP p=1 -> p=2 collapses at n=9
                       on exactly the instances p=1 had already solved.
_refix_n9p2            _refix_n9p2.py -- the same anomaly re-run under the FIXED
                       stopping rule (patience 30, best-cost tracked). 2 files.
_legacy_backfill       backfill_legacy.py -- re-derivation of the missing
                       `circuit_params` using `qaoa_sampling_legacy` (today's
                       trainer with only the old stopping test restored). Only
                       1 record reproduced, so this route was abandoned.
_recovered             recover_original.py -- the other attempt at the same
                       problem, searching for the init whose `best_cost` matches
                       the stored double-precision value. EMPTY: it did not
                       succeed either. See HANDOFF.md on dataset-A provenance.

_span_divisor and _recovered are empty directories, so git does not track them;
everything else here is committed.


LOOSE FILES
-----------
M_L1_vs_star_summary.pkl   analyse_M_L1_vs_star.py -- one DataFrame row per run
                           of the M*-vs-M_L1 comparison. A cache: delete it and
                           the analysis rebuilds it.
_rows_Mstar.pkl            the same script's cache of the M* rows alone.
corrs_PO.txt               pickled correlation arrays from the thermal study
                           (temperature_formula / temperature_effective /
                           norm_guess / norm_final).
thermal_study/             NPP and TSP CSVs from the same study.
qaoa_H2-2_costs_etareq_0.5_shots_100.csv
                           per-circuit Quantinuum H2-2 cost estimate,
                           HQC = 5 + (n_1q + 10*n_2q + 5*n_meas)*shots/5000.
                           The file also has a cost_usd column -- IGNORE IT,
                           costs are reported in HQC.
