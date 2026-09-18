"""Adapt the 'Noisy emulator' pipeline to the PO optimality ma-QAOA+INTERP runs.

Three things have to change, only the first of which is cosmetic:
  1. data_subdir_for must resolve the new folder from an M_strategy string.
  2. eta_eff_from_counts gates the objective test on `M_strategy == "optimality"`, which is
     FALSE for "optimality_maqaoa_percentile_interp" -- the noisy eta would silently be the
     feasibility-only number.
  3. eta_eff_from_counts scores against select_Ef(), but the percentile runs were trained
     against a per-instance E_f stored in the file. Using the wrong cutoff would compare the
     emulator to a different success criterion than the noiseless training.

Backward compatibility is preserved throughout: ("PO", "optimality") still resolves to
PO_optimality_maqaoa, and files without a stored E_f still fall back to select_Ef.
"""

import json
import shutil

NB = "qaoa_playground.ipynb"
shutil.copy2(NB, NB + ".bak_interp")
nb = json.load(open(NB))


def cell_src(i):
    return "".join(nb["cells"][i]["source"])


def set_src(i, s):
    nb["cells"][i]["source"] = s.splitlines(keepends=True)


def sub(i, old, new, count=1):
    s = cell_src(i)
    assert s.count(old) == count, f"cell {i}: found {s.count(old)} of {old[:60]!r}, expected {count}"
    set_src(i, s.replace(old, new))


# ── 1. data_subdir_for ───────────────────────────────────────────────────────────────
sub(13, '''def data_subdir_for(problem_type, M_strategy):
    """Folder of data_qaoa holding the runs we want.

    PO is ALWAYS taken from its ma-QAOA variant, never the plain PO_feasibility folder.
    (PO_optimality_maqaoa_old is the superseded select_Ef run and is not used.)
    """
    if problem_type == "PO":
        return f"PO_{M_strategy}_maqaoa"
    return f"{problem_type}_{M_strategy}"''',
    '''def data_subdir_for(problem_type, M_strategy):
    """Folder of data_qaoa holding the runs we want.

    PO is ALWAYS taken from a ma-QAOA variant, never the plain PO_feasibility folder.
    M_strategy is either the bare strategy -- which keeps the historical spelling used by
    every earlier cell -- or the full folder suffix, which is then taken as-is:

        ("PO", "feasibility")                          -> PO_feasibility_maqaoa
        ("PO", "optimality")                           -> PO_optimality_maqaoa
        ("PO", "optimality_maqaoa")                    -> PO_optimality_maqaoa
        ("PO", "optimality_maqaoa_percentile")         -> PO_optimality_maqaoa_percentile
        ("PO", "optimality_maqaoa_percentile_interp")  -> PO_optimality_maqaoa_percentile_interp

    (PO_optimality_maqaoa_old is the superseded select_Ef run and is not used.)
    """
    if problem_type != "PO":
        return f"{problem_type}_{M_strategy}"
    if M_strategy in ("feasibility", "optimality"):
        return f"PO_{M_strategy}_maqaoa"        # legacy spelling
    return f"PO_{M_strategy}"''')


# ── 2. E_f lookup + eta_eff_from_counts ──────────────────────────────────────────────
sub(13, '''def eta_eff_from_counts(counts, problem_type, M_strategy, n_bits, vseed, directory="PO_small"):''',
    '''def Ef_for_run(problem_type, M_strategy, n_bits, vseed, layers, eta_required=0.5,
               dir_data="./data_qaoa"):
    """The E_f this particular run was TRAINED against, and where it came from.

    The percentile sweeps choose E_f per instance (the 75th percentile of that instance's
    feasible objective spectrum) and record it in the training file. The older runs used the
    hardcoded select_Ef(problem_type, N_idx) and store nothing. Scoring noisy shots against
    the wrong cutoff would compare the emulator against a different success criterion than
    the noiseless number it is plotted next to, so prefer the stored value and fall back to
    select_Ef only when the file genuinely has none.
    """
    if layers is not None:
        f = (Path(dir_data) / data_subdir_for(problem_type, M_strategy) /
             f"test_{problem_type}_bits_{n_bits}_vseed_{vseed}"
             f"_layers_{layers}_etareq_{eta_required}.txt")
        if f.exists():
            with open(f, "rb") as fh:
                stored = pickle.load(fh)
            if "E_f" in stored:
                return float(stored["E_f"]), "file"
    return select_Ef(problem_type, N_idx_from_bits(problem_type, n_bits)), "select_Ef"


def eta_eff_from_counts(counts, problem_type, M_strategy, n_bits, vseed, directory="PO_small",
                        layers=None, eta_required=0.5, E_f=None, dir_data="./data_qaoa"):''')

sub(13, '''    Returns (eta_eff, n_shots, n_success).
    """
    Q_obj, const_obj, Q_pen, const_pen, E_f = obj_pen_unnormalized(
        problem_type, n_bits, vseed, directory)
''',
    '''    E_f defaults to whatever the run was trained against (see Ef_for_run): the per-instance
    percentile for the *_percentile* folders, select_Ef for the older ones. Pass `layers` so
    that lookup can find the training file; pass `E_f` explicitly to override both.

    Returns (eta_eff, n_shots, n_success).
    """
    Q_obj, const_obj, Q_pen, const_pen, _E_f_legacy = obj_pen_unnormalized(
        problem_type, n_bits, vseed, directory)
    if E_f is None:
        E_f, _ = Ef_for_run(problem_type, M_strategy, n_bits, vseed, layers,
                            eta_required=eta_required, dir_data=dir_data)

    # "optimality_maqaoa_percentile_interp" is still an optimality run -- an equality test
    # here would silently degrade it to the feasibility-only criterion
    is_optimality = "optimality" in M_strategy
''')

sub(13, '''        if M_strategy == "optimality" and not (evaluate_energy(x, Q_obj, const_obj) <= E_f):''',
    '''        if is_optimality and not (evaluate_energy(x, Q_obj, const_obj) <= E_f):''')


# ── 3. circuit-building sweep: add the new folder ────────────────────────────────────
sub(112, '''          ("PO",       "optimality"):  ([6, 9, 12],  [1, 2, 3]),''',
    '''          ("PO",       "optimality"):  ([6, 9, 12],  [1, 2, 3]),
          ("PO", "optimality_maqaoa_percentile_interp"): ([6, 9, 12, 15], [1, 2, 3]),''')


# ── 4. the four driver cells: document the new choice ────────────────────────────────
OLD_ONE = '''problem_type = "PO"             # "NPP" | "PO" | "TSP_rand"
M_strategy   = "optimality"     # "feasibility" | "optimality"'''
NEW_ONE = '''problem_type = "PO"             # "NPP" | "PO" | "TSP_rand"
# "feasibility" | "optimality" | (PO only) "optimality_maqaoa_percentile_interp"
M_strategy   = "optimality_maqaoa_percentile_interp"'''
sub(115, OLD_ONE, NEW_ONE)

for idx in (116, 117, 119):
    s = cell_src(idx)
    assert s.count('M_strategy   = "optimality"') == 1, f"cell {idx}: M_strategy line not found"
    set_src(idx, s.replace(
        'M_strategy   = "optimality"',
        'M_strategy   = "optimality_maqaoa_percentile_interp"   # or "optimality" / "feasibility"'))

# cell 119 must now pass `layers` so the per-instance E_f is picked up
sub(119, '''        eta, shots, hits = eta_eff_from_counts(counts, problem_type, M_strategy,
                                               n_bits, vseed, directory=directory)''',
    '''        # layers + eta_required let eta_eff_from_counts recover the E_f this run was
        # trained against (per-instance percentile for the *_percentile* folders)
        eta, shots, hits = eta_eff_from_counts(counts, problem_type, M_strategy,
                                               n_bits, vseed, directory=directory,
                                               layers=layers, eta_required=eta_required)''')

json.dump(nb, open(NB, "w"), indent=1)
print("patched", NB, "(backup at", NB + ".bak_interp)")
