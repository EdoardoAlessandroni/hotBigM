"""Make Ef_for_run fail loudly instead of falling back when a percentile folder lacks E_f.

A *_percentile* subdir is by definition one whose runs chose E_f per instance and recorded it.
If the stored value is missing there, silently reverting to select_Ef's 1e-1*n_bits**2 would
score the noisy shots against a cutoff that sits ABOVE the whole feasible spectrum (14.4 vs
~0.13 at n=12), turning the optimality criterion into a feasibility one and inflating eta --
with no visible sign in the output. Better to stop.
"""

import json
import shutil

NB = "qaoa_playground.ipynb"
shutil.copy2(NB, NB + ".bak_efstrict")
nb = json.load(open(NB))

src = "".join(nb["cells"][13]["source"])

OLD = '''    if layers is not None:
        f = (Path(dir_data) / data_subdir_for(problem_type, M_strategy) /
             f"test_{problem_type}_bits_{n_bits}_vseed_{vseed}"
             f"_layers_{layers}_etareq_{eta_required}.txt")
        if f.exists():
            with open(f, "rb") as fh:
                stored = pickle.load(fh)
            if "E_f" in stored:
                return float(stored["E_f"]), "file"
    return select_Ef(problem_type, N_idx_from_bits(problem_type, n_bits)), "select_Ef"'''

NEW = '''    subdir = data_subdir_for(problem_type, M_strategy)
    # a *_percentile* folder chose E_f per instance and recorded it; falling back to
    # select_Ef there would score against a cutoff above the entire feasible spectrum
    # (14.4 vs ~0.13 at n_bits=12) and silently turn optimality into feasibility
    expects_stored = "percentile" in subdir

    f = None
    if layers is not None:
        f = (Path(dir_data) / subdir /
             f"test_{problem_type}_bits_{n_bits}_vseed_{vseed}"
             f"_layers_{layers}_etareq_{eta_required}.txt")
        if f.exists():
            with open(f, "rb") as fh:
                stored = pickle.load(fh)
            if "E_f" in stored:
                return float(stored["E_f"]), "file"
            if expects_stored:
                raise KeyError(
                    f"{f.name} lives in {subdir!r} (a percentile folder) but stores no 'E_f'. "
                    f"Refusing to fall back to select_Ef = {select_Ef(problem_type, N_idx_from_bits(problem_type, n_bits)):.5g}, "
                    f"which would score these shots against a cutoff above the whole feasible "
                    f"spectrum. Re-run the training for this cell, or pass E_f=... explicitly.")

    if expects_stored:
        where = f"{f}" if f is not None else f"{subdir} (no `layers` given, so no file was looked up)"
        raise FileNotFoundError(
            f"{subdir!r} is a percentile folder, so E_f must come from the training file, "
            f"but it could not be read: {where}. "
            f"Pass layers=... (and the matching eta_required) so the file can be found, or "
            f"pass E_f=... explicitly.")

    return select_Ef(problem_type, N_idx_from_bits(problem_type, n_bits)), "select_Ef"'''

assert src.count(OLD) == 1, f"anchor found {src.count(OLD)} times"
nb["cells"][13]["source"] = src.replace(OLD, NEW).splitlines(keepends=True)

json.dump(nb, open(NB, "w"), indent=1)
print("patched", NB)
