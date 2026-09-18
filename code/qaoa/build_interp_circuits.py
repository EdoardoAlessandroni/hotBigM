"""Headless run of the notebook's 'store optimized circuits' step for the INTERP sweep.

Executes the notebook's own helper cells (imports, pennylane->pytket translation + H2
compilation, build_and_save_circuit, data_subdir_for) rather than reimplementing them, so the
circuits written here are byte-for-byte what cell 112 would produce. Nothing is submitted and
no credits are spent -- this is a local rebase/compile to the H2 native gateset.

Writes ./optimized_circuits/PO_optimality_maqaoa_percentile_interp_bits_{n}_vseed_{v}
_etareq_0.5.pkl -- one file per instance, a dict keyed by depth, so 40 files / 120 circuits.
"""

import glob
import json
import re
import sys
from timeit import default_timer as timer

NB = "qaoa_playground.ipynb"
HELPER_CELLS = [0, 9, 11, 13]        # imports | translate+compile | build | deploy helpers

nb = json.load(open(NB))
g = {"__name__": "__main__"}
for i in HELPER_CELLS:
    src = "".join(nb["cells"][i]["source"])
    exec(compile(src, f"<nb cell {i}>", "exec"), g)

build_and_save_circuit = g["build_and_save_circuit"]
data_subdir_for = g["data_subdir_for"]

PROBLEM, STRATEGY, ETA = "PO", "optimality_maqaoa_percentile_interp", 0.5
BITS, LAYERS = [6, 9, 12, 15], [1, 2, 3]
subdir = data_subdir_for(PROBLEM, STRATEGY)
assert subdir == "PO_optimality_maqaoa_percentile_interp", f"unexpected subdir {subdir!r}"
print(f"subdir = {subdir}\n", flush=True)

t0 = timer()
done, failed = 0, []
for n_bits in BITS:
    for layers in LAYERS:
        pattern = (f"./data_qaoa/{subdir}/test_{PROBLEM}_bits_{n_bits}"
                   f"_vseed_*_layers_{layers}_etareq_{ETA}.txt")
        vseeds = sorted(int(re.search(r"_vseed_(\d+)_", f).group(1)) for f in glob.glob(pattern))
        if not vseeds:
            print(f"!! no data for bits={n_bits} layers={layers}", flush=True)
            continue
        t1 = timer()
        for vseed in vseeds:
            try:
                e = build_and_save_circuit(PROBLEM, STRATEGY, n_bits, layers, vseed,
                                           eta_required=ETA)
                done += 1
            except Exception as exc:
                failed.append((n_bits, layers, vseed, repr(exc)[:120]))
                print(f"   FAIL n={n_bits} p={layers} v={vseed}: {exc!r:.120}", flush=True)
        print(f"bits={n_bits:2d} layers={layers}  -> {len(vseeds):2d} instances, "
              f"{timer()-t1:6.1f}s   [{done}/{len(BITS)*len(LAYERS)*10}]", flush=True)

print(f"\n{done} circuits stored, {len(failed)} failures, {(timer()-t0)/60:.1f} min", flush=True)
for f in failed:
    print("   ", f, flush=True)
