"""Execute the [M0-zero] notebook cells headlessly so bugs surface here, not in the notebook.

Mirrors the notebook's preamble (pandas/numpy/matplotlib/Line2D/display) and runs the tagged cells
in order in a single shared namespace, exactly as Jupyter would.
"""

import json
import sys
import traceback

import matplotlib
matplotlib.use("Agg")

TAG = "### [M0-zero]"

ns = {}
exec("import numpy as np\n"
     "import pandas as pd\n"
     "import matplotlib.pyplot as plt\n"
     "from matplotlib.lines import Line2D\n"
     "display = lambda *a: print(*a)\n", ns)

nb = json.load(open("qaoa_playground.ipynb"))
cells = [c for c in nb["cells"] if "".join(c["source"]).startswith(TAG)]
print(f"{len(cells)} tagged cells\n" + "=" * 78)

fails = 0
for i, c in enumerate(cells):
    src = "".join(c["source"])
    head = src.split("\n")[0]
    print(f"\n--- cell {i}: {head[len(TAG):].strip()[:70]}")
    try:
        exec(src, ns)
    except Exception:
        fails += 1
        traceback.print_exc()

print("=" * 78)
print(f"{len(cells) - fails}/{len(cells)} cells ran clean")
sys.exit(1 if fails else 0)
