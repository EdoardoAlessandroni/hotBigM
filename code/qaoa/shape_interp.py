"""What SHAPE is the INTERP output distribution over the feasible subspace?

eff #states says how many states carry the mass, not how it is arranged. Three hypotheses:
  * near-deterministic  -> one state with p ~ 1
  * uniform over a subset -> the top-k states share mass equally, then a sharp cliff
  * thermal              -> p(x) ~ exp(-beta_eff * E_obj(x)) across the whole feasible set

Discriminated by: the top-8 feasible probabilities (uniform shows a plateau then a cliff), and
a weighted least-squares fit of log p vs E_obj restricted to the support, whose R^2 says how
Boltzmann-like the arrangement is. A uniform-over-subset distribution has beta_eff ~ 0 with
poor R^2; a thermal one has beta_eff > 0 with high R^2 and no cliff.
"""

import itertools
import pickle

import numpy as np

from analyse_spread_interp import ma_qaoa_probs, structure

D = "data_qaoa/PO_optimality_maqaoa_percentile_interp"
CASES = [(9, 3, 42), (12, 2, 42), (12, 3, 42), (15, 3, 42), (15, 3, 142), (12, 3, 442)]

for n_bits, p, vs in CASES:
    r = pickle.load(open(f"{D}/test_PO_bits_{n_bits}_vseed_{vs}_layers_{p}_etareq_0.5.txt", "rb"))
    feas, E_obj, E_feas = structure(n_bits, vs)
    Qn = np.asarray(r["Q_norm"], float)
    qd = {idx: v for idx, v in np.ndenumerate(Qn) if not np.isclose(v, 0)}
    g, b = r["best_pars"]
    probs = ma_qaoa_probs(qd, n_bits, p, g, b)

    pf = probs[feas] / probs[feas].sum()
    Ef_ = E_obj[feas]
    order = np.argsort(-pf)
    top = pf[order][:8]

    # support = states carrying 99.9% of the mass
    cum = np.cumsum(pf[order])
    k999 = int(np.searchsorted(cum, 0.999) + 1)
    sup = order[:k999]

    # Boltzmann fit on the support, weighted by probability
    x, y, wgt = Ef_[sup], np.log(pf[sup]), pf[sup]
    xm = np.average(x, weights=wgt); ym = np.average(y, weights=wgt)
    cov = np.average((x - xm) * (y - ym), weights=wgt)
    var = np.average((x - xm) ** 2, weights=wgt)
    slope = cov / var if var > 0 else np.nan
    pred = ym + slope * (x - xm)
    ss_res = np.average((y - pred) ** 2, weights=wgt)
    ss_tot = np.average((y - ym) ** 2, weights=wgt)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    nz = pf[pf > 0]
    eff = float(np.exp(-(nz * np.log(nz)).sum()))
    print(f"n={n_bits} p={p} vseed={vs}   |feas|={len(pf)}  eta={r['eta_eff']:.3f}  "
          f"eff#st={eff:.1f}  support(99.9%)={k999}")
    print(f"   top-8 probs : {'  '.join(f'{t:.4f}' for t in top)}")
    print(f"   ratio p1/p4 = {top[0]/top[3]:.2f}   p1/p8 = {top[0]/top[7]:.2f}   "
          f"(1.00 = flat plateau)")
    print(f"   Boltzmann fit on support: beta_eff = {-slope:+.3f}, R^2 = {r2:.3f}\n", flush=True)
