"""M_L1 vs M*: eta_eff AND the objective energy actually sampled.

Both arms are trained identically (same grid, vseeds, E_f = P75, INTERP warm start, seed 442,
500 steps); the single variable is the deployed penalty weight, M* vs M_L1(beta) with the same
beta = 1/temperature_unnormalized. The eta_eff story alone is two-sided but coarse -- eta_eff
saturates at 0.75 for any sampler that is merely a uniform projector onto the feasible set. The
objective energy is the finer probe: M_L1 > M* shrinks the objective's share of the normalized
cost Hamiltonian, so the circuit should be *blinder* to it and return worse energies even where
it is perfectly feasible.

Reported per (n_bits, layers), averaged over the 10 vseeds, from the EXACT output distribution:
  * P_feas                 -- weight on E_pen == 0
  * eta_eff                -- weight on {E_pen == 0 and E_obj <= E_f}
  * <E_obj>, median        -- unconditional, over every state the circuit returns
  * <E_obj | feasible>     -- restricted to the feasible set
  * gap_feas               -- the scale-free headline: (<E_obj|feas> - E_min_feas) /
                              (E_uniform_feas - E_min_feas). 0 = sits on the best feasible state,
                              1 = uniform over the feasible set, i.e. the objective had no
                              influence whatsoever. Continuous, so unlike a percentile rank it is
                              not quantized by the small feasible sets at n = 6, 9.
  * q_uncond / q_feas      -- percentile rank of the mean sampled energy in the spectrum, kept as a
                              cross-check (uniform over feasible -> q_feas = 50).

Writes misc_plots/PO_M_L1_vs_star_energy.{pdf,png} and prints the table.
"""

import glob
import pickle
import re
import sys

import numpy as np

import pennylane as qml

import qaoa_sampling_oneshot as qso

D_STAR = "data_qaoa/PO_optimality_maqaoa_percentile_interp"
D_L1 = "data_qaoa/PO_optimality_maqaoa_percentile_interp_L1"
DIRECTORY, ETA, W = "PO_small", 0.5, 3
BITS, PS = (6, 9, 12, 15), (1, 2, 3)


def ma_qaoa_probs(qubo_dict, num_qubits, gammas, betas):
    """Exact output distribution of the ma-QAOA ansatz at the given (gammas, betas).

    Mirrors the aggregation + ansatz of `run_ma_qaoa_with_solution` verbatim (steps 1-3 there);
    kept as a separate function so that re-evaluating a stored circuit never touches the training
    path. `validate_reconstruction` below checks it against the distributions the L1 arm stored.
    """
    single_z, double_zz = {}, {}
    for (i, j), val in qubo_dict.items():
        if i == j:
            single_z[i] = single_z.get(i, 0.0) + (-val / 2)
        else:
            pair = (min(i, j), max(i, j))
            double_zz[pair] = double_zz.get(pair, 0.0) + (val / 4)
            single_z[pair[0]] = single_z.get(pair[0], 0.0) + (-val / 4)
            single_z[pair[1]] = single_z.get(pair[1], 0.0) + (-val / 4)

    single_z = {i: c for i, c in single_z.items() if not np.isclose(c, 0.0)}
    double_zz = {pr: c for pr, c in double_zz.items() if not np.isclose(c, 0.0)}
    single_terms = sorted(single_z.items())
    double_terms = sorted(double_zz.items())
    n_single = len(single_terms)

    gammas, betas = np.asarray(gammas, float), np.asarray(betas, float)
    p = gammas.shape[0]

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def prob_node(gammas, betas):
        for w_ in range(num_qubits):
            qml.Hadamard(wires=w_)
        for layer in range(p):
            for k, (i, coeff) in enumerate(single_terms):
                qml.RZ(2 * gammas[layer, k] * coeff, wires=i)
            for k, ((i, j), coeff) in enumerate(double_terms):
                qml.IsingZZ(2 * gammas[layer, n_single + k] * coeff, wires=[i, j])
            for i in range(num_qubits):
                qml.RX(2 * betas[layer, i], wires=i)
        return qml.probs(wires=range(num_qubits))

    return np.array(prob_node(gammas, betas))


def probs_of_run(rec):
    """Exact distribution of a stored run, recomputed from (Q_norm, best_pars)."""
    Q_norm = np.asarray(rec["Q_norm"], float)
    n_bits = int(rec["bits"])
    Q_dict = {idx: val for idx, val in np.ndenumerate(Q_norm) if not np.isclose(val, 0)}
    gammas, betas = rec["best_pars"]
    return ma_qaoa_probs(Q_dict, n_bits, gammas, betas)


def spectra(n_bits, vseed):
    """(E_obj, E_pen) over all 2^n bitstrings, in raw (unpenalized) objective units."""
    N = n_bits // W
    info = (DIRECTORY, vseed)
    Q_obj, c_obj = qso.get_QUBO_PO(N, W, 1, "seed", info, penalization=False, objective=True)
    Q_pen, c_pen = qso.get_QUBO_PO(N, W, 1, "seed", info, penalization=True, objective=False)
    idx = np.arange(2 ** n_bits)
    X = ((idx[:, None] >> np.arange(n_bits - 1, -1, -1)) & 1).astype(float)
    E_obj = np.einsum("si,ij,sj->s", X, Q_obj, X) + c_obj
    E_pen = np.einsum("si,ij,sj->s", X, Q_pen, X) + c_pen
    return E_obj, E_pen


def objective_share(n_bits, vseed, M):
    """How much of the normalized cost Hamiltonian Q/max|Q| is objective, at penalty weight M.

    This is the quantity that explains why M* and M_L1 give the same circuit: the trainer never
    sees M's overall scale (the two-step scaling reduces to Q/max|Q|), only the objective's share,
    and both Ms drive that share many orders of magnitude below anything the cost layer can act on.

    entry : max|Q_obj| / max|Q|, the entrywise ratio.
    feas  : (max - min of E_obj over the FEASIBLE set) / max|Q|. Inside the feasible set the penalty
            is identically zero, so this is the entire energy spread the circuit could use to tell
            two feasible states apart -- i.e. roughly the phase in radians a gamma ~ 1 cost layer
            can impart between them. It is the decisive one for "can it optimize at all?".
    """
    N = n_bits // W
    info = (DIRECTORY, vseed)
    Q_obj, _ = qso.get_QUBO_PO(N, W, 1, "seed", info, penalization=False, objective=True)
    Q, _ = qso.get_QUBO_PO(N, W, M, "seed", info)
    E_obj, E_pen = spectra(n_bits, vseed)
    feas = np.isclose(E_pen, 0)
    scale = np.abs(Q).max()
    return (np.abs(Q_obj).max() / scale,
            (E_obj[feas].max() - E_obj[feas].min()) / scale)


def add_shares(df):
    """Attach the objective-share columns to a summary table (cheap: no circuit evaluation)."""
    out = [objective_share(int(r.bits), int(r.vseed), float(r.M_used)) for r in df.itertuples()]
    df = df.copy()
    df["share_entry"] = [o[0] for o in out]
    df["share_feas"] = [o[1] for o in out]
    return df


def percentile_rank(value, sample):
    """Fraction of `sample` strictly below `value`, in percent (the inverse of np.percentile)."""
    return 100.0 * np.mean(sample < value)


def collect(folder, label):
    rows = []
    for f in sorted(glob.glob(f"{folder}/*.txt")):
        rec = pickle.load(open(f, "rb"))
        n_bits, p, vseed = int(rec["bits"]), int(rec["layers"]), int(rec["vseed"])
        E_obj, E_pen = spectra(n_bits, vseed)
        feas = np.isclose(E_pen, 0)
        probs = probs_of_run(rec)

        P_feas = probs[feas].sum()
        eta = probs[feas & (E_obj <= rec["E_f"])].sum()
        mean_all = float(probs @ E_obj)
        w_feas = probs[feas] / P_feas if P_feas > 1e-12 else None
        mean_feas = float(w_feas @ E_obj[feas]) if w_feas is not None else np.nan
        med_all = weighted_quantile(E_obj, probs, 0.5)
        med_feas = weighted_quantile(E_obj[feas], w_feas, 0.5) if w_feas is not None else np.nan

        # Scale-free "blindness ratio": 0 = sits on the best feasible state, 1 = uniform over the
        # feasible set (i.e. the objective had no influence at all), >1 = actively anti-correlated.
        # Unlike a percentile rank it is continuous, so it survives the small feasible sets at n=6,9.
        E_min_f, E_mean_f = float(E_obj[feas].min()), float(E_obj[feas].mean())
        E_min_a, E_mean_a = float(E_obj.min()), float(E_obj.mean())
        gap_feas = (mean_feas - E_min_f) / (E_mean_f - E_min_f)
        gap_all = (mean_all - E_min_a) / (E_mean_a - E_min_a)

        rows.append(dict(
            arm=label, bits=n_bits, layers=p, vseed=vseed,
            M_used=float(rec.get("M_used", rec["M_star"])),
            M_star=float(rec["M_star"]),
            P_feas=float(P_feas), eta=float(eta), eta_stored=float(rec["eta_eff"]),
            mean_all=mean_all, mean_feas=mean_feas,
            med_all=med_all, med_feas=med_feas,
            gap_all=float(gap_all), gap_feas=float(gap_feas),
            q_uncond=percentile_rank(mean_all, E_obj),
            q_feas=percentile_rank(mean_feas, E_obj[feas]) if np.isfinite(mean_feas) else np.nan,
            E_min_feas=E_min_f, E_mean_feas=E_mean_f,
            E_min_all=E_min_a, E_mean_all=E_mean_a,
            n_feas=int(feas.sum()),
        ))
        print(f"  [{label}] n={n_bits:2d} p={p} v={vseed:4d}  P_feas={P_feas:.3f} "
              f"eta={eta:.3f} (stored {rec['eta_eff']:.3f})  <E>={mean_all:+.4f} "
              f"<E|f>={mean_feas:+.4f}  gap_feas={gap_feas:.3f}", flush=True)
    return rows


def load_run(folder, n_bits, p, vseed, eta=ETA):
    """Everything needed to draw one run's sampled-energy distribution.

    Returns a dict with the raw spectra (E_obj, E_pen), the feasibility mask, the exact output
    distribution, and the cutoff E_f -- all aligned on the same 2^n state ordering.
    """
    f = f"{folder}/test_PO_bits_{n_bits}_vseed_{vseed}_layers_{p}_etareq_{eta}.txt"
    rec = pickle.load(open(f, "rb"))
    E_obj, E_pen = spectra(n_bits, vseed)
    feas = np.isclose(E_pen, 0)
    return dict(rec=rec, E_obj=E_obj, E_pen=E_pen, feas=feas,
                probs=probs_of_run(rec), E_f=float(rec["E_f"]),
                M_used=float(rec.get("M_used", rec["M_star"])))


def weighted_quantile(values, weights, q):
    """q-th quantile (q in [0,1]) of `values` under the probability weights `weights`."""
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cdf = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(cdf, q)])


def validate_reconstruction(folder, n_checks=6):
    """The L1 arm stored `probs`; confirm the recomputation path reproduces them bit-for-bit."""
    files = sorted(glob.glob(f"{folder}/*.txt"))
    if not files:
        return
    worst = 0.0
    for f in files[:: max(1, len(files) // n_checks)]:
        rec = pickle.load(open(f, "rb"))
        if "probs" not in rec:
            continue
        d = np.abs(probs_of_run(rec) - np.asarray(rec["probs"], float)).max()
        worst = max(worst, d)
    print(f"reconstruction check on {folder}: max |p_recomputed - p_stored| = {worst:.3e}",
          flush=True)
    assert worst < 1e-10, "probability reconstruction does not match the stored distribution"


if __name__ == "__main__":
    import os
    import pandas as pd

    validate_reconstruction(D_L1)

    # re-evaluating the M* arm's 120 circuits is the slow part and its data never changes, so cache it
    CACHE = "data_qaoa/_rows_Mstar.pkl"
    if os.path.exists(CACHE):
        star = pickle.load(open(CACHE, "rb"))
        print(f"reusing cached {len(star)} M* rows from {CACHE}", flush=True)
    else:
        star = collect(D_STAR, "M*")
        pickle.dump(star, open(CACHE, "wb"))

    rows = star + collect(D_L1, "M_L1")
    df = pd.DataFrame(rows)
    df.to_pickle("data_qaoa/M_L1_vs_star_summary.pkl")
    print(f"\nwrote data_qaoa/M_L1_vs_star_summary.pkl  ({len(df)} runs)")

    piv = df.groupby(["bits", "layers", "arm"]).agg(
        P_feas=("P_feas", "mean"), eta=("eta", "mean"),
        E_all=("mean_all", "mean"), E_feas=("mean_feas", "mean"),
        med_feas=("med_feas", "mean"),
        gap_all=("gap_all", "mean"), gap_feas=("gap_feas", "mean"),
        q_feas=("q_feas", "mean")).round(3)
    print(piv.to_string())
