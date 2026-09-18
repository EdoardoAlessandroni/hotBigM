"""Is the INTERP p=2/p=3 output actually spread out, or still a one-state 'Bernoulli sampler'?

eta_eff going from a 0/1 lottery to ~0.9 +- 0.12 is consistent with two very different things:
  (a) the circuit still collapses onto ~1 feasible basis state, but warm-starting makes it
      reliably pick a state BELOW E_f -- the Bernoulli coin is just biased to heads now, and
      the intermediate eta values come only from the small leakage off that one state;
  (b) the circuit genuinely spreads over many feasible states, most of them below E_f.
(a) would mean the nice plot is cosmetic; (b) would mean it is real.

Distinguished by two numbers per run, both computed from the stored best_pars:
  * eff. #states = exp(Shannon entropy of the feasible-conditioned distribution). 1 = delta,
    |feasible| = uniform over the feasible set.
  * mean quantile of the sampled feasible mass in the feasible objective spectrum. 0.5 =
    indifferent to the objective, < 0.5 = prefers good solutions.

The reconstruction is validated by recomputing eta_eff from the rebuilt distribution and
comparing it against the eta_eff stored in the file: if the Pauli-term ordering used here did
not match the one used at training time, the parameters would be applied to the wrong gates and
eta_eff would not match.
"""

import itertools
import pickle

import numpy as np
import pennylane as qml

from qaoa_sampling_oneshot import (evaluate_energy, get_PO_obj, get_QUBO_PO,
                                   state_inttobin)

D = "data_qaoa/PO_optimality_maqaoa_percentile_interp"
D_COLD = "data_qaoa/PO_optimality_maqaoa_percentile"
VSEEDS = list(range(42, 1000, 100))
w = 3


def ma_qaoa_probs(qubo_dict, num_qubits, p, gammas, betas):
    """Output distribution of the ma-QAOA ansatz -- term aggregation/ordering copied verbatim
    from run_ma_qaoa_with_solution so the stored parameters map onto the same gates."""
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
    single_terms, double_terms = sorted(single_z.items()), sorted(double_zz.items())
    n_single = len(single_terms)

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def node(g, b):
        for q in range(num_qubits):
            qml.Hadamard(wires=q)
        for layer in range(p):
            for k, (i, coeff) in enumerate(single_terms):
                qml.RZ(2 * g[layer, k] * coeff, wires=i)
            for k, ((i, j), coeff) in enumerate(double_terms):
                qml.IsingZZ(2 * g[layer, n_single + k] * coeff, wires=[i, j])
            for i in range(num_qubits):
                qml.RX(2 * b[layer, i], wires=i)
        return qml.probs(wires=range(num_qubits))

    return np.array(node(np.asarray(gammas, float), np.asarray(betas, float)))


# ---- per-size structural quantities, computed once ----
cache = {}
def structure(n_bits, vseed):
    key = (n_bits, vseed)
    if key in cache:
        return cache[key]
    N = n_bits // w
    Q_pen, c_pen = get_QUBO_PO(N, w, 1, "seed", ("PO_small", vseed), penalization=True, objective=False)
    Q_obj, c_obj = get_QUBO_PO(N, w, 1, "seed", ("PO_small", vseed), penalization=False, objective=True)
    b = 2**w - 1
    X = np.array([[int(x) for s in c for x in format(s, f"0{w}b")]
                  for c in itertools.product(range(b + 1), repeat=N) if sum(c) == b], float)
    Qo, Lo = get_PO_obj(n_bits, ("PO_small", vseed))
    E_feas = np.sort(np.einsum("ij,jk,ik->i", X, Qo + np.diag(Lo), X))
    feas = np.zeros(2**n_bits, dtype=bool)
    E_obj = np.zeros(2**n_bits)
    for i in range(2**n_bits):
        x = state_inttobin(i, n_bits)
        feas[i] = np.isclose(evaluate_energy(x, Q_pen, c_pen), 0)
        E_obj[i] = evaluate_energy(x, Q_obj, c_obj)
    cache[key] = (feas, E_obj, E_feas)
    return cache[key]


print("eff #states: 1 = single basis state (Bernoulli sampler), |feas| = uniform")
print("mean quant : 0.5 = indifferent to objective, < 0.5 = prefers low objective energy\n")
print(f"{'n':>3} {'p':>2} {'|feas|':>7} | {'COLD eff#st':>11} {'quant':>6} | "
      f"{'INTERP eff#st':>13} {'quant':>6} | {'eta chk':>8}")
print("-" * 78, flush=True)

for n_bits in (9, 12, 15):
    for p in (2, 3):
        row = {}
        for lab, folder in (("cold", D_COLD), ("interp", D)):
            effs, quants, errs = [], [], []
            for vs in VSEEDS:
                r = pickle.load(open(f"{folder}/test_PO_bits_{n_bits}_vseed_{vs}_layers_{p}_etareq_0.5.txt", "rb"))
                feas, E_obj, E_feas = structure(n_bits, vs)
                Qn = np.asarray(r["Q_norm"], float)
                qd = {idx: v for idx, v in np.ndenumerate(Qn) if not np.isclose(v, 0)}
                g, b_ = r["best_pars"]
                probs = ma_qaoa_probs(qd, n_bits, p, g, b_)

                pfe = probs[feas].sum()
                pf = probs[feas] / pfe
                nz = pf[pf > 0]
                effs.append(float(np.exp(-(nz * np.log(nz)).sum())))
                q = np.searchsorted(E_feas, E_obj[feas], side="right") / len(E_feas)
                quants.append(float((pf * q).sum()))
                eta_rebuilt = probs[feas & (E_obj <= r["E_f"])].sum()
                errs.append(abs(eta_rebuilt - r["eta_eff"]))
            row[lab] = (np.mean(effs), np.mean(quants), max(errs), len(E_feas))
        (ec, qc, _, nf), (ei, qi, err, _) = row["cold"], row["interp"]
        print(f"{n_bits:>3} {p:>2} {nf:>7} | {ec:>11.1f} {qc:>6.3f} | {ei:>13.1f} {qi:>6.3f} | "
              f"{err:>8.1e}", flush=True)

print("\n'eta chk' = max |eta_eff rebuilt from best_pars - eta_eff stored in file| over the "
      "10 instances;\nsmall values confirm the reconstruction reproduces the trained circuit.")
