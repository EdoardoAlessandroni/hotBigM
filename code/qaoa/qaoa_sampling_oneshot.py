"""One-shot span-proxy implementation of `qaoa_sampling` (see spec_approachA_oneshot.md).

Self-contained module (mirrors the QUBO-building/QAOA-running code of
`qaoa_playground.ipynb`, following the pattern already used by
`qaoa_simulations.py`) so it can be imported from the notebook and run
side-by-side with the original `qaoa_sampling`:

    from qaoa_sampling_oneshot import qaoa_sampling_oneshot
    results = qaoa_sampling_oneshot("PO", N_idx, info_instance, "feasibility", 0.8, layers)

Replaces the `select_normalization` / `norm_guess` heuristic with a spectral-span
proxy `s_conv ~= E_max^full - E_min^full` of the full QUBO, evaluated once at a
reference penalty weight `M0` (one-shot, as opposed to the self-consistent
root-finder of Approach B).
"""

import numpy as np
import random
from timeit import default_timer as timer

import cvxpy as cp
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model_reader import ModelReader

import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as pl_np

from myalgo import M_method_feas, M_method_opt


# Problem-size grids, mirrored from qaoa_playground.ipynb (kept fixed there
# across the notebook, and relied on by `select_Ef`) -- update here if the
# notebook's values ever change.
Ns = np.arange(3, 9)              # for NPP
P = 3                              # for NPP
N_city_circle = np.arange(2, 6)    # for TSP_circle
N_city_rand = np.arange(2, 6)      # for TSP_rand
N_stocks = np.arange(2, 9)         # for PO
w = 3                              # for PO


######### General auxiliary functions


def symmetrize(Q):
    return (Q + Q.T) / 2

def L1_norm(Q, const):
    return const + np.sum(np.abs(Q))

def L1_norm_hot(Q, const, n_bits, temperature, min_pfeas):
    return L1_norm(Q, const) + temperature * (n_bits * np.log(2) - np.log(1 - min_pfeas))

def evaluate_energy(solution, Q, const):
    return const + np.dot(solution, np.dot(Q, solution))

def state_inttobin(i, n):
    state_string = bin(i)[2:].zfill(n)
    return [int(b) for b in state_string]

def temperature_from_layers(p, N):
    return (2.8 * p / (N**(3/2)))**-1

def select_Ef(problem_type, N_idx):
    """ Choose E_f prop. to the square of nbits, since the number of terms in the Hamiltonian increase like this.
    The constant is chosen such that it's possible to sample even at eta_req = 0.75 at temperatures = DA_temp * 100 """
    if problem_type == "NPP":
        n_bits = Ns[N_idx] * P
        return 2e4 * n_bits**2 ### NOTE: before QAOA, it was 3e3
    elif problem_type == "TSP_circle":
        n_bits = N_city_circle[N_idx]**2
        return 3e5 * n_bits**2
    elif problem_type == "TSP_rand":
        n_bits = N_city_rand[N_idx]**2
        return 3e5 * n_bits**2
    elif problem_type == "PO":
        n_bits = N_stocks[N_idx] * w
        return 1e-1 * n_bits**2  ### NOTE: before QAOA, it was 3e-2. 1e-1 only for vseed=842


# NPP specifics

def get_QUBO_NPP(N, P, M, seed, penalization=True, objective=True):
    """build a NPP QUBO with N integers, P partitions, M penalty factor, for problem identified with seed"""
    Q, const = np.zeros((N * P, N * P)), 0
    if objective:
        numbs = build_numbs_set(N, P, seed)
        Ho, const_o = build_obj_NPP(numbs, N, P)
        Q += Ho
        const += const_o
    if penalization:
        Hp, const_p = build_pen_NPP(N, P)
        Q += M * Hp
        const += M * const_p
    return Q, const

def build_numbs_set(N, P, seed):
    random.seed(int(seed))  # stdlib random.seed rejects np.int64; cast to plain int
    numbs = np.array([int(1000 * random.uniform(0.0, 1.0)) for _ in range(N)])
    total = sum(numbs)
    if numbs[-1] - (total % P) > 1:
        numbs[-1] -= total % P
    else:
        numbs[-1] += P - (total % P)
    total = sum(numbs)
    return numbs

def idx_onehot(i, p, P):
    return i * P + p

def build_obj_NPP(numbs, N, P):
    H = np.zeros((N * P, N * P))
    alpha = np.sum(numbs) / P
    for p in range(P):
        for i in range(N):
            H[idx_onehot(i, p, P), idx_onehot(i, p, P)] = numbs[i]**2 - 2*alpha*numbs[i]
            for i_prime in range(N):
                if i_prime == i:
                    continue
                H[idx_onehot(i, p, P), idx_onehot(i_prime, p, P)] = numbs[i] * numbs[i_prime]
    const = P * alpha**2
    H = symmetrize(H)
    return H, const

def build_pen_NPP(N, P):
    H = np.zeros((N * P, N * P))
    for p in range(P):
        for i in range(N):
            H[idx_onehot(i, p, P), idx_onehot(i, p, P)] = -1
            for p_prime in range(P):
                if p_prime == p:
                    continue
                H[idx_onehot(i, p, P), idx_onehot(i, p_prime, P)] = 1
    const = N
    H = symmetrize(H)
    return H, const


# TSP specifics

def build_adjacency(Nc, info_instance, info_type):
    if info_type == "circle":
        coordinates = [(1_000_000 * np.cos((index / Nc) * 2 * np.pi), 1_000_000 * np.sin((index / Nc) * 2 * np.pi)) for index in range(Nc)]
        distance_table = [[int(np.sqrt((coordinates[c_1][0] - coordinates[c_0][0]) ** 2 + (coordinates[c_1][1] - coordinates[c_0][1]) ** 2)) for c_0 in range(Nc)] for c_1 in range(Nc)]
    elif info_type == "seed":
        seed = info_instance
        np.random.seed(seed)
        coordinates = 1e6 * np.array([np.random.uniform(low=-1, high=1, size=2) for i in range(Nc)])
        distance_table = [[int(np.sqrt((coordinates[c_1][0] - coordinates[c_0][0]) ** 2 + (coordinates[c_1][1] - coordinates[c_0][1]) ** 2)) for c_0 in range(Nc)] for c_1 in range(Nc)]
    elif info_type == "adjacency":
        distance_table = info_instance
    return np.array(distance_table)

def build_obj_TSP(distances, Nc):
    H = np.zeros((Nc**2, Nc**2))
    for t in range(Nc):
        for i in range(Nc):
            for i_prime in range(Nc):
                if i_prime != i:
                    H[idx_onehot(t, i, Nc), idx_onehot((t+1) % Nc, i_prime, Nc)] = distances[i, i_prime]
    H = symmetrize(H)
    const = 0
    return H, const

def build_pen_TSP(Nc):
    H = np.zeros((Nc**2, Nc**2))
    for t in range(Nc):
        for i in range(Nc):
            H[idx_onehot(t, i, Nc), idx_onehot(t, i, Nc)] = -2
            for i_prime in range(Nc):
                if i_prime != i:
                    H[idx_onehot(t, i, Nc), idx_onehot(t, i_prime, Nc)] += 1
            for t_prime in range(Nc):
                if t_prime != t:
                    H[idx_onehot(t, i, Nc), idx_onehot(t_prime, i, Nc)] += 1
    const = 2*Nc
    H = symmetrize(H)
    return H, const

def get_QUBO_TSP(Nc, M, info_instance, info_type, objective=True, penalization=True):
    Q, const = np.zeros((Nc**2, Nc**2)), 0
    if objective:
        distances = build_adjacency(Nc, info_instance, info_type)
        Ho, const_o = build_obj_TSP(distances, Nc)
        Q += Ho
        const += const_o
    if penalization:
        Hp, const_p = build_pen_TSP(Nc)
        Q += M*Hp
        const += M*const_p
    return Q, const


# PO specifics

def lower_bound_obj_PO(N, w, info_instance):
    n_bits = N*w
    directory, vseed = info_instance
    filename = f"../../data/{directory}/{n_bits}/random{vseed}_{n_bits}.lp"
    m = ModelReader.read(filename, ignore_names=True)
    qp = from_docplex_mp(m)
    n = qp.get_num_binary_vars()
    obj = qp.objective
    Q = obj.quadratic.to_array()
    L = obj.linear.to_array()

    Q_tilde = np.ndarray((n+1, n+1))
    Q_tilde[1:, 1:] = Q
    Q_tilde[0, 0] = 0
    Q_tilde[0, 1:] = .5*L.T
    Q_tilde[1:, 0] = .5*L
    X = cp.Variable((n+1, n+1), symmetric=True)
    constraints = [X >> 0]
    constraints += [X[i, i] == X[0, i] for i in range(1, n+1)]
    constraints += [X[0, 0] == 1]
    prob = cp.Problem(cp.Minimize(cp.trace(Q_tilde @ X)), constraints)
    prob.solve(solver="MOSEK")
    return prob.value

def build_pen_PO(N, w):
    A = np.array([2**i for i in range(w-1, -1, -1)]*N)
    b = 2**w - 1
    H = np.outer(A, A)
    H -= 2*b*np.diag(A)
    const = b**2
    H = symmetrize(H)
    return H, const

def get_PO_obj(n_bits, info_instance):
    directory, vseed = info_instance
    filename = f"../../data/{directory}/{n_bits}/random{vseed}_{n_bits}.lp"
    m = ModelReader.read(filename, ignore_names=True)
    qp = from_docplex_mp(m)
    obj = qp.objective
    Q = obj.quadratic.to_array()
    L = obj.linear.to_array()
    Q = symmetrize(Q)
    return Q, L

def get_QUBO_PO(N, w, M, info_type, info_instance, objective=True, penalization=True):
    Q, const = np.zeros(((N*w), (N*w))), 0
    if objective:
        if info_type == "seed":
            Q_o, L_o = get_PO_obj(N*w, info_instance)
            Q += Q_o + np.diag(L_o)
        else:
            raise ValueError(f"info_type = {info_type} not understood for PO instance")
    if penalization:
        Hp, const_p = build_pen_PO(N, w)
        Q += M*Hp
        const += M*const_p
    return Q, const


######### QAOA runner (copied verbatim from qaoa_playground.ipynb, self-contained)


def run_qaoa_with_solution(
    qubo_dict: dict,
    num_qubits: int,
    p: int = 2,
    init_params=None,
    steps: int = 150,
    stepsize: float = 0.05,
    n_samples: int = 1000,
    mixer_weights=None,
    seed: int | None = 42,
    verbose: bool = True,
    draw_samples: bool = True,
    tolerance_opt: float = 1e-4,
) -> dict:
    """
    Run QAOA on a QUBO problem and return samples from the optimised circuit.

    The QUBO is:  minimise  x^T Q x,   x in {0,1}^n
    Encoded via the standard QUBO->Ising substitution  x_i = (1 - Z_i)/2.
    """

    coeffs, obs = [], []
    for (i, j), val in qubo_dict.items():
        if i == j:
            coeffs.append(-val / 2)
            obs.append(qml.PauliZ(i))
        else:
            pair = (min(i, j), max(i, j))
            coeffs.append(val / 4)
            obs.append(qml.PauliZ(pair[0]) @ qml.PauliZ(pair[1]))
            coeffs.append(-val / 4)
            obs.append(qml.PauliZ(pair[0]))
            coeffs.append(-val / 4)
            obs.append(qml.PauliZ(pair[1]))
    cost_h = qml.simplify(qml.dot(coeffs, obs))

    if mixer_weights is None:
        mixer_h = qaoa.x_mixer(range(num_qubits))
    else:
        mw = np.asarray(mixer_weights, dtype=float)
        if mw.shape != (num_qubits,):
            raise ValueError(
                f"mixer_weights must have length num_qubits={num_qubits}, "
                f"got shape {mw.shape}"
            )
        if np.any(mw <= 0):
            raise ValueError("mixer_weights must be strictly positive")
        mw_norm = mw / np.mean(mw)
        mixer_h = qml.simplify(
            qml.dot(mw_norm.tolist(), [qml.PauliX(i) for i in range(num_qubits)])
        )
        if verbose:
            print(f"  Weighted mixer: raw w = {mw.tolist()}")
            print(f"                  normalised w = {np.round(mw_norm, 4).tolist()}")

    dev_sv = qml.device("default.qubit", wires=num_qubits)

    def qaoa_ansatz(gammas, betas):
        for wire in range(num_qubits):
            qml.Hadamard(wires=wire)
        for layer in range(p):
            qaoa.cost_layer(gammas[layer], cost_h)
            qaoa.mixer_layer(betas[layer], mixer_h)

    @qml.qnode(dev_sv)
    def cost_node(params):
        qaoa_ansatz(params[0], params[1])
        return qml.expval(cost_h)

    @qml.qnode(dev_sv)
    def prob_node(params):
        qaoa_ansatz(params[0], params[1])
        return qml.probs(wires=range(num_qubits))

    @qml.set_shots(shots=n_samples)
    @qml.qnode(dev_sv)
    def sample_node(params):
        qaoa_ansatz(params[0], params[1])
        return qml.counts(wires=range(num_qubits))

    if init_params is not None:
        init_params = np.asarray(init_params, dtype=float)
        if init_params.shape != (2, p):
            raise ValueError(
                f"init_params shape {init_params.shape} doesn't match (2, p) = (2, {p})"
            )
        params = pl_np.array(init_params.copy(), requires_grad=True)
    else:
        rng = np.random.default_rng(seed)
        noise = rng.uniform(-0.05, 0.05, size=(2, p))
        init_gammas = np.linspace(0.01, 1, p) + noise[0]
        init_betas = np.linspace(1, 0.01, p) + noise[1]
        params = pl_np.array([init_gammas, init_betas], requires_grad=True)

    optimizer = qml.AdamOptimizer(stepsize=stepsize)
    cost_history = []
    best_params = None
    best_cost = np.inf

    epsilon = tolerance_opt
    step_check = 10
    for step in range(steps):
        params, cost_val = optimizer.step_and_cost(cost_node, params)
        if cost_val < best_cost:
            best_cost = cost_val
            best_params = params.copy()

        if (step + 1) % step_check == 0:
            cost_history.append(float(cost_val))
            if verbose:
                print(f"  step {step + 1:>4d}/{steps}  |  cost = {cost_val:+.6f}")

            if len(cost_history) > 1:
                diff_10 = np.abs(cost_history[-2] - cost_history[-1])
                if diff_10 < epsilon:
                    print(f"...breaking at iteration {step} out of {steps}")
                    break

    probs = np.array(prob_node(best_params))

    params = np.array(best_params)
    if draw_samples:
        counts = sample_node(params)
        return params, counts, probs, float(best_cost)
    else:
        return params, probs, float(best_cost)




def run_ma_qaoa_with_solution(
    qubo_dict: dict,
    num_qubits: int,
    p: int = 2,
    init_params=None,
    steps: int = 150,
    stepsize: float = 0.05,
    n_samples: int = 1000,
    seed: int | None = 42,
    verbose: bool = True,
    draw_samples: bool = True,
    tolerance_opt: float = 1e-4,
    patience: int = 3,                # NEW: require N consecutive small diffs to stop
):
    """
    Multi-angle QAOA (ma-QAOA), Herrman et al. 2022.

    Each unique Pauli term in H_C gets its own γ per layer.
    Each qubit gets its own β per layer.
    """

    # ── 1. Aggregate QUBO into unique Pauli terms ───────────────────────────
    single_z, double_zz = {}, {}
    for (i, j), val in qubo_dict.items():
        if i == j:
            single_z[i] = single_z.get(i, 0.0) + (-val / 2)
        else:
            pair = (min(i, j), max(i, j))
            double_zz[pair]   = double_zz.get(pair, 0.0) + (val / 4)
            single_z[pair[0]] = single_z.get(pair[0], 0.0) + (-val / 4)
            single_z[pair[1]] = single_z.get(pair[1], 0.0) + (-val / 4)

    single_z  = {i: c for i, c in single_z.items()  if not np.isclose(c, 0.0)}
    double_zz = {pr: c for pr, c in double_zz.items() if not np.isclose(c, 0.0)}
    single_terms = sorted(single_z.items())
    double_terms = sorted(double_zz.items())
    n_single = len(single_terms)
    n_double = len(double_terms)
    n_gammas = n_single + n_double
    n_betas  = num_qubits

    if verbose:
        print(f"  ma-QAOA: n_gammas = {n_gammas} ({n_single} Z + {n_double} ZZ), "
              f"n_betas = {n_betas}, total params = {p * (n_gammas + n_betas)}")

    coeffs_all, obs_all = [], []
    for i, c in single_terms:
        coeffs_all.append(c);  obs_all.append(qml.PauliZ(i))
    for (i, j), c in double_terms:
        coeffs_all.append(c);  obs_all.append(qml.PauliZ(i) @ qml.PauliZ(j))
    cost_h = qml.simplify(qml.dot(coeffs_all, obs_all))

    # ── 2. Device & ansatz ──────────────────────────────────────────────────
    dev_sv = qml.device("default.qubit", wires=num_qubits)

    def ma_qaoa_ansatz(gammas, betas):
        for w_ in range(num_qubits):
            qml.Hadamard(wires=w_)
        for layer in range(p):
            for k, (i, coeff) in enumerate(single_terms):
                qml.RZ(2 * gammas[layer, k] * coeff, wires=i)
            for k, ((i, j), coeff) in enumerate(double_terms):
                qml.IsingZZ(2 * gammas[layer, n_single + k] * coeff, wires=[i, j])
            for i in range(num_qubits):
                qml.RX(2 * betas[layer, i], wires=i)

    # ── 3. QNodes — γ and β as SEPARATE positional args (key fix) ───────────
    @qml.qnode(dev_sv)
    def cost_node(gammas, betas):
        ma_qaoa_ansatz(gammas, betas)
        return qml.expval(cost_h)

    @qml.qnode(dev_sv)
    def prob_node(gammas, betas):
        ma_qaoa_ansatz(gammas, betas)
        return qml.probs(wires=range(num_qubits))

    @qml.set_shots(shots=n_samples)
    @qml.qnode(dev_sv)
    def sample_node(gammas, betas):
        ma_qaoa_ansatz(gammas, betas)
        return qml.counts(wires=range(num_qubits))

    # ── 4. Init ─────────────────────────────────────────────────────────────
    if init_params is not None:
        gammas_init, betas_init = init_params
        gammas_init = np.asarray(gammas_init, dtype=float)
        betas_init  = np.asarray(betas_init,  dtype=float)
        if gammas_init.shape != (p, n_gammas):
            raise ValueError(f"init gammas shape {gammas_init.shape} != ({p},{n_gammas})")
        if betas_init.shape != (p, n_betas):
            raise ValueError(f"init betas shape {betas_init.shape} != ({p},{n_betas})")
    else:
        rng = np.random.default_rng(seed)
        ramp_gamma = np.linspace(0.01, 1.0, p)
        ramp_beta  = np.linspace(1.0, 0.01, p)
        gammas_init = np.broadcast_to(ramp_gamma[:, None], (p, n_gammas)).copy()
        betas_init  = np.broadcast_to(ramp_beta[:, None],  (p, n_betas )).copy()
        gammas_init += rng.uniform(-0.05, 0.05, size=gammas_init.shape)
        betas_init  += rng.uniform(-0.05, 0.05, size=betas_init.shape)

    # Two SEPARATE trainable tensors (NOT a list)
    gammas = pl_np.array(gammas_init, requires_grad=True)
    betas  = pl_np.array(betas_init,  requires_grad=True)

    # ── 4b. Sanity check: confirm gradients are non-zero at init ────────────
    if verbose:
        grad_fn = qml.grad(cost_node, argnum=[0, 1])
        g_gamma, g_beta = grad_fn(gammas, betas)
        print(f"  init grad norms: ||∂C/∂γ|| = {np.linalg.norm(g_gamma):.4e}, "
              f"||∂C/∂β|| = {np.linalg.norm(g_beta):.4e}")

    # ── 5. Optimisation ─────────────────────────────────────────────────────
    optimizer = qml.AdamOptimizer(stepsize=stepsize)
    cost_history = []
    best_gammas, best_betas, best_cost = None, None, np.inf
    epsilon, step_check = tolerance_opt, 10

    for step in range(steps):
        (gammas, betas), cost_val = optimizer.step_and_cost(cost_node, gammas, betas)
        if cost_val < best_cost:
            best_cost   = cost_val
            best_gammas = gammas.copy()
            best_betas  = betas.copy()

        if (step + 1) % step_check == 0:
            cost_history.append(float(cost_val))
            if verbose:
                print(f"  step {step+1:>4d}/{steps}  |  cost = {cost_val:+.6f}")
            if len(cost_history) > 1:
                diff_10 = np.abs(cost_history[-2] - cost_history[-1])
                if diff_10 < epsilon:
                    print(f"...breaking at iteration {step} out of {steps}")
                    break

    # ── 6. Outputs ──────────────────────────────────────────────────────────
    probs = np.array(prob_node(best_gammas, best_betas))
    final_gammas = np.array(best_gammas)
    final_betas  = np.array(best_betas)
    if draw_samples:
        counts = sample_node(final_gammas, final_betas)
        return (final_gammas, final_betas), counts, probs, float(best_cost)
    else:
        return (final_gammas, final_betas), probs, float(best_cost)


######### Span-proxy helpers (spec_approachA_oneshot.md)


def sdp_bound(Q, const, sense="min"):
    """SDP relaxation bound on min_x/max_x (x^T Q x + const), x in {0,1}^n.

    Generalizes `lower_bound_obj_PO` to an arbitrary (possibly asymmetric) QUBO
    matrix `Q` and scalar `const`, working directly from the in-memory matrix.
    """
    if sense not in ("min", "max"):
        raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")

    n = Q.shape[0]
    Qm = symmetrize(Q)
    L = np.diag(Qm).copy()
    Qq = Qm - np.diag(L)

    Q_tilde = np.zeros((n+1, n+1))
    Q_tilde[1:, 1:] = Qq
    Q_tilde[0, 1:] = 0.5*L
    Q_tilde[1:, 0] = 0.5*L
    Q_tilde[0, 0] = 0

    X = cp.Variable((n+1, n+1), symmetric=True)
    constraints = [X >> 0]
    constraints += [X[i, i] == X[0, i] for i in range(1, n+1)]
    constraints += [X[0, 0] == 1]

    if sense == "min":
        prob = cp.Problem(cp.Minimize(cp.trace(Q_tilde @ X)), constraints)
    else:
        prob = cp.Problem(cp.Maximize(cp.trace(Q_tilde @ X)), constraints)
    prob.solve(solver="MOSEK")
    return prob.value + const


def l1_upper_bound(Q, const):
    """Cheap l1-style upper bound on max_x (x^T Q x + const): sum of positive
    entries of the symmetrized Q, plus max(const, 0)."""
    Qm = symmetrize(Q)
    return np.sum(Qm[Qm > 0]) + max(const, 0)


def span_proxy(Q_obj, const_obj, Q_pen, const_pen, M0, problem_type, E_LB_obj, use_sdp_pen=True):
    """s_conv = (UB_obj - E_LB_obj) + M0 * UB_pen, an estimate of the full-QUBO
    span Delta_E = E_max^full - E_min^full at penalty weight M0.

    Returns (s_conv, UB_obj, UB_pen, ub_pen_method).
    """
    UB_obj = sdp_bound(Q_obj, const_obj, sense="max")
    if use_sdp_pen:
        UB_pen = sdp_bound(Q_pen, const_pen, sense="max")
        ub_pen_method = "SDP"
    else:
        UB_pen = l1_upper_bound(Q_pen, const_pen)
        ub_pen_method = "L1"
    s_conv = (UB_obj - E_LB_obj) + M0 * UB_pen
    return s_conv, UB_obj, UB_pen, ub_pen_method


######### Main function


def qaoa_sampling_oneshot(problem_type, N_idx, info_instance, Mstrategy, eta_required,
                           layers, qaoa_samples=1000, qaoa_steps=100,
                           M0_ref="L1", use_sdp_pen=True,
                           print_time=True, verbose_qaoa=False, verbose=False):
    """ One-shot span-proxy variant of `qaoa_sampling` (see spec_approachA_oneshot.md).
    Replaces the select_normalization/norm_guess heuristic with a spectral-span
    proxy s_conv of the full (objective + M*penalty) QUBO, evaluated once at a
    reference penalty weight M0. Same input/output contract as `qaoa_sampling`,
    plus the span-proxy diagnostics (s_conv, M0, UB_obj, UB_pen) in `results`. """
    t1 = timer()

    ### 1. Get LCBO
    if problem_type == "NPP":
        N = Ns[N_idx]
        size = (N, P)
        n_bits = N * P
        Q_pen, const_pen = get_QUBO_NPP(N, P, 1, info_instance, penalization=True, objective=False)
        Q_obj, const_obj = get_QUBO_NPP(N, P, 1, info_instance, penalization=False, objective=True)
        info_type = "seed"
        problem_type_Mfunc = "NPP"
    elif problem_type == "TSP_circle":
        Nc = N_city_circle[N_idx]
        size = (Nc)
        n_bits = Nc**2
        info_instance = None
        info_type = "circle"
        Q_pen, const_pen = get_QUBO_TSP(Nc, 1, None, None, penalization=True, objective=False)
        Q_obj, const_obj = get_QUBO_TSP(Nc, 1, info_instance, info_type, penalization=False, objective=True)
        problem_type_Mfunc = "TSP"
    elif problem_type == "TSP_rand":
        Nc = N_city_rand[N_idx]
        size = (Nc)
        n_bits = Nc**2
        info_type = "seed"
        Q_pen, const_pen = get_QUBO_TSP(Nc, 1, None, None, penalization=True, objective=False)
        Q_obj, const_obj = get_QUBO_TSP(Nc, 1, info_instance, info_type, penalization=False, objective=True)
        problem_type_Mfunc = "TSP"
    elif problem_type == "PO":
        N = N_stocks[N_idx]
        size = (N, w)
        n_bits = N*w
        info_type = "seed"
        Q_pen, const_pen = get_QUBO_PO(N, w, 1, info_type, info_instance, penalization=True, objective=False)
        Q_obj, const_obj = get_QUBO_PO(N, w, 1, info_type, info_instance, penalization=False, objective=True)
        problem_type_Mfunc = "PO"
    else:
        raise ValueError("What problem are we solving?")

    ### 2. Span-proxy temperature, then M^*, eta_guaranteed via our algorithm
    temperature_normalized = temperature_from_layers(layers, n_bits)
    min_pfeas = eta_required  # eta

    peak_max = 4  # 4 for NPP and TSP, 25 for PO
    if problem_type == "PO":
        E_LB = lower_bound_obj_PO(N, w, info_instance)
        peak_max = 25
    else:
        E_LB = 0

    # Reference penalty weight M0 at which the span is evaluated. "L1" evaluates
    # the closed-form M_L1 (Eq. 12) at the *normalized* temperature scale, purely
    # as a reference; its exact value only matters when the penalty term
    # dominates the span, in which case any O(1)-consistent choice is reabsorbed.
    if isinstance(M0_ref, str):
        if M0_ref == "L1":
            M0 = L1_norm_hot(Q_obj, const_obj, n_bits, temperature_normalized, min_pfeas)
        else:
            raise ValueError(f"Unknown M0_ref {M0_ref!r}; use 'L1' or pass a float")
    else:
        M0 = float(M0_ref)

    s_conv, UB_obj, UB_pen, ub_pen_method = span_proxy(
        Q_obj, const_obj, Q_pen, const_pen, M0, problem_type_Mfunc, E_LB, use_sdp_pen
    )

    temperature_unnormalized = s_conv * temperature_normalized
    beta = 1 / temperature_unnormalized

    if Mstrategy == "optimality":
        E_f = select_Ef(problem_type, N_idx)
        M_star, eta_guaranteed = M_method_opt(size, problem_type_Mfunc, info_instance, info_type, beta, peak_max, min_pfeas, E_f, E_LB, log_stable=True)
    elif Mstrategy == "feasibility":
        M_star, eta_guaranteed = M_method_feas(size, problem_type_Mfunc, info_instance, info_type, beta, peak_max, min_pfeas, E_LB, log_stable=True)
    else:
        raise ValueError(f"Mstrategy {Mstrategy!r} not understood")

    M_L1 = L1_norm_hot(Q_obj, const_obj, n_bits, temperature_unnormalized, min_pfeas)
    # print(f"M choice from one_shot strategy is {M_star}; span proxy was {s_conv}; M_l1 is {M_L1}")

    if verbose:
        print(f"n_bits = {n_bits}, layers = {layers}")
        print(f"temperature_normalized = {temperature_normalized:.6g}  ->  s_conv = {s_conv:.6g}  ->  temperature_unnormalized = {temperature_unnormalized:.6g}  ->  beta = {beta:.6g}")
        print(f"M0 (ref, {M0_ref}) = {M0:.6g}   M_star = {M_star:.6g}   M_L1 = {M_L1:.6g}")
        print(f"E_LB (obj) = {E_LB:.6g}   UB_obj (SDP) = {UB_obj:.6g}   UB_pen ({ub_pen_method}) = {UB_pen:.6g}")

    ### 3. Run QAOA sampler on QUBO(M^*) and collect samples
    if problem_type == "NPP":
        Q, const = get_QUBO_NPP(N, P, M_star, info_instance)
    elif problem_type[:3] == "TSP":
        Q, const = get_QUBO_TSP(Nc, M_star, info_instance, info_type)
    elif problem_type == "PO":
        Q, const = get_QUBO_PO(N, w, M_star, info_type, info_instance)
    else:
        raise ValueError("What problem are we solving?")

    # Compile normalization only (trainability gauge for Adam); the span already
    # lives in beta, so this does not feed back into beta/M_star.
    
    # norm_final = np.max(np.abs(Q))
    # Q_norm, const_norm = Q/norm_final, const/norm_final
    #### TODO: changed
    Q_norm, const_norm = Q/s_conv, const/s_conv
    norm_final = np.max(np.abs(Q_norm))
    Q_norm, const_norm = Q_norm/norm_final, const_norm/norm_final

    if verbose:
        print(f"norm_final = {norm_final:.6g}")

    Q_dict = {idx: val for idx, val in np.ndenumerate(Q_norm) if not np.isclose(val, 0)}
    circuit_params, states, probs_qaoa, _ = run_qaoa_with_solution(Q_dict, n_bits, p=layers, steps=qaoa_steps, n_samples=qaoa_samples, verbose=verbose_qaoa, draw_samples=True)

    ### 4. From [LCBO, X] computed sampled energies [E_o, E_p]
    eners = np.ndarray((len(states), 3))
    for i, x in enumerate(states):
        x_list = [int(j) for j in x]
        eners[i, 0] = evaluate_energy(x_list, Q_obj, const_obj)
        eners[i, 1] = evaluate_energy(x_list, Q_pen, const_pen)
        eners[i, 2] = states[x]

    ### 5. From [E_o, E_p] compute eta_effective from samples
    if Mstrategy == "feasibility":
        eta_eff = np.sum(np.isclose(eners[:, 1], 0)*eners[:, 2]) / qaoa_samples
    elif Mstrategy == "optimality":
        eta_eff = np.sum(np.logical_and(np.isclose(eners[:, 1], 0), eners[:, 0] <= E_f) * eners[:, 2]) / qaoa_samples

    eta_eff_exact = 0
    if Mstrategy == "feasibility":
        for i in range(2**n_bits):
            if np.isclose( evaluate_energy(state_inttobin(i, n_bits), Q_pen, const_pen), 0 ):
                eta_eff_exact += probs_qaoa[i]
    elif Mstrategy == "optimality":
        for i in range(2**n_bits):
            if np.logical_and(  np.isclose(  evaluate_energy(state_inttobin(i, n_bits), Q_pen, const_pen), 0)  ,  evaluate_energy(state_inttobin(i, n_bits), Q_obj, const_obj) <= E_f   ):
                eta_eff_exact += probs_qaoa[i]

    print(f"Etas: req = {np.round(eta_required, 3)} \t gua = {np.round(eta_guaranteed, 3)} \t eff_ex = {np.round(eta_eff_exact, 3)}")

    t2 = timer()
    if print_time:
        print(f"One instance with size n_bits = {n_bits}, {layers} layers, of {problem_type} took {t2 - t1:.4f}s\n")

    # fill up results dictionary
    results = {}
    results["eta_req"] = eta_required
    results["layers"] = layers
    results["n_bits"] = n_bits
    results["eta_eff_ex"] = eta_eff_exact
    results["norm_final"] = norm_final
    results["temperature_norm"] = temperature_normalized
    results["energies_samples"] = eners
    results["probability_states"] = probs_qaoa
    results["M_star"] = M_star
    results["QUBO"] = (Q, const)
    results["circuit_params"] = circuit_params
    # span-proxy diagnostics
    results["s_conv"] = s_conv
    results["norm_guess"] = s_conv  # backward-compat alias for downstream plotting
    results["M0_ref"] = M0
    results["M_L1"] = M_L1
    results["UB_obj"] = UB_obj
    results["UB_pen"] = UB_pen
    return results





### NEW maQAOA (oneshot strategy)
def maqaoa_sampling_oneshot(problem_type, N_idx, info_instance, Mstrategy, eta_required,
                           layers, qaoa_samples=1000, qaoa_steps=100,
                           M0_ref="L1", use_sdp_pen=True,
                           print_time=True, verbose_qaoa=False, verbose=False):
    """ Run an instance, first using our M algo to compute M^* and then sampling with QAOA the resulting QUBO.
     Fixed are the instance (seed and size), the Mstrategy and relative probablity required (eta) and the temperature schedule for SA, which copies DA's, scaled by a factor """
    t1 = timer()
    
    ### 1. Get LCBO
    if problem_type == "NPP":
        N = Ns[N_idx]
        size = (N, P)
        n_bits = N * P
        Q_pen, const_pen = get_QUBO_NPP(N, P, 1, info_instance, penalization=True, objective=False)
        Q_obj, const_obj = get_QUBO_NPP(N, P, 1, info_instance, penalization=False, objective=True)
        info_type = "seed"
        problem_type_Mfunc = "NPP"
    elif problem_type == "TSP_circle":
        Nc = N_city_circle[N_idx]
        size = (Nc)
        n_bits = Nc**2
        info_instance = None
        info_type = "circle"
        Q_pen, const_pen = get_QUBO_TSP(Nc, 1, None, None, penalization=True, objective=False)
        Q_obj, const_obj = get_QUBO_TSP(Nc, 1, info_instance, info_type, penalization=False, objective=True)
        problem_type_Mfunc = "TSP"
    elif problem_type == "TSP_rand":
        Nc = N_city_rand[N_idx]
        size = (Nc)
        n_bits = Nc**2
        info_type = "seed"
        Q_pen, const_pen = get_QUBO_TSP(Nc, 1, None, None, penalization=True, objective=False)
        Q_obj, const_obj = get_QUBO_TSP(Nc, 1, info_instance, info_type, penalization=False, objective=True)
        problem_type_Mfunc = "TSP"
    elif problem_type == "PO":
        N = N_stocks[N_idx]
        size = (N, w)
        n_bits = N*w
        info_type = "seed"
        Q_pen, const_pen = get_QUBO_PO(N, w, 1, info_type, info_instance, penalization=True, objective=False)
        Q_obj, const_obj = get_QUBO_PO(N, w, 1, info_type, info_instance, penalization=False, objective=True)
        problem_type_Mfunc = "PO"
    else:
        raise ValueError("What problem are we solving?")
    
    ### 2. Span-proxy temperature, then M^*, eta_guaranteed via our algorithm
    temperature_normalized = temperature_from_layers(layers, n_bits)
    min_pfeas = eta_required  # eta

    peak_max = 4 # 4 for NPP and TSP, 9 for PO
    if problem_type == "PO":
        E_LB = lower_bound_obj_PO(N, w, info_instance)
        peak_max = 25
    else:
        E_LB = 0

    # Reference penalty weight M0 at which the span is evaluated. "L1" evaluates
    # the closed-form M_L1 (Eq. 12) at the *normalized* temperature scale, purely
    # as a reference; its exact value only matters when the penalty term
    # dominates the span, in which case any O(1)-consistent choice is reabsorbed.
    if isinstance(M0_ref, str):
        if M0_ref == "L1":
            M0 = L1_norm_hot(Q_obj, const_obj, n_bits, temperature_normalized, min_pfeas)
        else:
            raise ValueError(f"Unknown M0_ref {M0_ref!r}; use 'L1' or pass a float")
    else:
        M0 = float(M0_ref)

    s_conv, UB_obj, UB_pen, ub_pen_method = span_proxy(
        Q_obj, const_obj, Q_pen, const_pen, M0, problem_type_Mfunc, E_LB, use_sdp_pen
    )

    temperature_unnormalized = s_conv * temperature_normalized
    beta = 1 / temperature_unnormalized

    if Mstrategy == "optimality":
        E_f = select_Ef(problem_type, N_idx)
        M_star, eta_guaranteed = M_method_opt(size, problem_type_Mfunc, info_instance, info_type, beta, peak_max, min_pfeas, E_f, E_LB, log_stable = True)
    elif Mstrategy == "feasibility":
        M_star, eta_guaranteed = M_method_feas(size, problem_type_Mfunc, info_instance, info_type, beta, peak_max, min_pfeas, E_LB, log_stable = True)
    M_L1 = L1_norm_hot(Q_obj, const_obj, n_bits, temperature_unnormalized, min_pfeas)
    # print(f"With an instance with {n_bits} bits, {layers} layers and (thus) est. temp. (UNnorm) {np.round(temperature_unnormalized, 3)} [norm -> {np.round(temperature_normalized, 3)}], the appropriate M is {M_star}")
    # print(f"With an instance with {n_bits} bits, {layers} layers, M* is {M_star}, M_L1 is {M_L1}")

    ### 3. Run QAOA sampler on QUBO(M^*) and collect samples
    if problem_type == "NPP":
        Q, const = get_QUBO_NPP(N, P, M_star, info_instance)
    elif problem_type[:3] == "TSP":
        Q, const = get_QUBO_TSP(Nc, M_star, info_instance, info_type)
    elif problem_type == "PO":
        Q, const = get_QUBO_PO(N, w, M_star, info_type, info_instance)
    else:
        raise ValueError("What problem are we solving?")

    # Normalizing Q according to the temperature scaling we did in step 2
    Q_norm, const_norm = Q/s_conv, const/s_conv
    norm_final = np.max(np.abs(Q_norm))
    Q_norm, const_norm = Q_norm/norm_final, const_norm/norm_final

    Q_dict = {idx: val for idx, val in np.ndenumerate(Q_norm) if not np.isclose(val, 0)} # Create a dictionary with the non-zero entries of the Q matrix
    circuit_params, states, probs_qaoa, _ = run_ma_qaoa_with_solution(
        Q_dict, n_bits,
        p=layers,                 # try small p first; expressivity per layer is much higher
        steps=qaoa_steps,
        n_samples=qaoa_samples,
        verbose=verbose_qaoa,
        draw_samples=True,
        seed=442,
    )

    ### 4. From [LCBO, X] computed sampled energies [E_o, E_p]
    eners = np.ndarray((len(states), 3))
    for i, x in enumerate(states):
        x_list = [int(j) for j in x]
        eners[i, 0] = evaluate_energy(x_list, Q_obj, const_obj)
        eners[i, 1] = evaluate_energy(x_list, Q_pen, const_pen)
        eners[i, 2] = states[x]

    eta_eff_exact = 0
    if Mstrategy == "feasibility":
        for i in range(2**n_bits):
            if np.isclose( evaluate_energy(state_inttobin(i, n_bits), Q_pen, const_pen), 0 ):
                eta_eff_exact += probs_qaoa[i]
    elif Mstrategy == "optimality":
        for i in range(2**n_bits):
            if np.logical_and(  np.isclose(  evaluate_energy(state_inttobin(i, n_bits), Q_pen, const_pen), 0)  ,  evaluate_energy(state_inttobin(i, n_bits), Q_obj, const_obj) <= E_f   ):
                eta_eff_exact += probs_qaoa[i]
                
    print(f"Etas: req = {np.round(eta_required, 3)} \t gua = {np.round(eta_guaranteed, 3)} \t eff_ex = {np.round(eta_eff_exact, 3)}")

    t2 = timer()
    if print_time:
        print(f"One instance with size n_bits = {n_bits}, {layers} layers, of {problem_type} took {t2 - t1:.4f}s\n")

    # fill up results dictionary
    results = {}
    results["eta_req"] = eta_required
    results["layers"] = layers
    results["n_bits"] = n_bits
    results["eta_eff_ex"] = eta_eff_exact
    results["norm_final"] = norm_final
    results["temperature_norm"] = temperature_normalized
    results["energies_samples"] = eners
    results["probability_states"] = probs_qaoa
    results["M_star"] = M_star
    results["QUBO"] = (Q, const)
    results["circuit_params"] = circuit_params
    # span-proxy diagnostics
    results["s_conv"] = s_conv
    results["norm_guess"] = s_conv  # backward-compat alias for downstream plotting
    results["M0_ref"] = M0
    results["M_L1"] = M_L1
    results["UB_obj"] = UB_obj
    results["UB_pen"] = UB_pen
    return results
