import numpy as np
import random
from myalgo import (
    M_method_feas,
    M_method_opt,
)  ### TODO: add E_f estimation for the optimality strategy too
import os
import json

from simulated_annealing import simulated_annealing
from timeit import default_timer as timer

# Note: the following code implements SA simulation flow for NPP only for now. TODO: add TSP (and PO) functions soon


# useful functions
def get_QUBO(N, P, M, seed, penalization=True, objective=True):  # TODO: add TSP
    """build a NPP QUBO with N integers, P partitions, M penalty factor, for problem identified with seed"""
    Q, const = np.zeros((N * P, N * P)), 0
    if objective:
        numbs = build_numbs_set(N, P, seed)
        Ho, const_o = build_obj(numbs, N, P)
        Q += Ho
        const += const_o
    if penalization:
        Hp, const_p = build_pen(N, P)
        Q += M * Hp
        const += M * const_p
    return Q, const


def build_numbs_set(N, P, seed):
    # rng = np.random.Generator(np.random.PCG64(seed=seed))
    # numbs = rng.random.uniform(size=(N,))
    random.seed(seed)
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


def build_obj(numbs, N, P):
    H = np.zeros((N * P, N * P))
    alpha = np.sum(numbs) / P

    for p in range(P):
        for i in range(N):
            H[idx_onehot(i, p, P), idx_onehot(i, p, P)] = (
                numbs[i] ** 2 - 2 * alpha * numbs[i]
            )
            for i_prime in range(N):
                if i_prime == i:
                    continue
                H[idx_onehot(i, p, P), idx_onehot(i_prime, p, P)] = (
                    numbs[i] * numbs[i_prime]
                )
    const = P * alpha**2
    H = symmetrize(H)
    return H, const


def build_pen(N, P):
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


def symmetrize(Q):
    return (Q + Q.T) / 2


def L1_norm(Q, const):
    return const + np.sum(np.abs(Q))


def L1_norm_hot(Q, const, n_bits, temperature, min_pfeas):
    return L1_norm(Q, const) + temperature * (n_bits * np.log(2) - np.log(min_pfeas))


def evaluate_energy(solution, Q, const):
    return const + np.dot(solution, np.dot(Q, solution))


def copy_DA_temperatures(N_idx, vseed):
    directory = "../data/scan_NPP/"
    N = Ns[N_idx]
    P = Ps[N_idx]
    bits = P * N
    filename = directory + f"results-N={N}_P={P}-short.json"
    if not os.path.isfile(filename):
        raise ValueError(f"{filename} doesn't exist")

    with open(filename, "r") as fp:
        summary = json.load(fp)
        Ms = sorted(list(summary.keys()))
        temp_i, temp_f = np.zeros((len(Ms))), np.zeros((len(Ms)))
        for M_idx, M in enumerate(Ms):
            t_i, t_f = [], []
            for seed in summary[M]:
                if seed[:2] == str(vseed) and seed[-4:] == "data":
                    t_i.append(summary[M][seed]["temperature_start"])
                    t_f.append(summary[M][seed]["temperature_end"])
            temp_i[M_idx] = np.mean(
                t_i
            )  # mean across the value and DA-initialization seeds
            temp_f[M_idx] = np.mean(t_f)
    return np.mean(temp_i), np.mean(temp_f)  # mean across Ms


def temperature_schedule_exp(Ti, Tf, n_steps):
    # compute the n_steps temperatures from Ti to Tf varying with an exponential law: reduce temperature by factor (1-gamma) every step.
    # Total number of different temperatures: n_steps
    gamma = 1 - np.power(Ti / Tf, 1 / (n_steps - 1))
    Ts = np.array([np.power(1 - gamma, n_steps - (i + 1)) for i in range(n_steps)]) * Tf
    return Ts


######### MAIN


### set framework
problem_type = "NPP"
vseeds = np.arange(42, 46)
Ps = np.arange(2, 15)
Ns = 8 * Ps


### 1. Get LCBO
## choose size of the problem and specific instance
N_idx = 2  # between 0 and 13
vseed_idx = 3  # between 0 and 3
vseed = vseeds[vseed_idx]
N, P = Ns[N_idx], Ps[N_idx]
size = (N, P)
print(f"NPP problem, size:  N = {N}, P = {P}")
Q_pen, const_pen = get_QUBO(N, P, 0, vseed, penalization=True, objective=False)
Q_obj, const_obj = get_QUBO(N, P, 0, vseed, penalization=False, objective=True)


### 2. Choose an annealing schedule for SA (same as DA? same only its final temperature? completely different to adapt to SA different scheme?)
# same as DA for now
temp_initial, temp_final = copy_DA_temperatures(N_idx, vseed)
beta_final = 1 / temp_final
n_steps = (N * P) ** 2
temperature_schedule = temperature_schedule_exp(temp_initial, temp_final, n_steps)


### 3. From [LCBO, \beta_{final}] compute M^*, \eta_{guarantee} using our algorithm. Also, compute M_{\ell_1}
min_pfeas = 0.5  # eta   ### consider changing it later on
peak_max = 4
E_LB = 0
# E_f = 1e7  ### needed only for M_method_opt, not for M_method_feas
# M, eta_guaranteed = M_method_opt(size, problem_type, vseed, "seed", beta_final, peak_max, min_pfeas, E_f, E_LB)
M_star, eta_guaranteed = M_method_feas(
    size, problem_type, vseed, "seed", beta_final, peak_max, min_pfeas, E_LB
)
M_L1 = L1_norm_hot(Q_obj, const_obj, N * P, 1 / beta_final, min_pfeas)


### 4. Run SA on QUBO(M^*) and collect samples X
Q, const = get_QUBO(N, P, M_star, vseed)
# TODO: implement something that does the following
# samples = SA_algorithm(Q, const, temperature_schedule, n_samples=100)
num_samples = 8
samples = []
t1 = timer()
for _ in range(num_samples):
    (x, _) = simulated_annealing(
        Q,
        t_0=temp_initial,
        t_end=temp_final,
        num_t_values=n_steps,
        constant_temperature_steps=N,
    )
    samples.append(x)
t2 = timer()
print(f"{num_samples} simulated annealing runs took {t2 - t1:.4f}s.")


### 5. From [LCBO, X] computed sampled energies [E_o, E_p](both objective and penalization)
eners = np.ndarray(
    (len(samples), 2)
)  # second index discriminate obj=0 energy and pen=1 energy
for i, x in enumerate(samples):
    eners[i, 0] = evaluate_energy(x, Q_obj, const_obj)
    eners[i, 1] = evaluate_energy(x, Q_pen, const_pen)


### 6. From [E_o, E_p] compute \eta_{effective} and prob_{feas}
eta_eff = np.sum(eners[:, 1] == 0) / len(samples)
print(
    f"For this NPP instance of size N={N}, P={P}, seed={vseed}, the selected M is {M_star}, that gives effective success probability of {eta_eff} vs the guaranteed {eta_guaranteed} and the required {min_pfeas}"
)
print("M_l1 = ", M_L1)
