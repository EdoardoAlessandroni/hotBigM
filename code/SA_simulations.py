import numpy as np
import random
from myalgo import M_method_feas, M_method_opt
import os
import json
import pickle
from tqdm import tqdm

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
    """ Copy the final and initial temperatures that DA solver used for the instance with seed = vseed, from the dataset iindicated in the directory """
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
            temp_i[M_idx] = np.mean(t_i)  # mean across the value and DA-initialization seeds
            temp_f[M_idx] = np.mean(t_f)
    return np.array([np.mean(temp_i), np.mean(temp_f)])  # mean across Ms


def temperature_schedule_exp(Ti, Tf, n_steps):
    # compute the n_steps temperatures from Ti to Tf varying with an exponential law: reduce temperature by factor (1-gamma) every step.
    # Total number of different temperatures: n_steps
    gamma = 1 - np.power(Ti / Tf, 1 / (n_steps - 1))
    Ts = np.array([np.power(1 - gamma, n_steps - (i + 1)) for i in range(n_steps)]) * Tf
    return Ts

def select_Ef(n_bits):
    """ Choose E_f prop. to the square of nbits, since the number of terms in the Hamiltonian increase like this. 
    The costant is chosen such that it's possible to sample even at eta_req = 0.75 at temperatures = DA_temp * 100 """
    return 300 * n_bits**2 # E_f is chosen proportional to the square of the number of bits


def run_instance_Mstrategy_DAtemps(dict_run, problem_type, N_idx, vseed, temperatures, Mstrategy, eta_required, SA_samples = 128):
    """ Run an instance, first using our M algo to compute M^* and then sampling with SA the resulting QUBO.
     Fixed are the instance (seed and size), the Mstrategy and relative probablity required (eta) and the temperature schedule for SA, which copies DA's, scaled by a factor """
    t1 = timer()
    
    ### 1. Get LCBO
    if problem_type == "NPP":
        N, P = Ns[N_idx], Ps[N_idx]
        size = (N, P)
        n_bits = N * P
    else:
        raise ValueError("What problem are we solving?")
    Q_pen, const_pen = get_QUBO(N, P, 1, vseed, penalization=True, objective=False)
    Q_obj, const_obj = get_QUBO(N, P, 1, vseed, penalization=False, objective=True)

    ### 2. Choose an annealing schedule for SA (same as DA, scaled)
    temp_initial, temp_final = temperatures
    beta_final = 1 / temp_final
    n_steps_SA = n_bits ** 2

    ### 3. From [LCBO, \beta_{final}] compute M^*, \eta_{guarantee} using our algorithm. Also, compute M_{\ell_1}
    min_pfeas = eta_required  # eta
    peak_max = 4
    if problem_type == "NPP":
        E_LB = 0
    if Mstrategy == "optimality":
        E_f = select_Ef(n_bits)
        M_star, eta_guaranteed = M_method_opt(size, problem_type, vseed, "seed", beta_final, peak_max, min_pfeas, E_f, E_LB)
    elif Mstrategy == "feasibility":
        M_star, eta_guaranteed = M_method_feas(size, problem_type, vseed, "seed", beta_final, peak_max, min_pfeas, E_LB)
    M_L1 = L1_norm_hot(Q_obj, const_obj, N * P, 1 / beta_final, min_pfeas)

    ### 4. Run SA on QUBO(M^*) and collect samples X
    Q, const = get_QUBO(N, P, M_star, vseed)
    samples = []
    for _ in range(SA_samples):
        x, _ = simulated_annealing(Q, temp_initial, temp_final, num_t_values=n_steps_SA, constant_temperature_steps=N)
        samples.append(x)

    ### 5. From [LCBO, X] computed sampled energies [E_o, E_p](both objective and penalization)
    eners = np.ndarray((len(samples), 2))  # second index discriminate obj=0 energy and pen=1 energy
    for i, x in enumerate(samples):
        eners[i, 0] = evaluate_energy(x, Q_obj, const_obj)
        eners[i, 1] = evaluate_energy(x, Q_pen, const_pen)

    ### 6. From [E_o, E_p] compute \eta_{effective}
    if Mstrategy == "feasibility":
        eta_eff = np.sum(eners[:, 1] == 0) / len(samples)
    elif Mstrategy == "optimality":
        eta_eff = np.sum(  np.logical_and(eners[:, 1] == 0, eners[:, 0] <= E_f)) / len(samples)

    # save data
    dict_run["eta_required"] = eta_required
    dict_run["eta_guaranteed"] = eta_guaranteed
    dict_run["eta_effective"] = eta_eff
    dict_run["M_star"] = M_star
    dict_run["M_L1"] = M_L1

    t2 = timer()
    print(f"One instance took {t2 - t1:.4f}s")
    return 


def run_database(N_idx, vseeds, M_strategies, etas_req, temperature_scalers):
    data = {}
    for vseed in tqdm(vseeds):
        data["vseed_"+str(vseed)] = {}
        E_f = select_Ef(Ns[N_idx] * Ps[N_idx])
        data["vseed_"+str(vseed)]["E_f"] = E_f
        for DAtemps_scaler in temperature_scalers:
            data["vseed_"+str(vseed)]["Tscale_"+str(DAtemps_scaler)] = {}
            temps = copy_DA_temperatures(N_idx, vseed) * DAtemps_scaler
            data["vseed_"+str(vseed)]["Tscale_"+str(DAtemps_scaler)]["temp_initial"] = temps[0]
            data["vseed_"+str(vseed)]["Tscale_"+str(DAtemps_scaler)]["temp_final"] = temps[1]
            for Mstrategy in M_strategies:
                data["vseed_"+str(vseed)]["Tscale_"+str(DAtemps_scaler)][Mstrategy] = {}
                for eta_idx, eta_required in enumerate(etas_req):
                    data["vseed_"+str(vseed)]["Tscale_"+str(DAtemps_scaler)][Mstrategy]["eta_run_"+str(eta_idx)] = {}
                    dict_run = data["vseed_"+str(vseed)]["Tscale_"+str(DAtemps_scaler)][Mstrategy]["eta_run_"+str(eta_idx)] 
                    run_instance_Mstrategy_DAtemps(dict_run, problem_type, N_idx, vseed, temps, Mstrategy, eta_required)
    return data


######### MAIN


### set framework of the dataset
problem_type = "NPP"
Ps = np.arange(2, 15)
Ns = 8 * Ps

N_idx = 4 # between 0 and 13
vseeds = range(42,46) # between 42 and 45
M_strategies = ["feasibility", "optimality"]
etas_req = [.25, .5, .75]
temperature_scalers = [1, 10, 100] # only as integers, for keys of dictionary


filename = f"../data/SA_NPP/results-N={Ns[N_idx]}_P={Ps[N_idx]}.txt"
if os.path.exists(filename):
    raise ValueError(f"Filename {filename} already exists, are you sure you want to overwrite it?")

data = run_database(N_idx, vseeds, M_strategies, etas_req, temperature_scalers)

file = open(filename, "wb")
pickle.dump(data, file)
file.close()