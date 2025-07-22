import numpy as np
import random
from myalgo import M_method_feas, M_method_opt
import os
import json
import pickle
from tqdm import tqdm
import importlib.util
import sys
import time

from simulated_annealing import simulated_annealing
from timeit import default_timer as timer

# Note: the following code implements SA simulation flow for NPP only for now.


### useful functions

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


def build_obj_NPP(numbs, N, P):
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


# TSP specific

def load_adjacency_usecases(cases_name):
    base_path = os.path.abspath("../data/scan_DA_TSP_case/cases_adj")
    file_path = os.path.join(base_path, cases_name + ".py")    
    spec = importlib.util.spec_from_file_location(cases_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, cases_name)
    return np.array(func())

def build_adjacency(Nc, info_instance, info_type):
    if info_type == "circle":
        # trivial construction of adj. matrix of graph with N cities on a circle. Modify this function to get more complex structures
        coordinates = [(1_000_000 * np.cos((index / Nc) * 2 * np.pi), 1_000_000 * np.sin((index / Nc) * 2 * np.pi)) for index in range(Nc)]
        distance_table = [[int(np.sqrt((coordinates[c_1][0] - coordinates[c_0][0]) ** 2 + (coordinates[c_1][1] - coordinates[c_0][1]) ** 2)) for c_0 in range(Nc)] for c_1 in range(Nc)]
    elif info_type == "seed":
        # random construction of the adjacency matrix, based on euclidian distance. (cities are uniformly randomly placed inside a square)
        seed = info_instance
        np.random.seed(seed)
        coordinates = 1e6 * np.array( [np.random.uniform(low = -1, high = 1, size = 2) for i in range(Nc)] )
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
                    H[idx_onehot(t, i, Nc), idx_onehot((t+1)%Nc, i_prime, Nc)] = distances[i, i_prime]
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


def get_QUBO_TSP(Nc, M, info_instance, info_type, objective = True, penalization=True):
    # build a TSP QUBO with Nc cities, M penalty factor, for problem identified with seed
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



### general

def symmetrize(Q):
    return (Q + Q.T) / 2


def L1_norm(Q, const):
    return const + np.sum(np.abs(Q))


def L1_norm_hot(Q, const, n_bits, temperature, min_pfeas):
    return L1_norm(Q, const) + temperature * (n_bits * np.log(2) - np.log(min_pfeas))


def evaluate_energy(solution, Q, const):
    return const + np.dot(solution, np.dot(Q, solution))


def copy_DA_temperatures(problem_type, N_idx, vseed):
    """ Copy the final and initial temperatures that DA solver used for the instance with seed = vseed, from the dataset iindicated in the directory """
    if problem_type == "NPP":
        N = Ns[N_idx]
        P = Ps[N_idx]
        filename = f"../data/scan_NPP/results-N={N}_P={P}-short.json"
    elif problem_type == "TSP_circle":
        Nc = N_city_circle[N_idx]
        filename = f"../data/scan_DA_TSP_circle/results-N={Nc}-short.json"
    elif problem_type == "TSP_case":
        Nc = N_city_case[N_idx]
        filename = f"../data/scan_DA_TSP_case/results-{cases_names[str(Nc)][0]}-short.json"
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

def select_Ef(problem_type, N_idx):
    """ Choose E_f prop. to the square of nbits, since the number of terms in the Hamiltonian increase like this. 
    The costant is chosen such that it's possible to sample even at eta_req = 0.75 at temperatures = DA_temp * 100 """
    if problem_type == "NPP":
        n_bits = Ns[N_idx] * Ps[N_idx]
        return 300 * n_bits**2
    elif problem_type == "TSP_circle":
        n_bits = N_city_circle[N_idx]**2
        return 2.5e4 * n_bits**2
    elif problem_type == "TSP_case":
        n_bits = N_city_case[N_idx]**2
        return 3.5e-2 * n_bits**2


def run_SA(dict_run, problem_type, N_idx, vseed, temperatures, Mstrategy, eta_required, SA_samples = 128, print_time = True):
    """ Run an instance, first using our M algo to compute M^* and then sampling with SA the resulting QUBO.
     Fixed are the instance (seed and size), the Mstrategy and relative probablity required (eta) and the temperature schedule for SA, which copies DA's, scaled by a factor """
    t1 = timer()
    
    ### 1. Get LCBO
    if problem_type == "NPP":
        N, P = Ns[N_idx], Ps[N_idx]
        size = (N, P)
        n_bits = N * P
        Q_pen, const_pen = get_QUBO_NPP(N, P, 1, vseed, penalization=True, objective=False)
        Q_obj, const_obj = get_QUBO_NPP(N, P, 1, vseed, penalization=False, objective=True)
        info_instance = vseed
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
    elif problem_type == "TSP_case":
        Nc = N_city_case[N_idx]
        size = (Nc)
        n_bits = Nc**2
        adj = load_adjacency_usecases(cases_names[str(Nc)][0])
        info_instance = adj
        info_type = "adjacency"
        Q_pen, const_pen = get_QUBO_TSP(Nc, 1, None, None, penalization=True, objective=False)
        Q_obj, const_obj = get_QUBO_TSP(Nc, 1, info_instance, info_type, penalization=False, objective=True)
        problem_type_Mfunc = "TSP"
    else:
        raise ValueError("What problem are we solving?")

    ### 2. Choose an annealing schedule for SA (same as DA, scaled)
    temp_initial, temp_final = temperatures
    beta_final = 1 / temp_final
    n_steps_SA = n_bits ** 2

    ### 3. From [LCBO, \beta_{final}] compute M^*, \eta_{guarantee} using our algorithm. Also, compute M_{\ell_1}
    min_pfeas = eta_required  # eta
    peak_max = 4
    if problem_type == "NPP" or problem_type == "TSP_circle" or problem_type == "TSP_case":
        E_LB = 0
    if Mstrategy == "optimality":
        E_f = select_Ef(problem_type, N_idx)
        M_star, eta_guaranteed = M_method_opt(size, problem_type_Mfunc, info_instance, info_type, beta_final, peak_max, min_pfeas, E_f, E_LB)
    elif Mstrategy == "feasibility":
        M_star, eta_guaranteed = M_method_feas(size, problem_type_Mfunc, info_instance, info_type, beta_final, peak_max, min_pfeas, E_LB)
    M_L1 = L1_norm_hot(Q_obj, const_obj, n_bits, 1 / beta_final, min_pfeas)

    ### 4. Run SA on QUBO(M^*) and collect samples X
    if problem_type == "NPP":
        Q, const = get_QUBO_NPP(N, P, M_star, vseed)
        SA_const_steps = N
    elif problem_type[:3] == "TSP":
        Q, const = get_QUBO_TSP(Nc, M_star, info_instance, info_type)
        SA_const_steps = Nc
    else:
        raise ValueError("What problem are we solving?")

    samples = []
    for _ in range(SA_samples):
        x, _ = simulated_annealing(Q, temp_initial, temp_final, num_t_values=n_steps_SA, constant_temperature_steps=SA_const_steps)
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

    print(f"Etas: req = {np.round(eta_required, 3)} \t gua = {np.round(eta_guaranteed, 3)} \t eff = {np.round(eta_eff, 3)}")

    t2 = timer()
    if print_time:
        print(f"One instance with size N_idx = {N_idx} of {problem_type} took {t2 - t1:.4f}s")
    return 


def run_instance(problem_type, N_idx, vseed, M_strategy, eta_required, DAtemp_scaler):
    data = {}
    data["vseed_"+str(vseed)] = {}
    E_f = select_Ef(problem_type, N_idx)
    data["vseed_"+str(vseed)]["E_f"] = E_f
    data["vseed_"+str(vseed)]["Tscale_"+str(DAtemp_scaler)] = {}
    temps = copy_DA_temperatures(problem_type, N_idx, vseed) * DAtemp_scaler
    print("Computed temperatures are ", temps)
    data["vseed_"+str(vseed)]["Tscale_"+str(DAtemp_scaler)]["temp_initial"] = temps[0]
    data["vseed_"+str(vseed)]["Tscale_"+str(DAtemp_scaler)]["temp_final"] = temps[1]
    data["vseed_"+str(vseed)]["Tscale_"+str(DAtemp_scaler)][M_strategy] = {}
    data["vseed_"+str(vseed)]["Tscale_"+str(DAtemp_scaler)][M_strategy]["eta_req_"+str(eta_required)] = {}
    dict_run = data["vseed_"+str(vseed)]["Tscale_"+str(DAtemp_scaler)][M_strategy]["eta_req_"+str(eta_required)] 
    run_SA(dict_run, problem_type, N_idx, vseed, temps, M_strategy, eta_required)
    return data


######### MAIN

### set framework of the dataset
Ps = np.arange(2, 15)
Ns = 8 * Ps
N_city_circle = np.arange(4, 27, 2)
N_city_case = np.array([14, 16, 17, 21, 22, 26])
cases_names = {'38': ['ftv38'], '33': ['ftv33'], '42': ['dantzig42', 'swiss42'], '48': ['ry48p', 'hk48', 'gr48', 'att48'], '44': ['ftv44'], '43': ['p43'], '17': ['br17', 'gr17'], '53': ['ft53'], '21': ['gr21'], '55': ['ftv55'], '58': ['brazil58'], '14': ['burma14'], '29': ['bayg29', 'bays29'], '16': ['ulysses16'], '35': ['ftv35'], '47': ['ftv47'], '52': ['berlin52'], '22': ['ulysses22'], '26': ['fri26']}
# N_idx = 0 # between 0 and 13
# vseeds = range(42,46) # between 42 and 45
# M_strategies = ["feasibility", "optimality"]
# temperature_scalers = [1, 10, 100] # only as integers, for keys of dictionary
# etas_req = [.25, .5, .75]

try:
    problem_type = sys.argv[1]
    N_idx = int(sys.argv[2])
    vseed = int(sys.argv[3])
    M_strategy = sys.argv[4]
    temperature_scaler = int(sys.argv[5])
    eta_req = float(sys.argv[6])
except (IndexError, ValueError):
    print("Wrong usage of code. Correct usage:\npython SA_simulations.py problem_model(TSP/NPP) size_index vseed M_strategy(opt/feas) DA_temperature_scaler eta_required")
    sys.exit(1)
    

if problem_type == "NPP":
    filename = f"../data/SA_NPP/results-N={Ns[N_idx]}_P={Ps[N_idx]}_pars_{N_idx}_{vseed}_{M_strategy}_{temperature_scaler}_{eta_req}.txt"
elif problem_type == "TSP_circle":
    filename = f"../data/SA_TSP_circle/results-Nc={N_city_circle[N_idx]}_pars_{N_idx}_{vseed}_{M_strategy}_{temperature_scaler}_{eta_req}.txt"
elif problem_type == "TSP_case":
    filename = f"../data/SA_TSP_case/results-Nc={N_city_case[N_idx]}_pars_{N_idx}_{vseed}_{M_strategy}_{temperature_scaler}_{eta_req}.txt"
print(filename)
if os.path.exists(filename):
    raise ValueError(f"Filename {filename} already exists, are you sure you want to overwrite it?")

data = run_instance(problem_type, N_idx, vseed, M_strategy, eta_req, temperature_scaler)

file = open(filename, "wb")
pickle.dump(data, file)
file.close()