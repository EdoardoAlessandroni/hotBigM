import numpy as np
import random
from myalgo import M_method_feas, M_method_opt
import os
import pickle
import sys
from timeit import default_timer as timer
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model_reader import ModelReader
import cvxpy as cp



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


# TSP specific

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
    # build a TSP QUBO with Nc cities, M penalty factor, for problem identified with seed or on a circle
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


# PO specific

def lower_bound_obj_PO(N, w, info_instance):
    n_bits = N*w
    directory, vseed = info_instance
    filename = f"../data/{directory}/{n_bits}/random{vseed}_{n_bits}.lp"
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
    constraints += [X[i,i] == X[0,i] for i in range(1, n+1) ]
    constraints += [X[0,0] == 1]      ###### alternative way of imposing boundness, rather than  0 <= X_ij <= 1  for all i,j (2 previous lines)
    prob = cp.Problem(cp.Minimize(cp.trace(Q_tilde @ X)), constraints)
    prob.solve(solver = "MOSEK")
    return prob.value

def build_pen_PO(N, w):
    ''' return quadratic part and constant part of the penalization part of the hamiltonian '''
    # constraints in the formulation      Ax = b 
    A = np.array( [2**i for i in range(w-1, -1, -1)]*N )
    b = 2**w - 1
    # build [H, const] from [A, b]. easier formulation since b is scalar and A only 1 row, since there's only 1 constraint.
    H = np.outer(A, A)
    H -= 2*b*np.diag(A)
    const = b**2
    return H, const

def get_PO_obj(n_bits, info_instance):
    directory, vseed = info_instance
    filename = f"../data/{directory}/{n_bits}/random{vseed}_{n_bits}.lp"
    m = ModelReader.read(filename, ignore_names=True)
    qp = from_docplex_mp(m)
    obj = qp.objective
    Q = obj.quadratic.to_array()
    L = obj.linear.to_array()
    return Q, L

def get_QUBO_PO(N, w, M, info_type, info_instance, objective = True, penalization=True):
    ''' build a PO QUBO with N stocks, w qubits per stock, M penalty factor, for problem identified with seed '''
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


### Gibbs sampling


def state_inttobin(i, n):
    state_string = bin(i)[2:].zfill(n)
    return [int(b) for b in state_string]

def compute_Gibbs_probs(beta, Q, const):
    n, _ = np.shape(Q)
    ener = np.array([ evaluate_energy(state_inttobin(i, n), Q, const) for i in range(2**n)])
    probs = np.exp(-beta*ener)
    probs /= probs.sum()
    return probs#, ener

def sample_Gibbs(probs, n_samples, nbits): # you can generalize the function with input     probs, ener, n_samples, nbits
    states_sampled_int = np.random.choice(np.arange(2**(nbits)), p = probs, size = n_samples)
    #eners_sampled = eners[states_sampled_int]
    states_sampled_bin = [state_inttobin(x, nbits) for x in states_sampled_int]
    return states_sampled_bin#, eners_sampled 

def perf_GibbsSampler(beta, Q, const, nbits, n_samples):
    probs = compute_Gibbs_probs(beta, Q, const)  # you can generalize the function with output    probs, ener
    return sample_Gibbs(probs, n_samples, nbits) # you can generalize the function with input     probs, ener, n_samples, nbits



### general

def symmetrize(Q):
    return (Q + Q.T) / 2

def L1_norm(Q, const):
    return const + np.sum(np.abs(Q))

def L1_norm_hot(Q, const, n_bits, temperature, min_pfeas):
    return L1_norm(Q, const) + temperature * (n_bits * np.log(2) - np.log(min_pfeas))

def evaluate_energy(solution, Q, const):
    return const + np.dot(solution, np.dot(Q, solution))


def pick_temperature(problem_type, N_idx):
    """ Pick a temperature for the Gibbs solver """
    temp = 1e4
    return temp


def select_Ef(problem_type, N_idx):
    """ Choose E_f prop. to the square of nbits, since the number of terms in the Hamiltonian increase like this. 
    The costant is chosen such that it's possible to sample even at eta_req = 0.75 at temperatures = DA_temp * 100 """
    if problem_type == "NPP":
        n_bits = Ns[N_idx] * P
        return 3e3 * n_bits**2
    elif problem_type == "TSP_circle":
        n_bits = N_city_circle[N_idx]**2
        return 3e5 * n_bits**2
    elif problem_type == "TSP_rand":
        n_bits = N_city_rand[N_idx]**2
        return 3e5 * n_bits**2
    elif problem_type == "PO":
        n_bits = N_stocks[N_idx] * w
        return 3e-2 * n_bits**2


def run_Gibbs(dict_run, problem_type, N_idx, info_instance, temperature, Mstrategy, eta_required, Gibbs_samples = 128, print_time = True):
    """ Run an instance, first using our M algo to compute M^* and then sampling with SA the resulting QUBO.
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


    ### 2. From [LCBO, \beta] compute M^*, \eta_{guarantee} using our algorithm. Also, compute M_{\ell_1}
    beta = 1 / temperature
    min_pfeas = eta_required  # eta
    peak_max = 4 # 4 for NPP and TSP, 9 for PO
    if problem_type == "PO":
        E_LB = lower_bound_obj_PO(N, w, info_instance)
        peak_max = 25
    else:
        E_LB = 0
    if Mstrategy == "optimality":
        E_f = select_Ef(problem_type, N_idx)
        M_star, eta_guaranteed = M_method_opt(size, problem_type_Mfunc, info_instance, info_type, beta, peak_max, min_pfeas, E_f, E_LB)
    elif Mstrategy == "feasibility":
        M_star, eta_guaranteed = M_method_feas(size, problem_type_Mfunc, info_instance, info_type, beta, peak_max, min_pfeas, E_LB)
    M_L1 = L1_norm_hot(Q_obj, const_obj, n_bits, temperature, min_pfeas)

    ### 3. Run Gibbs sampler on QUBO(M^*) and collect samples
    if problem_type == "NPP":
        Q, const = get_QUBO_NPP(N, P, M_star, info_instance)
    elif problem_type[:3] == "TSP":
        Q, const = get_QUBO_TSP(Nc, M_star, info_instance, info_type)
    elif problem_type == "PO":
        Q, const = get_QUBO_PO(N, w, M_star, info_type, info_instance)
    else:
        raise ValueError("What problem are we solving?")

    states = perf_GibbsSampler(beta, Q, const, n_bits, Gibbs_samples) # in the more general version fo the function the output is      states, Es

    ### 4. From [LCBO, X] computed sampled energies [E_o, E_p](both objective and penalization)
    eners = np.ndarray((len(states), 2))  # second index discriminate obj=0 energy and pen=1 energy
    for i, x in enumerate(states):
        eners[i, 0] = evaluate_energy(x, Q_obj, const_obj)
        eners[i, 1] = evaluate_energy(x, Q_pen, const_pen)

    ### 5. From [E_o, E_p] compute \eta_{effective}
    if Mstrategy == "feasibility":
        eta_eff = np.sum( np.isclose(eners[:, 1], 0)) / len(states)
    elif Mstrategy == "optimality":
        eta_eff = np.sum( np.logical_and( np.isclose(eners[:, 1], 0) , eners[:, 0] <= E_f)) / len(states)

    # save data
    dict_run["eta_required"] = eta_required
    dict_run["eta_guaranteed"] = eta_guaranteed
    dict_run["eta_effective"] = eta_eff
    dict_run["M_star"] = M_star
    dict_run["M_L1"] = M_L1

    #print(f"Recommended M for strategy: {Mstrategy} is \t {M_star}")
    print(f"Etas: req = {np.round(eta_required, 3)} \t gua = {np.round(eta_guaranteed, 3)} \t eff = {np.round(eta_eff, 3)}")

    t2 = timer()
    # if print_time:
    #     print(f"One instance with size N_idx = {N_idx} of {problem_type} took {t2 - t1:.4f}s\n")
    return 


def run_instance(problem_type, N_idx, info_instance, M_strategy, eta_required, temp_scaler):
    if problem_type == "PO":
        vseed = info_instance[1]
    else:
        vseed = info_instance
    data = {}
    data["vseed_"+str(vseed)] = {}
    E_f = select_Ef(problem_type, N_idx)
    data["vseed_"+str(vseed)]["E_f"] = E_f
    data["vseed_"+str(vseed)]["Tscale_"+str(temp_scaler)] = {}
    temp = pick_temperature(problem_type, N_idx) * temp_scaler
    data["vseed_"+str(vseed)]["Tscale_"+str(temp_scaler)]["temperature"] = temp
    data["vseed_"+str(vseed)]["Tscale_"+str(temp_scaler)][M_strategy] = {}
    data["vseed_"+str(vseed)]["Tscale_"+str(temp_scaler)][M_strategy]["eta_req_"+str(eta_required)] = {}
    dict_run = data["vseed_"+str(vseed)]["Tscale_"+str(temp_scaler)][M_strategy]["eta_req_"+str(eta_required)] 
    run_Gibbs(dict_run, problem_type, N_idx, info_instance, temp, M_strategy, eta_required, Gibbs_samples = n_Gibbs_samples) # Gibbs_samples = 1000 for NPP and TSP
    return data


######### MAIN

### set framework of the dataset
Ns = np.arange(3, 9)
P = 3
N_city_circle = np.arange(2, 6)
N_city_rand = np.arange(2, 6)
N_stocks = np.arange(2, 9)
w = 3
directory = "PO_small"
n_Gibbs_samples = 1000

try:
    problem_type = sys.argv[1]
    N_idx = int(sys.argv[2])
    vseed = int(sys.argv[3])
    M_strategy = sys.argv[4]
    temperature_scaler = int(sys.argv[5])
    eta_req = float(sys.argv[6])
except (IndexError, ValueError):
    print("Wrong usage of code. Correct usage:\npython Gibbs_simulations.py problem_model(TSP/NPP/PO) size_index vseed M_strategy(opt/feas) temperature_scaler eta_required")
    sys.exit(1)
    
info_instance = vseed
if problem_type == "NPP":
    filename = f"../data/Gibbs_NPP/results-N={Ns[N_idx]}_P={P}_pars_{N_idx}_{vseed}_{M_strategy}_{temperature_scaler}_{eta_req}.txt"
elif problem_type == "TSP_circle":
    filename = f"../data/Gibbs_TSP_circle/results-Nc={N_city_circle[N_idx]}_pars_{N_idx}_{vseed}_{M_strategy}_{temperature_scaler}_{eta_req}.txt"
elif problem_type == "TSP_rand":
    filename = f"../data/Gibbs_TSP_rand/results-Nc={N_city_rand[N_idx]}_pars_{N_idx}_{vseed}_{M_strategy}_{temperature_scaler}_{eta_req}.txt"
elif problem_type == "PO":
    filename = f"../data/Gibbs_PO/results-N={N_stocks[N_idx]}_w={w}_pars_{N_idx}_{vseed}_{M_strategy}_{temperature_scaler}_{eta_req}.txt"
    info_instance = (directory,  vseed)
    n_Gibbs_samples = 100_000
print(filename)
if os.path.exists(filename):
    raise ValueError(f"Filename {filename} already exists, are you sure you want to overwrite it?")

data = run_instance(problem_type, N_idx, info_instance, M_strategy, eta_req, temperature_scaler)

file = open(filename, "wb")
pickle.dump(data, file)
file.close()