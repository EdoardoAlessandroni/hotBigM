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




### general

def symmetrize(Q):
    return (Q + Q.T) / 2

def L1_norm(Q, const):
    return const + np.sum(np.abs(Q))

def L1_norm_hot(Q, const, n_bits, temperature, min_pfeas):
    return L1_norm(Q, const) + temperature * (n_bits * np.log(2) - np.log(1-min_pfeas))


def pick_temperature(problem_type, N_idx):
    """ Pick a temperature for the Gibbs solver """
    temp = 1e3 # TODO: check if that's alright, before it was 1e4
    return temp





def run_Ms(dict_run, problem_type, N_idx, info_instance, temperature, eta_required, print_time = True):
    """ Run an instance, first using our M algo to compute M^* and then sampling with SA the resulting QUBO.
     Fixed are the instance (seed and size), the Mstrategy and relative probablity required (eta) and the temperature schedule for SA, which copies DA's, scaled by a factor """
    t1 = timer()
    
    ### 1. Get LCBO
    if problem_type == "NPP":
        N, P = Ns[N_idx], Ps[N_idx]
        size = (N, P)
        n_bits = N * P
        Q_obj, const_obj = get_QUBO_NPP(N, P, 1, info_instance, penalization=False, objective=True)
        info_type = "seed"
        problem_type_Mfunc = "NPP"
    elif problem_type == "TSP_circle":
        Nc = N_city_circle[N_idx]
        size = (Nc)
        n_bits = Nc**2
        info_instance = None
        info_type = "circle"
        Q_obj, const_obj = get_QUBO_TSP(Nc, 1, info_instance, info_type, penalization=False, objective=True)
        problem_type_Mfunc = "TSP"
    elif problem_type == "TSP_rand":
        Nc = N_city_rand[N_idx]
        size = (Nc)
        n_bits = Nc**2
        info_type = "seed"
        Q_obj, const_obj = get_QUBO_TSP(Nc, 1, info_instance, info_type, penalization=False, objective=True)
        problem_type_Mfunc = "TSP"
    elif problem_type == "PO":
        N = N_stocks[N_idx]
        size = (N, w)
        n_bits = N*w
        info_type = "seed"
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
    M_star, eta_guaranteed = M_method_feas(size, problem_type_Mfunc, info_instance, info_type, beta, peak_max, min_pfeas, E_LB, log_stable = True)
    M_L1 = L1_norm_hot(Q_obj, const_obj, n_bits, temperature, min_pfeas)

    # save data
    dict_run["M_star"] = M_star
    dict_run["M_L1"] = M_L1

    t2 = timer()
    if print_time:
        print(f"One instance with size N_idx = {N_idx} of {problem_type} took {t2 - t1:.4f}s")
    return 


def run_instance(problem_type, N_idx, info_instance, eta_required, temp_scaler):
    if problem_type == "PO":
        vseed = info_instance[1]
    else:
        vseed = info_instance
    data = {}
    data["vseed_"+str(vseed)] = {}
    data["vseed_"+str(vseed)]["Tscale_"+str(temp_scaler)] = {}
    temp = pick_temperature(problem_type, N_idx) * temp_scaler
    data["vseed_"+str(vseed)]["Tscale_"+str(temp_scaler)]["temperature"] = temp
    data["vseed_"+str(vseed)]["Tscale_"+str(temp_scaler)]["eta_req_"+str(eta_required)] = {}
    dict_run = data["vseed_"+str(vseed)]["Tscale_"+str(temp_scaler)]["eta_req_"+str(eta_required)] 
    run_Ms(dict_run, problem_type, N_idx, info_instance, temp, eta_required)
    return data


######### MAIN

### set framework of the dataset
Ps = np.arange(2, 17, 2)
Ns = 8 * Ps
N_city_circle = np.arange(4, 65, 4)
N_city_rand = np.arange(4, 65, 4)
N_stocks = np.array([2, 4, 6, 8, 10, 15, 20, 30, 40])
w = 5
directory = "PO_big"


try:
    problem_type = sys.argv[1]
    N_idx = int(sys.argv[2])
    vseed = int(sys.argv[3])
    temperature_scaler = int(sys.argv[4])
    eta_req = float(sys.argv[5])
except (IndexError, ValueError):
    print("Wrong usage of code. Correct usage:\npython Ms_simulations.py problem_model(TSP/NPP/PO) size_index vseed temperature_scaler eta_required")
    sys.exit(1)
    
info_instance = vseed
if problem_type == "NPP":
    filename = f"../data/Ms_NPP/results-N={Ns[N_idx]}_P={Ps[N_idx]}_pars_{N_idx}_{vseed}_{temperature_scaler}_{eta_req}.txt"
elif problem_type == "TSP_circle":
    filename = f"../data/Ms_TSP_circle/results-Nc={N_city_circle[N_idx]}_pars_{N_idx}_{vseed}_{temperature_scaler}_{eta_req}.txt"
elif problem_type == "TSP_rand":
    filename = f"../data/Ms_TSP_rand/results-Nc={N_city_rand[N_idx]}_pars_{N_idx}_{vseed}_{temperature_scaler}_{eta_req}.txt"
elif problem_type == "PO":
    filename = f"../data/Ms_PO/results-N={N_stocks[N_idx]}_w={w}_pars_{N_idx}_{vseed}_{temperature_scaler}_{eta_req}.txt"
    info_instance = (directory,  vseed)
print(filename)
if os.path.exists(filename):
    raise ValueError(f"Filename {filename} already exists, are you sure you want to overwrite it?")

data = run_instance(problem_type, N_idx, info_instance, eta_req, temperature_scaler)

file = open(filename, "wb")
pickle.dump(data, file)
file.close()