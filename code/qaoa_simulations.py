import numpy as np
import random
from myalgo import M_method_feas, M_method_opt
import os
import pickle
import sys
from timeit import default_timer as timer
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model_reader import ModelReader
# from qibo import gates, hamiltonians, Circuit
# import qiboopt as qopt
from qiboopt.opt_class.opt_class import QUBO
import cvxpy as cp

import pennylane as qml
# from pennylane import numpy as np
from pennylane import qaoa



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




# def state_inttobin(i, n):
#     state_string = bin(i)[2:].zfill(n)
#     return [int(b) for b in state_string]

# def compute_Gibbs_probs(beta, Q, const):
#     n, _ = np.shape(Q)
#     ener = np.array([ evaluate_energy(state_inttobin(i, n), Q, const) for i in range(2**n)])
#     probs = np.exp(-beta*ener)
#     probs /= probs.sum()
#     return probs#, ener

# def sample_Gibbs(probs, n_samples, nbits): # you can generalize the function with input     probs, ener, n_samples, nbits
#     states_sampled_int = np.random.choice(np.arange(2**(nbits)), p = probs, size = n_samples)
#     #eners_sampled = eners[states_sampled_int]
#     states_sampled_bin = [state_inttobin(x, nbits) for x in states_sampled_int]
#     return states_sampled_bin#, eners_sampled 

# def perf_GibbsSampler(beta, Q, const, nbits, n_samples):
#     probs = compute_Gibbs_probs(beta, Q, const)  # you can generalize the function with output    probs, ener
#     return sample_Gibbs(probs, n_samples, nbits) # you can generalize the function with input     probs, ener, n_samples, nbits

### general

def symmetrize(Q):
    return (Q + Q.T) / 2

def L1_norm(Q, const):
    return const + np.sum(np.abs(Q))

def L1_norm_hot(Q, const, n_bits, temperature, min_pfeas):
    return L1_norm(Q, const) + temperature * (n_bits * np.log(2) - np.log(1-min_pfeas))

def evaluate_energy(solution, Q, const):
    return const + np.dot(solution, np.dot(Q, solution))


# def pick_temperature(problem_type, N_idx):
#     """ Pick a temperature for the Gibbs solver """
#     temp = 1e4
#     return temp


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
    


def run_qaoa_with_solution(qubo_dict, num_qubits, p, opt_type="Adam", steps=100, qaoa_samples=10000):
    # 1. Hamiltonian Construction
    coeffs, obs = [], []
    for (i, j), val in qubo_dict.items():
        if i == j: # Linear mapping
            coeffs.append(-val / 2); obs.append(qml.PauliZ(i))
        else:      # Quadratic mapping
            coeffs.append(val / 4); obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
            coeffs.append(-val / 4); obs.append(qml.PauliZ(i))
            coeffs.append(-val / 4); obs.append(qml.PauliZ(j))

    cost_h = qml.Hamiltonian(coeffs, obs)
    mixer_h = qaoa.x_mixer(range(num_qubits))
    dev = qml.device("default.qubit", wires=num_qubits)

    # Core QAOA Circuit logic
    def qaoa_circuit(params):
        for i in range(num_qubits): qml.Hadamard(wires=i)
        gammas, betas = params[0], params[1]
        for i in range(p):
            qaoa.cost_layer(gammas[i], cost_h)
            qaoa.mixer_layer(betas[i], mixer_h)

    # QNode for training (expectation value)
    @qml.qnode(dev)
    def cost_qnode(params):
        qaoa_circuit(params)
        return qml.expval(cost_h)

    # QNode for final result (probabilities)
    @qml.qnode(dev)
    def probability_qnode(params):
        qaoa_circuit(params)
        return qml.probs(wires=range(num_qubits))

    @qml.set_shots(shots=10000)
    @qml.qnode(dev)
    def measure_qnode(params):
        qaoa_circuit(params)
        return qml.counts(wires=range(num_qubits))

    # 2. Linear Ramp Initialization
    raw_gammas = np.linspace(0.01, 0.5, p)
    raw_betas = np.linspace(0.5, 0.01, p)
    params = np.array([raw_gammas, raw_betas], requires_grad=True)

    # 3. Optimization
    opt_map = {
        "Adam": qml.AdamOptimizer(stepsize=0.05),
        "Adagrad": qml.AdagradOptimizer(stepsize=0.1),
        "RMSProp": qml.RMSPropOptimizer(stepsize=0.01)
    }
    optimizer = opt_map.get(opt_type, opt_map[opt_type])

    for i in range(steps):
        params = optimizer.step(cost_qnode, params)
        if (i + 1) % 10 == 0:
            print(f"Step {i+1}: Cost = {cost_qnode(params):.4f}")

    # 4. Extracting the Best Solution
    probs = probability_qnode(params)
    best_idx = np.argmax(probs)
    # Convert the index of the highest probability to a bitstring
    best_bitstring = format(best_idx, f'0{num_qubits}b')
    counts = measure_qnode(params)

    return best_bitstring, params, counts






def run_qaoa(dict_run, problem_type, N_idx, info_instance, Mstrategy, eta_required, layers, qaoa_samples = 1000, print_time = True):
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

    temperature = temperature_from_layers(layers, n_bits)

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
        M_star, eta_guaranteed = M_method_opt(size, problem_type_Mfunc, info_instance, info_type, beta, peak_max, min_pfeas, E_f, E_LB, log_stable = True)
    elif Mstrategy == "feasibility":
        M_star, eta_guaranteed = M_method_feas(size, problem_type_Mfunc, info_instance, info_type, beta, peak_max, min_pfeas, E_LB, log_stable = True)
    M_L1 = L1_norm_hot(Q_obj, const_obj, n_bits, temperature, min_pfeas)
    print(f"With an instance with {n_bits} bits, {layers} layers and (thus) est. temp. {temperature}, the appropriate M is {M_star}")

    ### 3. Run QAOA sampler on QUBO(M^*) and collect samples
    if problem_type == "NPP":
        Q, const = get_QUBO_NPP(N, P, M_star, info_instance)
    elif problem_type[:3] == "TSP":
        Q, const = get_QUBO_TSP(Nc, M_star, info_instance, info_type)
    elif problem_type == "PO":
        Q, const = get_QUBO_PO(N, w, M_star, info_type, info_instance)
    else:
        raise ValueError("What problem are we solving?")
    
    norm = Q.max() #If we want to normalize Q
    # norm = 1

    ### Create a dictionary with the non-zero entries of the Q matrix
    Q_dict = {idx: val/norm for idx, val in np.ndenumerate(Q) if not np.isclose(val, 0)}

    best_bitstring, params, counts = run_qaoa_with_solution(Q_dict, n_bits, p=layers, opt_type="Adam", steps=100, qaoa_samples=10000)

    # qubo_obj = QUBO(0, Q_dict)
    # result = qubo_obj.train_QAOA(p=layers, method='COBYLA', maxiter=1000, nshots=qaoa_samples)
    # states = result[4]

    ### 4. From [LCBO, X] computed sampled energies [E_o, E_p](both objective and penalization)
    eners = np.ndarray((len(counts), 3))  # second index discriminate obj=0 energy and pen=1 energy   ...   and the number of those states for idx=2?
    for i, x in enumerate(counts):
        x_list = [int(j) for j in x]
        eners[i, 0] = evaluate_energy(x_list, Q_obj, const_obj)
        eners[i, 1] = evaluate_energy(x_list, Q_pen, const_pen)
        eners[i, 2] = counts[x]

    ### 5. From [E_o, E_p] compute \eta_{effective}
    if Mstrategy == "feasibility":
        eta_eff = np.sum( np.isclose(eners[:, 1], 0)*eners[:, 2]) / qaoa_samples
    elif Mstrategy == "optimality":
        eta_eff = np.sum( np.logical_and(np.isclose(eners[:, 1], 0), eners[:, 0] <= E_f) * eners[:, 2] ) / qaoa_samples

    # save data
    dict_run["eta_required"] = eta_required
    dict_run["eta_guaranteed"] = eta_guaranteed
    dict_run["eta_effective"] = eta_eff
    dict_run["M_star"] = M_star
    dict_run["M_L1"] = M_L1

    #print(f"Recommended M for strategy: {Mstrategy} is \t {M_star}")
    print(f"Etas: req = {np.round(eta_required, 3)} \t gua = {np.round(eta_guaranteed, 3)} \t eff = {np.round(eta_eff, 3)}")

    t2 = timer()
    if print_time:
        print(f"One instance with size n_bits = {n_bits} of {problem_type} took {t2 - t1:.4f}s\n")
    return 


def run_instance(problem_type, N_idx, info_instance, M_strategy, eta_required, layers):
    if problem_type == "PO":
        vseed = info_instance[1]
    else:
        vseed = info_instance
    data = {}
    data["vseed_"+str(vseed)] = {}
    E_f = select_Ef(problem_type, N_idx)
    data["vseed_"+str(vseed)]["E_f"] = E_f
    data["vseed_"+str(vseed)]["layers_"+str(layers)] = {}
    data["vseed_"+str(vseed)]["layers_"+str(layers)]["layers"] = layers
    data["vseed_"+str(vseed)]["layers_"+str(layers)][M_strategy] = {}
    data["vseed_"+str(vseed)]["layers_"+str(layers)][M_strategy]["eta_req_"+str(eta_required)] = {}
    dict_run = data["vseed_"+str(vseed)]["layers_"+str(layers)][M_strategy]["eta_req_"+str(eta_required)] 
    run_qaoa(dict_run, problem_type, N_idx, info_instance, M_strategy, eta_required, layers, qaoa_samples = qaoa_samples) # Gibbs_samples = 1000 for NPP and TSP
    return data

######### NEW FUNCTION

def temperature_from_layers(p, N):
    return (2.8*p/(N**(3/2)))**-1


######### MAIN

### set framework of the dataset
Ns = np.arange(3, 9)
P = 3
N_city_circle = np.arange(2, 6)
N_city_rand = np.arange(2, 6)
N_stocks = np.arange(2, 9)
w = 3
directory = "PO_small"

qaoa_samples = 1000

try:
    problem_type = sys.argv[1]
    N_idx = int(sys.argv[2])
    vseed = int(sys.argv[3])
    M_strategy = sys.argv[4]
    layers = int(sys.argv[5])
    eta_req = float(sys.argv[6])
except (IndexError, ValueError):
    print("Wrong usage of code. Correct usage:\npython qaoa_simulations.py problem_model(TSP/NPP/PO) size_index vseed M_strategy(opt/feas) layers eta_required")
    sys.exit(1)
    
info_instance = vseed
if problem_type == "NPP":
    filename = f"../data/qaoa_NPP/results-N={Ns[N_idx]}_P={P}_pars_{N_idx}_{vseed}_{M_strategy}_{layers}_{eta_req}.txt"
elif problem_type == "TSP_circle":
    filename = f"../data/qaoa_TSP_circle/results-Nc={N_city_circle[N_idx]}_pars_{N_idx}_{vseed}_{M_strategy}_{layers}_{eta_req}.txt"
elif problem_type == "TSP_rand":
    filename = f"../data/qaoa_TSP_rand/results-Nc={N_city_rand[N_idx]}_pars_{N_idx}_{vseed}_{M_strategy}_{layers}_{eta_req}.txt"
elif problem_type == "PO":
    filename = f"../data/qaoa_PO/results-N={N_stocks[N_idx]}_w={w}_pars_{N_idx}_{vseed}_{M_strategy}_{layers}_{eta_req}.txt"
    info_instance = (directory,  vseed)
    n_Gibbs_samples = 100_000
print(filename)
if os.path.exists(filename):
    raise ValueError(f"Filename {filename} already exists, are you sure you want to overwrite it?")

data = run_instance(problem_type, N_idx, info_instance, M_strategy, eta_req, layers)

# file = open(filename, "wb")
# pickle.dump(data, file)
# file.close()