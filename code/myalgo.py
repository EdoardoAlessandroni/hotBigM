import numpy as np
from scipy.special import factorial, comb
import random
from timeit import default_timer as timer
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model_reader import ModelReader
from scipy.special import logsumexp as LSE



######### General auxiliary functions


def sample_uniform_feasible(size, info_instance, info_type, problem_type, n_samples):
    ''' Returns n_samples energies, uniformly sampled in the feasible subspace'''
    match problem_type:
        case "NPP":
            N, P = size
            match info_type: 
                case "numbers":
                    eners = RandomSampler_Feasible_NPP_wnumbersset(N, P, np.array(info_instance), n_samples)
                case "seed":
                    eners = RandomSampler_Feasible_NPP_wseed(N, P, info_instance, n_samples)
                case _:
                    raise ValueError(f"Info type {info_type} is not amomg the implemented ones")
        case "TSP":
            Nc = size
            match info_type: 
                case "adjacency":
                    eners = RandomSampler_Feasible_TSP_wadjacency(Nc, info_instance, n_samples)
                case "seed":
                    eners = RandomSampler_Feasible_TSP_wseed(Nc, info_instance, n_samples)
                case "circle":
                    eners = RandomSampler_Feasible_TSP_circle(Nc, n_samples)
                case _:
                    raise ValueError(f"Info type {info_type} is not amomg the implemented ones")
        case "PO":
            N, w = size
            if info_type == "seed":
                eners = RandomSampler_Feasible_PO_wseed(N, w, info_instance, n_samples)
            else:
                raise ValueError(f"Info type {info_type} is not amomg the implemented ones")
        case _:
            raise ValueError(f"Problem type {problem_type} is not amomg the implemented ones")
    return eners


def logsafe(arr):
    """Return log(arr) but map zeros -> -inf, negatives raise."""
    arr = np.asarray(arr)
    if np.any(arr < 0):
        raise ValueError("Negative entries not allowed where logs are taken.")
    out = np.full_like(arr, -np.inf, dtype=float)
    mask = arr > 0
    out[mask] = np.log(arr[mask])
    return out


def n_violations_problem(problem_type, size, peak_max):
    ''' Returns the vector of violations, i.e. number of bitstrings that violate the constraints by a particular violation number '''
    match problem_type:
        case "NPP":
            N, P = size
            n_viols = n_viols_exact_NPP(N, P, peak_max)
        case "TSP":
            Nc = size
            n_viols = n_viols_exact_TSP(Nc, peak_max)
        case "PO":
            N, w = size
            n_viols = n_viols_exact_PO(N, w, peak_max)
    return n_viols

def my_root_finder(f, x0):
    ''' Forces usage of np.longdouble to find root of (monotonically decreasing (f)'''
    x = np.float128(x0)
    tol = 1

    # find interval (multiply by 10)
    n_iter = 0
    if f(x) < 0:
        interval = [0, x]
    else:
        while f(x) >= 0:
            if f(x) == 0:
                return x
            n_iter += 1
            x_old = x
            x = 10*x
            if n_iter == 100:
                raise ValueError("Iteration 100 of first bisection algorithm part reached (finding the interval)")
        interval = [x_old, x]

    # pinpoint root (divide by 2)
    while interval[1] - interval[0] > tol:
        n_iter += 1
        midpoint = (interval[1] + interval[0])/2
        if f(midpoint) > 0:
            interval = [midpoint, interval[1]]
        else:
            interval = [interval[0], midpoint]
    # root = (interval[1] + interval[0])/2
    root = interval[1] ### try to return the endpoint rather than the midpoint, to be sure that the decreasing function is negative there
    #print(f"n_iters of my_root_finder = {n_iter1}")
    return root







######### Feasibility strategy


### Main function
def M_method_feas(size, problem_type, info_instance, info_type, beta, peak_max, min_pfeas, E_LB, log_stable = False):
    ''' Main algorithm for the feasibility strategy. Returns penalty weigth M (and guaranteed success prob) such that QUBO reformulation with M, sampled by a Gibbs sampler at temperature beta, guarantees a certain success probability'''
    ## evaluate violation peaks and cumulatives
    n_viols = n_violations_problem(problem_type, size, peak_max)
    if log_stable:
        cumul_exp_feas_log = cumulative_exp_integral_feas_log(beta, size, info_instance, info_type, problem_type, E_LB, n_viols[0])
        integral = cumul_exp_feas_log
    else:
        cumul_exp_feas = cumulative_exp_integral_feas(beta, size, info_instance, info_type, problem_type, E_LB, n_viols[0])
        integral = cumul_exp_feas
    
    # Ensure poly is correctly built
    # check if poly(1)>0
    min_pfeas_triviality = f_min_pfeas_triviality_feas(n_viols[1:], integral, beta, E_LB, log_stable)
    if min_pfeas <= min_pfeas_triviality:
        print(f"With tolerance {min_pfeas}, the sampling probability requirement is satisfied for any M. Setting M=0.")
        return 0, min_pfeas
    
    ## build function 
    if log_stable:
        p_feas_term = - np.log(1/min_pfeas - 1) - integral
    else:
        p_feas_term = -(1/min_pfeas - 1) * integral
    func_M = build_func_for_root(n_viols, beta, E_LB, p_feas_term, log_stable)
        
    ## find root of function
    x0 = 1
    M = my_root_finder(func_M, x0)
    M = np.float64(M)
    return M, min_pfeas


def cumulative_exp_integral_feas(beta, size, info_instance, info_type, problem_type, E_LB, n_feas, n_chunks = 10_000, n_samples = 10_000):
    ''' Computes the exponential-cumulative feasible term (ie approximates the Gibbs integral)'''
    eners = sample_uniform_feasible(size, info_instance, info_type, problem_type, n_samples)
    delta = (np.max(eners) - E_LB) / n_chunks
    cumul_exp_feas = np.sum([ np.exp(- np.float128(beta) * (E_LB + (j+1)*delta) ) * np.sum( np.logical_and(eners > E_LB + j*delta, eners <= E_LB + (j+1)*delta) ) for j in np.arange(n_chunks) ]) / len(eners) * n_feas
    return cumul_exp_feas


def cumulative_exp_integral_feas_log(beta, size,  info_instance, info_type, problem_type, E_LB, n_feas, n_chunks = 10_000, n_samples = 10_000):
    ''' Computes the exponential-cumulative feasible term (ie approximates the Gibbs integral) in the numerically stable log space'''
    eners = sample_uniform_feasible(size, info_instance, info_type, problem_type, n_samples)

    delta = (np.max(eners) - E_LB) / n_chunks
    n_occupations = np.array([  np.sum( np.logical_and(eners > E_LB + j*delta, eners <= E_LB + (j+1)*delta) ) for j in np.arange(n_chunks)]) / len(eners) * n_feas
    log_n_occ = logsafe(n_occupations)
    arguments = np.array([ ( log_n_occ[j] - np.float128(beta)*(E_LB + (j+1)*delta) ) for j in np.arange(n_chunks)])
    ret = LSE(arguments)
    return ret

def f_min_pfeas_triviality_feas(n_viols, integral_feas, beta, E_LB, log_stable):
    ''' Lower bound of eta for root existence. Below this, no solution => any M is good'''
    if log_stable:
        log_n_viol = logsafe(n_viols)
        arguments = np.array([ log_n_viol[j] for j in np.arange(len(n_viols))])  -  beta*E_LB
        log_inf = LSE(arguments)
        log_diff = log_inf - integral_feas
        if log_diff > 709: # in this case, np.exp(log_diff) would return an overflow error, but the probablity to be returned should be 0 anyway
            return 0
        return 1/(1 + np.exp( log_diff ))
    else:
        return integral_feas / (integral_feas + np.exp(-beta*E_LB)*np.sum(n_viols))


def build_func_for_root(n_viols, beta, E_LB, p_feas_term, log_stable):
    ''' Returns the function whose 0 encodes the solution (target M). Different functions are used, in linear and in log space for numerically stable results. '''
    if log_stable:
        def func_M(M):
            ''' Implements H function.  H(beta, M) =  .... replaces g function in log space algebraically '''
            M = np.float128(M)
            log_n_viols = logsafe(n_viols)
            arguments = np.array([ (log_n_viols[j] - beta*(E_LB + M*j)) for j in np.arange(1, len(n_viols))])
            return LSE(arguments) + p_feas_term    
    else:
        def func_M(M):
            ''' Implements g function.  g(beta, M) =  sum_v [ exp(-beta*(E_LB + Mv)) * p_pen(v) ]  +  p_pen(0)*[ exp(-beta*E_f)*(1 - cumul(E_f)) - (1-eta)/eta * cumul_exp_Ef ] '''
            M = np.float128(M)
            exp_vec = np.array([ np.exp(-beta*(E_LB + M*j)) for j in np.arange(1, len(n_viols)) ], dtype = np.float128)
            return np.dot(exp_vec.flatten(), n_viols[1:]) + np.float128(p_feas_term)
    return func_M










######### Optimality strategy


### Main function
def M_method_opt(size, problem_type, info_instance, info_type, beta, peak_max, min_pfeas, E_f, E_LB, log_stable = False):
    ''' Main algorithm for the optimality strategy. Returns penalty weigth M (and guaranteed success prob) such that QUBO reformulation with M, sampled by a Gibbs sampler at temperature beta, guarantees a certain success probability'''
    ## evaluate violation peaks and cumulatives
    n_viols = n_violations_problem(problem_type, size, peak_max)

    if log_stable:
        cumul_exp_good_log, cumul_exp_tail_log = cumulatives_exp_integral_log(E_f, beta, size, info_instance, info_type, problem_type, E_LB, n_viols[0])
        integral_good, integral_tail = cumul_exp_good_log, cumul_exp_tail_log
    else:
        cumul_exp_good, cumul_exp_tail = cumulatives_exp_integral(E_f, beta, size, info_instance, info_type, problem_type, E_LB, n_viols[0])
        integral_good, integral_tail = cumul_exp_good, cumul_exp_tail

    ## Ensure poly is correctly built
    # check if poly(1)>0
    min_pfeas_triviality = f_min_pfeas_triviality(n_viols[1:], integral_good, integral_tail, beta, E_LB, log_stable)
    if min_pfeas <= min_pfeas_triviality:
        print(f"With tolerance {min_pfeas}, the sampling probability requirement is satisfied for any M. Setting M=0.")
        return 0, min_pfeas
    # ensures poly(0)<0 and thus solution existence (decrease desidered probability if it is unattainable)
    max_pfeas_existence = f_min_pfeas_existence(integral_good, integral_tail, log_stable)
    if min_pfeas >= max_pfeas_existence:
        min_pfeas = decrease_requirement(min_pfeas, max_pfeas_existence)

    ## build function 
    if log_stable:
        p_good_term = - np.log(1/min_pfeas - 1) - integral_good
        func_M = build_func_for_root_opt_log(n_viols, beta, E_LB, p_good_term, integral_tail)
    else:
        p_feas_term = integral_tail - (1/min_pfeas - 1)*integral_good
        func_M = build_func_for_root(n_viols, beta, E_LB, p_feas_term, log_stable)
    
    ## find root of function
    x0 = np.max((1, E_f))
    M = my_root_finder(func_M, x0)
    M = np.float64(M)
    return M, min_pfeas


def build_func_for_root_opt_log(n_viols, beta, E_LB, p_good_term, integral_tail):
    ''' Returns the function whose 0 encodes the solution (target M), in log space for numerically stable results. '''
    def func_M(M):
        ''' Implements H function.  H(beta, M) =  .... replaces g function in log space algebraically '''
        M = np.float128(M)
        log_n_viols = logsafe(n_viols)
        arguments = np.array([ (log_n_viols[j] - beta*(E_LB + M*j)) for j in np.arange(1, len(n_viols))])
        p_inf_term = LSE(arguments)
        return LSE([p_inf_term, integral_tail]) + p_good_term
    return func_M


def cumulatives_exp_integral(E_f, beta, size, info_instance, info_type, problem_type, E_LB, n_feas, n_chunks = 10_000, n_samples = 10_000):
    ''' Computes the exponential-cumulative until E_f (ie approximates the Gibbs integral) and the tail (ie 1 - cumulative)'''
    eners = sample_uniform_feasible(size, info_instance, info_type, problem_type, n_samples)
    delta_avg = (np.max(eners) - E_LB) / (2*n_chunks) # this is needed to set the number of integration steps for tail and head, so that they're integers, all edge cases are taken care of and \delta should be similiar in head and tail 
    # distinguish case E_f \ge max(En)
    if E_f > np.max(eners):
        E_f = np.max(eners) + 1e-3 # adding + 1e-3 is useful to avoid rounding errors in the last interval counting [E_f - delta_dead, E_f]
        cumul_exp_tail = 0
    else:
        # calculate integral on tail
        n_chunks_tail = np.ceil((np.max(eners) - E_f) / delta_avg)
        delta_tail = (np.max(eners) - E_f) / n_chunks_tail
        cumul_exp_tail = np.sum([ np.exp(- np.float128(beta) * (E_f + j*delta_tail) ) * np.sum( np.logical_and(eners > E_f + j*delta_tail, eners <= E_f + (j+1)*delta_tail) ) for j in np.arange(n_chunks_tail) ]) / len(eners) * n_feas
    # calculate integral on head
    n_chunks_head = np.ceil((E_f - E_LB) / delta_avg)
    delta_head = (E_f - E_LB) / n_chunks_head
    cumul_exp_good = np.sum([ np.exp(- np.float128(beta) * (E_LB + (j+1)*delta_head) ) * np.sum( np.logical_and(eners > E_LB + j*delta_head, eners <= E_LB + (j+1)*delta_head) ) for j in np.arange(n_chunks_head) ]) / len(eners) * n_feas
    return cumul_exp_good, cumul_exp_tail


def cumulatives_exp_integral_log(E_f, beta, size, info_instance, info_type, problem_type, E_LB, n_feas, n_chunks = 10_000, n_samples = 10_000):
    ''' Computes the exponential-cumulative until E_f (ie approximates the Gibbs integral) and the tail (ie 1 - cumulative)'''
    eners = sample_uniform_feasible(size, info_instance, info_type, problem_type, n_samples)

    delta_avg = (np.max(eners) - E_LB) / (2*n_chunks) # this is needed to set the number of integration steps for tail and head, so that they're integers, all edge cases are taken care of and \delta should be similiar in head and tail 
    
    # distinguish case E_f \ge max(En)
    if E_f > np.max(eners):
        E_f = np.max(eners) + 1e-3 # adding + 1e-3 is useful to avoid rounding errors in the last interval counting [E_f - delta_head, E_f]
        cumul_exp_tail_log = -np.inf # in the logspace, if we have log(0) we just set it to -inf, it will be taken care by LSE
    else:
        # calculate integral on tail
        n_chunks_tail = int(np.ceil((np.max(eners) - E_f) / delta_avg))
        delta_tail = (np.max(eners) - E_f) / n_chunks_tail
        n_occupations = np.array([  np.sum( np.logical_and(eners > E_f + j*delta_tail, eners <= E_f + (j+1)*delta_tail) ) for j in np.arange(n_chunks_tail)]) / len(eners) * n_feas
        log_n_occ = logsafe(n_occupations)
        arguments = np.array([ ( log_n_occ[j] - np.float128(beta)*(E_f + j*delta_tail) ) for j in np.arange(n_chunks_tail)])
        cumul_exp_tail_log = LSE(arguments)

    # calculate integral on head
    n_chunks_head = int(np.ceil((E_f - E_LB) / delta_avg))
    delta_head = (E_f - E_LB) / n_chunks_head
    n_occupations = np.array([  np.sum( np.logical_and(eners > E_LB + j*delta_head, eners <= E_LB + (j+1)*delta_head) ) for j in np.arange(n_chunks_head)]) / len(eners) * n_feas
    log_n_occ = logsafe(n_occupations)
    arguments = np.array([ ( log_n_occ[j] - np.float128(beta)*(E_LB + (j+1)*delta_head) ) for j in np.arange(n_chunks_head) ])
    cumul_exp_good_log = LSE(arguments)
    return cumul_exp_good_log, cumul_exp_tail_log



def f_min_pfeas_triviality(n_violations, integral_good, integral_tail, beta, E_LB, log_stable):
    ''' Lower bound of eta for poly root existence. Below this, no solution -> any M is good'''
    if log_stable:
        log_n_viol = logsafe(n_violations)
        arguments = np.array([ log_n_viol[j] for j in np.arange(len(n_violations))])  -  beta*E_LB
        log_inf = LSE(arguments)
        log_inf_and_tail = LSE([log_inf, integral_tail])
        log_diff = log_inf_and_tail - integral_good
        return 1/(1 + np.exp( log_diff ))
    else:
        return integral_good / (integral_good + integral_tail + np.exp(-beta*E_LB)*np.sum(n_violations))

def f_min_pfeas_existence(integral_good, integral_tail, log_stable):
    ''' Upper bound of eta for poly root existence. Above this, no solution -> no M can be good'''
    if log_stable:
        log_diff = integral_tail - integral_good
        return 1/(1 + np.exp( log_diff ))
    else:
        return integral_good / (integral_good + integral_tail) 
    
def decrease_requirement(min_pfeas, max_pfeas_existence):
    epsilon = 1e-2
    if max_pfeas_existence < epsilon:
        raise ValueError("Modified eta as \t\t[eta_new = eta_max - epsilon] is negative")
    if np.isclose(max_pfeas_existence, 0):
        raise ValueError(f"Maximum eta possible is too small, evaluates to {max_pfeas_existence}.\nYou should increase E_f to allow more random samples in [E_LB, E_f] to evaluate the integral. Atm E_LB = {E_LB} and E_f = {E_f}")
    print(f"With tolerance {min_pfeas}, poly root doesn't exist. Feasibility tolerance decreased to {max_pfeas_existence - epsilon}")
    min_pfeas = max_pfeas_existence - epsilon
    return min_pfeas


####### LCBO speficic functions

### Random feasible Sampler functions

## NPP

def RandomSampler_Feasible_NPP_wseed(N, P, seed, n_sample):
    numbs = np.array(build_numbs_set(N, P, seed))
    Es = np.ndarray((n_sample))
    for j in range(n_sample):
        assign = np.random.randint(P, size = N)
        subset_sums = [ np.sum(numbs[assign == p]) for p in range(P)]
        Es[j] = P*np.var(subset_sums) # NB the objective energy can be reformulated as the variance of the subsets' sums (scaled by P)
    return Es

def build_numbs_set(N, P, seed):
    random.seed(seed)
    numbs = [int(1000*random.uniform(0.0, 1.0)) for _ in range(N)]
    total = sum(numbs)
    if numbs[-1] - (total % P) > 1:
        numbs[-1] -= (total % P)
    else:
        numbs[-1] += (P - (total % P))
    total = sum(numbs)
    return numbs

def RandomSampler_Feasible_NPP_wnumbersset(N, P, numbs, n_sample):
    Es = np.ndarray((n_sample))
    for j in range(n_sample):
        assign = np.random.randint(P, size = N)
        subset_sums = [ np.sum(numbs[assign == p]) for p in range(P)]
        Es[j] = P*np.var(subset_sums) # NB the objective energy can be reformulated as the variance of the subsets' sums (scaled by P)
    return Es

## TSP

def RandomSampler_Feasible_TSP_wseed(Nc, seed, n_sample):
    adj = build_adjacency(Nc, seed)
    Es = np.ndarray((n_sample))
    for j in range(n_sample):
        permu = np.random.permutation(np.arange(Nc))
        Es[j] = cost_permutation(permu, adj)
    return Es

def build_adjacency(Nc, seed, circle_flag = False):
    if circle_flag:
        # trivial construction of adj. matrix of graph with N cities on a circle. Modify this function to get more complex structures
        coordinates = [(1_000_000 * np.cos((index / Nc) * 2 * np.pi), 1_000_000 * np.sin((index / Nc) * 2 * np.pi)) for index in range(Nc)]
        distance_table = [[int(np.sqrt((coordinates[c_1][0] - coordinates[c_0][0]) ** 2 + (coordinates[c_1][1] - coordinates[c_0][1]) ** 2)) for c_0 in range(Nc)] for c_1 in range(Nc)]
    else:
        # random construction of the adjacency matrix, based on euclidian distance. (cities are uniformly randomly placed inside a square)
        np.random.seed(seed)
        coordinates = 1e6 * np.array( [np.random.uniform(low = -1, high = 1, size = 2) for i in range(Nc)] )
        distance_table = [[int(np.sqrt((coordinates[c_1][0] - coordinates[c_0][0]) ** 2 + (coordinates[c_1][1] - coordinates[c_0][1]) ** 2)) for c_0 in range(Nc)] for c_1 in range(Nc)]
    return np.array(distance_table)

def cost_permutation(perm, adj):
    Nc = len(perm)
    cost = 0
    for i in range(Nc):
        cost += adj[perm[i], perm[(i+1)%Nc]]
    return cost

def RandomSampler_Feasible_TSP_wadjacency(Nc, adj, n_sample):
    Es = np.ndarray((n_sample))
    for j in range(n_sample):
        permu = np.random.permutation(np.arange(Nc))
        Es[j] = cost_permutation(permu, adj)
    return Es

def RandomSampler_Feasible_TSP_circle(Nc, n_sample):
    adj = build_adjacency(Nc, None, circle_flag = True)
    Es = np.ndarray((n_sample))
    for j in range(n_sample):
        permu = np.random.permutation(np.arange(Nc))
        Es[j] = cost_permutation(permu, adj)
    return Es

## PO

def get_PO_obj(n_bits, info_instance):
    directory, vseed = info_instance
    filename = f"../data/{directory}/{n_bits}/random{vseed}_{n_bits}.lp"
    m = ModelReader.read(filename, ignore_names=True)
    qp = from_docplex_mp(m)
    obj = qp.objective
    Q = obj.quadratic.to_array()
    L = obj.linear.to_array()
    return Q, L

def random_portfolio_bin(N, w, n_samples):
    ''' Uniformly sample feasible portfolios using the stars and bars method'''
    if N == 1:
        return np.ones((w, n_samples))
    M = 2**w-1 + N - 1 # number of objects in the stars and bars method
    ports = np.empty((n_samples, N), dtype=int)
    for i in range(n_samples):
        divs = np.sort(np.random.choice(M, size=N-1, replace=False))
        prev = -1
        counts = []
        for d in divs:
            counts.append(d - prev - 1)
            prev = d
        counts.append(M - 1 - prev)
        ports[i] = counts
    port_bin = np.array([[[int(bit) for bit in format(stock, f'0{w}b')] for stock in port] for port in ports])
    port_bin = np.reshape(port_bin, (n_samples, N*w))
    return port_bin

def RandomSampler_Feasible_PO_wseed(N, w, info_instance, n_sample):
    Q, L = get_PO_obj(N*w, info_instance)
    Q += np.diag(L)
    Es = np.ndarray((n_sample))
    ports = random_portfolio_bin(N, w, n_sample)
    for j in range(n_sample):
        x = ports[j]
        Es[j] = np.dot(x, np.dot(Q, x))
    return Es





### Penalization densities


def n_viols_exact_NPP(N, P, v_max):
    ''' Returns, in order, the number of bitstring of a given violation family '''
    if v_max >= 8:
        raise ValueError(f"Peaks violations not implemented for NPP for v greater than {v_max}")
    p = np.ndarray((v_max + 1), dtype = np.float128)
    for v in range(v_max + 1):
        match v:
            case 0:
                p[v] = np.float128(P)**N # feasible solutions
            case i if 1 <= i < 4:
                p[v] = np.float128(P)**(N-v) * comb(N, v) * (1 + comb(P, 2))**v
            case i if 4 <= i < 8:
                p[v] = np.float128(P)**(N-v) * comb(N, v) * (1 + comb(P, 2))**v  +  np.float128(v-3) * comb(N, v-3) * np.float128(P)**(N - v + 3) * comb(P, 3) * (1 + comb(P, 2))**(v-4)
    return p

def n_viols_exact_TSP(Nc, v_max):
    ''' Returns, in order, the number of bitstring of a given violation family'''
    if v_max >= 8:
        raise ValueError(f"Peaks violations not implemented for TSP for v greater than {v_max}")
    p = np.zeros((v_max+1), dtype=np.longdouble)
    for i in range(v_max + 1):
        match i:
            case 0:
                p[i] = np.longdouble(factorial(Nc)) # feasible solutions
            case 2:
                p[i] = np.longdouble(factorial(Nc)) * (Nc + 4*comb(Nc,2) + 3/2*comb(Nc,3))
            case 4:
                p[i] = np.longdouble(factorial(Nc)) * (comb(Nc,2) + 21*comb(Nc,3) + 57*comb(Nc,4) + 45*comb(Nc,5) + 45/4*comb(Nc,6))
            case 6:
                p[i] = np.longdouble(factorial(Nc)) * 5*(47/15*comb(Nc,3) + 24*comb(Nc,4) + 137*comb(Nc,5) + 1157*comb(Nc,6) + 567/4*comb(Nc,7) + 126*comb(Nc,8) + 63/2*comb(Nc,9))
    return p


def n_viols_exact_PO(N, w, v_max):
    ''' Returns, in order, the number of bitstring of a given violation family'''
    if v_max >= 36:
        raise ValueError(f"Peaks violations not implemented for PO for v greater than {v_max}")
    p = np.zeros((v_max + 1))
    for i in range(v_max + 1):
        match i:
            case 0:
                p[i] = comb(2**w+N-2, N-1)  # feasible solutions
            case 1:
                p[i] = comb(2**w+N-1, N-1) + comb(2**w+N-3, N-1) - N
            case 4:
                p[i] = comb(2**w+N, N-1) + comb(2**w+N-4, N-1) - N**2
            case 9:
                p[i] = comb(2**w+N+1, N-1) + comb(2**w+N-5, N-1) - (N+1)/2 * N**2 
            case 16:
                p[i] = comb(2**w+N+2, N-1) + comb(2**w+N-6, N-1) - (N+1)/2 * N**2 * (N+2)/3
            case 25:
                p[i] = comb(2**w+N+3, N-1) + comb(2**w+N-7, N-1) - (N+1)/2 * N**2 * (N+2)/3 * (N+3)/4
    return p