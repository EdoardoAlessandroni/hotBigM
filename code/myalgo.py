import numpy as np
from scipy.special import factorial, comb
import random



### Main function


def M_method_opt(size, problem_type, info_instance, info_type, beta, peak_max, min_pfeas, E_f, E_LB = 0, circle_flag = False):
    ## evaluate violation peaks and cumulatives
    match problem_type:
        case "NPP":
            N, P = size
            p_viols = p_viols_exact_NPP(N, P, peak_max)
        case "TSP":
            Nc = size
            p_viols = p_viols_exact_TSP(Nc, peak_max)

    #cumul_exp, cumul_tail = cumulatives_exp_unique(E_f, beta, size, info_instance, info_type, problem_type, circle_flag = circle_flag)
    cumul_exp_good, cumul_exp_tail_loose, cumul_exp_tail = cumulatives_exp_integral(E_f, beta, size, info_instance, info_type, problem_type, E_LB, circle_flag = circle_flag)
    
    ## Ensure poly is correctly built
    # check if poly(1)>0
    min_pfeas_triviality = f_min_pfeas_triviality(p_viols[0], p_viols[1:], cumul_exp_good, cumul_exp_tail, beta, E_LB)
    if min_pfeas <= min_pfeas_triviality:
        print(f"With tolerance {min_pfeas}, the sampling probability requirement is satisfied for any M. Setting M=0.")
        return 0
    # ensures poly(0)<0 and thus solution existence (decrease desidered probability if it is unattainable)
    max_pfeas_existence = f_min_pfeas_existence(cumul_exp_good, cumul_exp_tail)
    if min_pfeas >= max_pfeas_existence:
        epsilon = 1e-2
        if max_pfeas_existence < epsilon:
            raise ValueError("Modified eta as \t\t[eta_new = eta_max - epsilon] is negative")
        if np.isclose(max_pfeas_existence, 0):
            raise ValueError(f"Maximum eta possible is too small, evaluates to {max_pfeas_existence}.\nYou should increase E_f to allow more random samples in [E_LB, E_f] to evaluate the integral. Atm E_LB = {E_LB} and E_f = {E_f}")
        print(f"With tolerance {min_pfeas}, poly root doesn't exist. Feasibility tolerance decreased to {max_pfeas_existence - epsilon}")
        min_pfeas = max_pfeas_existence - epsilon
    
    ## build and solve poly
    p_feas_term = -(1/min_pfeas - 1) * cumul_exp_good  +  cumul_exp_tail

    # poly_alpha = lambda alpha:  p_feas_term*p_viols[0] + np.sum([ np.exp(-beta*E_LB)* p_violation * alpha**(j+1) for j, p_violation in enumerate(p_viols[1:])])
    # print(f"Poly_a(1) = {poly_alpha(1)}\tPoly_a(0) = {poly_alpha(0)}")
    # alpha_star = sp.optimize.brentq(poly_alpha, a = 0, b = 1)
    # M = -np.log(alpha_star)/beta
    def func_M(M):
        ''' Implements g function.  g(beta, M) =  \sum_v [ exp(-beta*(E_LB + Mv)) * p_pen(v) ]  +  p_pen(0)*[ exp(-beta*E_f)*(1 - cumul(E_f)) - (1-eta)/eta * cumul_exp_Ef ] '''
        M = np.float128(M)
        exp_vec = np.array([ np.exp(-beta*(E_LB + M*j)) for j in np.arange(1, len(p_viols)) ], dtype = np.float128)
        return np.dot(exp_vec.flatten(), p_viols[1:]) + np.float128(p_feas_term)*p_viols[0]
    #print(f"Func_M(0) = {func_M(0)!s}\tFunc_M(1e9) = {func_M(1e9)!s}")
    x0 = np.max((1, E_f))
    M = my_root_finder(func_M, x0)
    #result = sp.optimize.root(func_M, x0 = 0)  # scipy root finder. However, float128 are not usable inside it
    #M = result.x[0]
    print(f"For this instance, sampling temperature = {np.format_float_scientific(1/beta, 3)} and required probability in [0, E_f = {E_f}] at least {np.round(min_pfeas, 3)}, the chosen M is {np.format_float_scientific(M, 2)} (last violation peak used is {peak_max})")
    
    return M, min_pfeas



### Random feasible Sampler functions


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

def RandomSampler_Feasible_NPP_wnumbers(N, P, numbs, n_sample):
    Es = np.ndarray((n_sample))
    for j in range(n_sample):
        assign = np.random.randint(P, size = N)
        subset_sums = [ np.sum(numbs[assign == p]) for p in range(P)]
        Es[j] = P*np.var(subset_sums) # NB the objective energy can be reformulated as the variance of the subsets' sums (scaled by P)
    return Es


def RandomSampler_Feasible_TSP_wseed(Nc, seed, n_sample, circle_flag):
    adj = build_adjacency(Nc, seed, circle_flag)
    Es = np.ndarray((n_sample))
    for j in range(n_sample):
        permu = np.random.permutation(np.arange(Nc))
        Es[j] = cost_permutation(permu, adj)
    return Es

def build_adjacency(Nc, seed, circle_flag):
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


### Auxiliary functions

def my_root_finder(f, x0):
    ''' Forces usage of np.longdouble to find root of (monotonically decreasing (f)'''
    x = np.float128(x0)
    tol = 1

    # find interval (multiply by 10)
    n_iter = 0
    if f(x) < 0:
        interval = [0, x]
    else:
        while f(x) > 0:
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
    root = (interval[1] + interval[0])/2

    #print(f"n_iters of my_root_finder = {n_iter1}")
    return root


def cumulatives_exp_integral(E_f, beta, size,  info_instance, info_type, problem_type, E_LB = 0, n_chunks = 10_000, n_samples = 10_000, circle_flag = False):
    ''' Computes the exponential-cumulative until E_f (ie approximates the Gibbs integral) and the tail (ie 1 - cumulative)'''
    match problem_type:
        case "NPP":
            N, P = size
            match info_type: 
                case "numbers":
                    eners = RandomSampler_Feasible_NPP_wnumbers(N, P, np.array(info_instance), n_samples)
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
                    eners = RandomSampler_Feasible_TSP_wseed(Nc, info_instance, n_samples, circle_flag)
                case _:
                    raise ValueError(f"Info type {info_type} is not amomg the implemented ones")
        case _:
            raise ValueError(f"Problem type {problem_type} is not amomg the implemented ones")
            
    delta = (E_f - E_LB) / n_chunks
    cumul_exp_good = np.sum([ np.exp(- np.float128(beta) * (E_LB + (j+1)*delta) ) * np.sum( np.logical_and(eners > E_LB + j*delta, eners <= E_LB + (j+1)*delta) ) / len(eners) for j in np.arange(n_chunks) ])
    cumul_exp_tail = np.sum(eners > E_f)/len(eners) * np.exp(-beta*E_f)
    delta = (np.max(eners) - E_f) / n_chunks
    cumul_exp_tail_integral = np.sum([ np.exp(- np.float128(beta) * (E_f + j*delta) ) * np.sum( np.logical_and(eners > E_f + j*delta, eners <= E_f + (j+1)*delta) ) / len(eners) for j in np.arange(n_chunks) ])

    return cumul_exp_good, cumul_exp_tail, cumul_exp_tail_integral

# def cumulatives_exp_unique(E_f, beta, size, seed, problem_type, n_samples = 10_000, circle_flag = False):
#     ''' Computes the feasible cumulative at E_f, multiplied by the exponential (loose approximation fo the Gibbs integral) and the tail (ie 1 - cumulative)'''
#     raise NotImplementedError
#     if problem_type == "NPP":
#         N, P = size
#         eners = RandomSampler_Feasible_NPP(N, P, seed, n_samples)
#     elif problem_type == "TSP":
#         Nc = size
#         eners = RandomSampler_Feasible_TSP(Nc, seed, n_samples, circle_flag)

#     crude_cumul = np.sum(eners <= E_f)/len(eners)
#     if crude_cumul == 0:
#         print("Empirical sampling cumulative at E_f evaluates to 0")
#     cumul_tail = np.sum(eners > E_f)/len(eners)
#     return crude_cumul*np.exp(-beta*E_f), cumul_tail


def f_min_pfeas_triviality(p_feas, p_violations, cumul_exp_good, cumul_exp_tail, beta, E_LB):
    ''' Lower bound of eta for poly root existence. Below this, no solution -> any M is good'''
    return cumul_exp_good / (cumul_exp_good + cumul_exp_tail + np.exp(-beta*E_LB)*np.sum(p_violations)/p_feas)

def f_min_pfeas_existence(cumul_exp_good, cumul_exp_tail):
    ''' Upper bound of eta for poly root existence. Above this, no solution -> no M can be good'''
    return cumul_exp_good / (cumul_exp_good + cumul_exp_tail) 


def p_viols_exact_NPP(N, P, v_max):
    ''' Returns, in order, the probability of sampling solutions of a given violation family, starting from v=1 '''
    p = np.ndarray((8), dtype = np.float128)
    p[0] = (P / np.float128(2)**P)**N # feasible solutions
    for v in np.arange(1, 4):
        p[v] = (P / np.float128(2)**P)**(N-v) * comb(N, v) * (1 + comb(P, 2))**v / float(2)**(v*P)
    for v in np.arange(4, 8):
        p[v] = (P /np.float128(2)**P)**(N-v) * comb(N, v) * (1 + comb(P, 2))**v / float(2)**(v*P)  +  np.float128(v-3) * comb(N, v-3) * (P / np.float128(2)**P)**(N - v + 3) * comb(P, 3) * (1 + comb(P, 2))**(v-4) / float(2)**((v-3)*P)
    return p[:v_max+1]

def p_viols_exact_TSP(Nc, v_max):
    ''' Returns, in order, the probability of sampling solutions of a given violation family'''
    p = np.ndarray((5), dtype=np.longdouble)
    p[0] = np.longdouble(factorial(Nc)) # feasible solutions
    p[1] = 0
    p[2] = np.longdouble(factorial(Nc)) * Nc * (2*Nc - 1 + (Nc-1)*(Nc-2)/4) 
    p[3] = 0
    p[4] = np.longdouble(factorial(Nc))**2/factorial(Nc-3)/2 * ( 1 + (Nc-1)*(Nc+5)/4 + 1/(Nc-2) + 2*(Nc-2) + (Nc-3)*(Nc-4)/2*( 1 + (Nc-5)/16 ) )
    return p[:v_max+1] / np.longdouble(2)**(Nc**2)