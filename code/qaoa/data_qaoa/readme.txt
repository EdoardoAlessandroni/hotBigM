\Old strategy
    Test 0:
    - Normalize temperature
    - Run 100 QAOA opt steps

    Test 1:
    - Normalize temperature
    - Renormalize Q(M)
    - Run 500 QAOA opt steps (breaking before if eps < tol)

    Test 2:
    - Renormalize Q(M)
    - Run 100 QAOA opt steps

    Actions:
    Normalize temperature = Guess a normalizing factor N (so far, spectral norm of Q_obj + Q_pen + const_obj + const_pen); 
                            given required layers p, compute (normalized) temperature from paper formula and compute the unnormalized temperature = T_norm * N;
                            work with our algo in this unnormalized setting until you have QUBO(M)_unnorm;
                            Normalize the QUBO dividing by N: now in this nromalized setting the normalized temperature is the right one (coming from p)
    Renormalize Q(M) =      Before feeding it to the QAOA normalize the QUBO by dividing it by np.max(np.abs(Q))



\{problem_type}_{M_strategy}
    runs on problem_type (\in {NPP, PO, TSP_rand}) applied to M_strategy (\in {feasibility, optimality}) goal of QUBO solving.

\PO_{M_strategy}_maqaoa_{version}
    same as above, but the quantum circuit is a maQAOA instead of QAOA (layers \in {1, 2, 3} instead of {5, 10, 20}).
    Vrsion indicates, for optimality strategy, in select_Ef in the old one ('old', where it was 3e-2) or the new, current one (just \PO_{M_strategy}_maqaoa, where it is 1e-1)
    


TO REFINE:

PO_optimality_maqaoa_percentile
Ef choosen as 75% percentile; ... p = 3 is pretty sad, mostly because it's a near-deterministic sampler (effective states are abuot 1)

PO_optimality_maqaoa_percentile_interp
PO_optimality_maqaoa_percentile_interp_L1 
comparison bertween M* and Ml1, with tol = 1e-4, steps = 500, M0=Ml1(temp)

PO_optimality_maqaoa_percentile_interp_M00
PO_optimality_maqaoa_percentile_interp_L1_M00
comparison bertween M* and Ml1, with tol = 1e-8, steps = 3000, M0=0