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
    same as above, but the quantum circuit is a maQAOA instead of QAOA (layers \in {1, 2, 3} instead of {5, 10, 20}). version indicates, for optimality strategy, in select_Ef in the old one (v1, where it was 3e-2) or the new, current one (v2, where it is 1e-1)
    