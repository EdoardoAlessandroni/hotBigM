#!/usr/bin/env python3.8
import os
import time

def SendJob(N_idx, vseed, M_strat, temp_scale, eta_req):
    """Submits a Slurm job with specified parameters."""
    time.sleep(.1)
    os.system(f"python ./SA_simulations.py {problem_type} {N_idx} {vseed} {M_strat} {temp_scale} {eta_req}\n\n")
    return 0

# Iterate over parameters and submit jobs

problem_type = "PO"                                                                                                          
N_idxs= [3]  # for NPP [0,..,5], for TSP [0,...,3], for PO [0,...,6]
vseeds = range(42, 343, 100) # between 42 and 142 for NPP and TSP_rand, between 42 and 9942 (with jumps of 100) for PO, only 42 for TSP_circle
M_strategies = ["feasibility", "optimality"]
temperature_scalers = [1, 10, 100]
etas_req = [.25, .5, .75]

for N_idx in N_idxs:
    for vseed in vseeds:
        for M_strat in M_strategies:
            for temp_scale in temperature_scalers:
                for eta_req in etas_req:
                    SendJob(N_idx, vseed, M_strat, temp_scale, eta_req)