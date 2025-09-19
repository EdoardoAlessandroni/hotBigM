#!/usr/bin/env python3.8
import os
import time

def SendJob(N_idx, vseed, M_strat, temp_scale, eta_req):
    """Submits a Slurm job with specified parameters."""
    time.sleep(.1)
    os.system(f"python ./Gibbs_simulations.py {problem_type} {N_idx} {vseed} {M_strat} {temp_scale} {eta_req}\n\n")
    return 0

# Iterate over parameters and submit jobs

problem_type = "NPP"                                                                                                          
N_idxs= [0,1,2,3.4]  # for NPP [0,..,5], for TSP [0,...,3]
vseeds = range(22,142) # between 42 and 51 for NPP and TSP_rand, only 42 for TSP_circle
M_strategies = ["feasibility", "optimality"]
temperature_scalers = [1, 10, 100]
etas_req = [.25, .5, .75]

for N_idx in N_idxs:
    for vseed in vseeds:
        for M_strat in M_strategies:
            for temp_scale in temperature_scalers:
                for eta_req in etas_req:
                    SendJob(N_idx, vseed, M_strat, temp_scale, eta_req)