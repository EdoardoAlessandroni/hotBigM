#!/usr/bin/env python3.8
import os
import time

def SendJob(N_idx, vseed, temp_scale, eta_req):
    """Submits a Slurm job with specified parameters."""
    time.sleep(.1)
    os.system(f"python ./Ms_simulations.py {problem_type} {N_idx} {vseed} {temp_scale} {eta_req}\n\n")
    return 0

# Iterate over parameters and submit jobs

problem_type = "PO"                                                                                                          
N_idxs= [8]  # for NPP [0,..,5], for TSP [0,...,3], for PO [0,...,6] for Gibbs
                       # for NPP [0,..,12], for TSP (circle and rand) [0,...,7], for PO [0,...,8] for SA
                    # for NPP [0,..,7], for TSP (circle and rand) [0,...,15], for PO [0,...,8] for Ms only
# let us do 10 instances per system size
vseeds = range(42, 943, 100) # between 42 and 142 for NPP and TSP_rand, between 42 and 9942 (with jumps of 100) for PO, only 42 for TSP_circle
temperature_scalers = [1000]
etas_req = [.25, .5, .75]

for N_idx in N_idxs:
    for vseed in vseeds:
        for temp_scale in temperature_scalers:
            for eta_req in etas_req:
                SendJob(N_idx, vseed, temp_scale, eta_req)