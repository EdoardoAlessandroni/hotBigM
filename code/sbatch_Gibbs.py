#!/usr/bin/env python3.8
import os
import time

def ReSendJob(N_idx, vseed, M_strat, temp_scale, eta_req):
    """Submits a Slurm job with specified parameters."""
    time.sleep(.1)

    job_filename = f"send_job_{problem_type}_{N_idx}_{vseed}_{M_strat}_{temp_scale}_{eta_req}.sh"
    
    with open(job_filename, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#\n")
        fh.writelines(f"#SBATCH --job-name={problem_type}.{N_idx}.{vseed}.{M_strat}.{temp_scale}.{eta_req}\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --ntasks-per-node=1\n")
        fh.writelines("#SBATCH --cpus-per-task=1\n")
        fh.writelines("#SBATCH --time=11:59:00\n")
        fh.writelines("#SBATCH --partition=regular1,regular2\n") # long1,long2   or   regular1,regular2
        fh.writelines("#SBATCH --exclusive\n")
        fh.writelines("#SBATCH --mem=10000mb\n")
        fh.writelines("#SBATCH --output=%x.o\n")
        fh.writelines("#SBATCH --error=%x.e\n")
        # fh.writelines("#     module load gnu8/8.3.0\n")
        # fh.writelines("#     module load cmake\n")
        # fh.writelines("#     module load openmpi3/3.1.4\n")
        # fh.writelines("#     module load python3\n")
        # fh.writelines("#     module load gsl\n")
        fh.writelines("source ~/mambaforge/etc/profile.d/conda.sh\n")
        fh.writelines("conda activate qubo_project\n")
        fh.writelines(f"python ~/hotBigM/code/Gibbs_simulations.py {problem_type} {N_idx} {vseed} {M_strat} {temp_scale} {eta_req} \n")  # Pass parameters
        fh.writelines("conda deactivate\n")

    os.system(f"sbatch {job_filename}")
    return 0

# Iterate over parameters and submit jobs

problem_type = "NPP"                                                                                                                                                                                                                      
N_idxs= [0,1,2,3,4,5] # for NPP [0,..,5], for TSP [0,1,2,3]
vseeds = range(42,52) # between 42 and 51 for NPP and TSP_rand, only 42 for TSP_circle
M_strategies = ["feasibility", "optimality"]
temperature_scalers = [1, 10, 100]
etas_req = [.25, .5, .75]

for N_idx in N_idxs:
    for vseed in vseeds:
        for M_strat in M_strategies:
            for temp_scale in temperature_scalers:
                for eta_req in etas_req:
                    ReSendJob(N_idx, vseed, M_strat, temp_scale, eta_req)