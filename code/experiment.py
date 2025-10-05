import os

def WriteNewSchedule(alpha, delta, a, b):
    job_filename = "schedule_imp.py"
    with open(job_filename, "w") as fh:
        fh.writelines("import numpy as np\n")
        fh.writelines("def infer_temperature(n_bits):\n")
        fh.writelines(f"\ta, b = {a}, {b}\n")
        fh.writelines("\tti = a/( np.log(n_bits) + b )\n")
        fh.writelines("\tgamma = -2.6\n")
        fh.writelines(f"\talpha, delta =  {alpha}, {delta}\n")
        fh.writelines("\ttf = alpha*( 1 + gamma/(np.log(n_bits)+delta) )\n")
        fh.writelines("\treturn ti, tf\n")
    return 0

# Iterate over parameters and submit jobs

aas = [7e4, 1e5]
bbs = [.7]
alphas = [1e3]
deltas = [.4]

for a in aas:
    for b in bbs:
        for alpha in alphas:
            for delta in deltas:
                print("\nParameters:\t", a, b, alpha, delta)
                WriteNewSchedule(alpha, delta, a, b)
                os.system(f"python ./SA_simulations.py PO 0 42 feasibility 1 .75\n\n")
