import random
import json
import os
import numpy as np
from dadk.BinPol import *
from dadk.QUBOSolverDAv2 import *
from dadk.QUBOSolverBase import KEYWORD_HIDE_SAMPLING_INFO


for case in range(1000, 1000 + 128):
    filename = f'case_{case}.json'

    if os.path.isfile(filename):
        continue
    
    random.seed(42+case)
    
    P = random.randint(4, 32)
    N = random.randint(4*P, int(math.floor(4096/P)))
    
    print(P, N)
    
    values = [random.randint(100, 300) for _ in range(N)]
    
    print(values)
    
    BinPol.freeze_var_shape_set(VarShapeSet(BitArrayShape(name='x', shape=(N, P))))
    sums = {p: BinPol.sum(Term(values[i], ('x', i, p)) for i in range(N)) for p in range(P)}
    H_objective = 1000 * BinPol.sum((sums[p] - sum(values) / P) ** 2 for p in range(P))
    penalties = {i: (1 - BinPol.sum(Term(1, ('x', i, p)) for p in range(P))) ** 2 for i in range(N)}
    H_penalty = BinPol.sum(penalties[i] for i in range(N))
    
    Ms = {}
    
    bits = N*P    
    
    for probability in [0.5, 0.25, 0.75]:
        
        M_low = max( [0] + [ M for M in Ms if Ms[M]['rate'] < probability ]  )
        M_high = min( [2_000_000_000] + [ M for M in Ms if Ms[M]['rate'] > probability ]  ) 
        
        while True:
        
            M = int(0.5*(M_low + M_high))
            
            H = H_objective + M * H_penalty
            H.user_data[KEYWORD_HIDE_SAMPLING_INFO] = True
            
            valid = 0
            invalid = 0

            temperature_start_list = []
            temperature_end_list = []
            temperature_interval_list = []
            offset_increase_rate_list = []
            
            for i in range(8):
                solver = QUBOSolverDAv2(
                    number_iterations=bits**2,
                    temperature_interval=P,
                    number_runs=128,
                    random_seed=42+i)
                
                solution_list = solver.minimize(H)

                temperature_start_list.append(solution_list.da_parameter.temperature_start)
                temperature_end_list.append(solution_list.da_parameter.temperature_end)
                temperature_interval_list.append(solution_list.da_parameter.temperature_interval)
                offset_increase_rate_list.append(solution_list.da_parameter.offset_increase_rate)

                for solution in solution_list:
                    if H_penalty.compute(solution.configuration) == 0:
                        valid += 1
                    else:
                        invalid += 1
    
            rate = valid / (valid + invalid)
    
            print(probability, M, rate )
            Ms[M] = {'rate':rate, 
                     'temperature_start':np.mean(temperature_start_list),
                     'temperature_end':np.mean(temperature_end_list), 
                     'temperature_interval':np.mean(temperature_interval_list), 
                     'offset_increase_rate':np.mean(offset_increase_rate_list)}
            
            if rate > probability:
                M_high = M
            else:
                M_low = M
        
            if M_high - M_low < 100:
                break

    data = {
        'P':P,
        'N':N,
        'values':values,
        'measurements':{
            M:Ms[M] for M in sorted(Ms.keys())
        }
    }
    
    with open(filename, 'w') as fp:
        json.dump(data, fp, indent=2)
    

