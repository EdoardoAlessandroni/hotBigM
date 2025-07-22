import matplotlib.pyplot as plt
import random
from IPython.display import display, HTML
from ipywidgets import interact

from dadk.BinPol import *
from dadk.QUBOSolverDAv2 import QUBOSolverDAv2

from tabulate import tabulate
import json

steps = 32

number_of_seeds = 4

for N,P in [(n*8,n) for n in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]:

    for seed in range(42, 42+number_of_seeds):

        random.seed(seed)
        data = [int(1000*random.uniform(0.0, 1.0)) for _ in range(N)]
        total = sum(data)
        if data[-1] - (total % P) > 1:
            data[-1] -= (total % P)
        else:
            data[-1] += (P - (total % P))
        total = sum(data)

        with open(f'values_{N}_{seed}.json', 'w') as fp:
            json.dump(data, fp)
        
        BinPol.freeze_var_shape_set(VarShapeSet(BitArrayShape(name='x', shape=(N, P), axis_names=['Item', 'Person'])))
        sums = {p: BinPol.sum(Term(data[i], ('x', i, p)) for i in range(N)) for p in range(P)}
        penalties = {i: (1 - BinPol.sum(Term(1, ('x', i, p)) for p in range(P))) ** 2 for i in range(N)}
        objective = BinPol.sum((sums[p] - sum(data) / P) ** 2 for p in range(P))
        penalty = BinPol.sum(penalties[i] for i in range(N))
        
        summary = {}

        filename = f'results-N={N}_P={P}.json'
        
        if os.path.isfile(filename):
            with open(filename, 'r') as fp:
                summary = json.load(fp)    
                summary = {int(M):{seed:summary[M][seed] for seed in summary[M]} for M in summary}

        for M in [int(50_000 + 1_000 * i) for i in range(50)] + [int(100_000 * 2 ** (i / steps)) for i in range(steps+16+1)]:

            if M not in summary:
                summary[M] = {}

            for da_seed in range(42, 42+number_of_seeds):

                seed_id = f'{seed}-{da_seed}'
                
                if seed_id in summary[M]:
                    # print(N, P, M, seed_id, len(summary[M][seed_id]), 'found')
                    continue
    
                solver = QUBOSolverDAv2(
                    number_iterations=(N*P)**2,          # Total number of iterations per run.
                    number_runs=128,                     # Number of stochastically independent runs.
                    temperature_mode=0,                  # 0, 1, or 2 to define the cooling curve:
                                                         #    0, 'EXPONENTIAL':
                                                         #       reduce temperature by factor (1-temperature_decay) every temperature_interval steps
                                                         #    1, 'INVERSE':
                                                         #       reduce temperature by factor (1-temperature_decay*temperature) every temperature_interval steps
                                                         #    2, 'INVERSE_ROOT':
                                                         #       reduce temperature by factor (1-temperature_decay*temperature^2) every temperature_interval steps
                    temperature_interval=N,              # Number of iterations keeping temperature constant.
    
                    use_access_profile=True,
                    random_seed=da_seed,
    
                    auto_tuning=AutoTuning.SAMPLING, # Following methods for scaling ``qubo`` and determining temperatures are available:
                                                         #    AutoTuning.NOTHING:
                                                         #       no action
                                                         #    AutoTuning.SCALING:
                                                         #       ``scaling_factor`` is multiplied to ``qubo``, ``temperature_start``, ``temperature_end`` and ``offset_increase_rate``.
                                                         #    AutoTuning.AUTO_SCALING:
                                                         #       A maximum scaling factor w.r.t. ``scaling_bit_precision`` is multiplied to ``qubo``, ``temperature_start``, ``temperature_end`` and ``offset_increase_rate``.
                                                         #    AutoTuning.SAMPLING:
                                                         #       ``temperature_start``, ``temperature_end`` and ``offset_increase_rate`` are automatically determined.
                                                         #    AutoTuning.AUTO_SCALING_AND_SAMPLING:
                                                         #       Temperatures and scaling factor are automatically determined and applied.


                    probability_model=ProbabilityModel.EXPECTATION_VALUE,
                    flip_probabilities=[0.99, 0.01],
                    annealing_steps=[0.00, 0.50],
                    sampling_runs=96, # instead of 100, 8 threads each with 12 samples
                )
    
                qubo = objective + M * penalty            
                qubo.user_data[KEYWORD_HIDE_SAMPLING_INFO] = True
    
                solution_list = solver.minimize(qubo)

                """
                            number_runs: int = 16,  # Number of runs
            : int = 1000000,  # Number of iterations
            : float = 1000.0,  # Temperature start
            : float = 1.0,  # Temperature end
            : int = 0,  # Temperature mode
            temperature_interval: int = 100,  # Temperature interval
            offset_increase_rate
            """

                summary[M][seed_id+'_data'] = {
                'number_runs': solution_list.da_parameter.number_runs,
                'number_iterations': solution_list.da_parameter.number_iterations,
                'temperature_start': solution_list.da_parameter.temperature_start,
                'temperature_end': solution_list.da_parameter.temperature_end,
                'temperature_mode': solution_list.da_parameter.temperature_mode,
                'temperature_interval': solution_list.da_parameter.temperature_interval,
                'offset_increase_rate': solution_list.da_parameter.offset_increase_rate,
                }
    
                summary[M][seed_id] = []
                for solution in solution_list.solutions:
                    e_objective = int(objective.compute(solution.configuration))
                    valid = int(penalty.compute(solution.configuration)) == 0
                    for _ in range(solution.frequency):
                        summary[M][seed_id].append((e_objective, valid, solution.configuration))
    
                print(N, P, M, seed_id, len(summary[M][seed_id]), 'done')    
    
                with open(filename, 'w') as fp:
                    json.dump(summary, fp)
