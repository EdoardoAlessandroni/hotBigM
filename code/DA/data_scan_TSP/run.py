from dadk.BinPol import *
from dadk.QUBOSolverDAv2 import QUBOSolverDAv2

import json

number_of_seeds = 4

Ns = [64]
Ms = sorted(list(set( list(range(25_000, 2_000_001, 25_000)) + list(range(400_000, 600_001, 5_000)) )))

print('Ns', Ns)
print('Ms', Ms)

for N in Ns:

    coordinates = [(1_000_000 * math.cos((index / N) * 2 * math.pi), 1_000_000 * math.sin((index / N) * 2 * math.pi)) for index in range(N)]
    distance_table = [[int(math.sqrt((coordinates[c_1][0] - coordinates[c_0][0]) ** 2 + (coordinates[c_1][1] - coordinates[c_0][1]) ** 2)) for c_0 in range(N)] for c_1 in range(N)]

    var_shape_set = VarShapeSet(BitArrayShape('x', (N, N), one_hot=OneHot.two_way))

    objective = BinPol(var_shape_set)
    for t in range(N):
        for c_0 in range(N):
            for c_1 in range(N):
                if c_0 != c_1:
                    objective.add_term(distance_table[c_0][c_1],
                                       ('x', t, c_0),
                                       ('x', (t + 1) % N, c_1))

    penalty = BinPol(var_shape_set)
    for t in range(N):
        penalty.add_exactly_1_bit_on(bits=[('x', t, c) for c in range(N)])
    for c in range(N):
        penalty.add_exactly_1_bit_on(bits=[('x', t, c) for t in range(N)])

    for seed in [42]:

        summary = {}

        filename = f'results-N={N}.json'

        if os.path.isfile(filename):
            with open(filename, 'r') as fp:
                summary = json.load(fp)
                summary = {int(M): {seed: summary[M][seed] for seed in summary[M]} for M in summary}

        for M in Ms:

            if M not in summary:
                summary[M] = {}

            for da_seed in range(42, 42 + number_of_seeds):

                seed_id = f'{seed}-{da_seed}'

                if seed_id in summary[M]:
                    continue

                solver = QUBOSolverDAv2(
                    number_iterations=(N * N) ** 2,
                    number_runs=128,
                    temperature_mode=0,
                    temperature_interval=N,
                    use_access_profile=True,
                    random_seed=da_seed,
                    auto_tuning=AutoTuning.SAMPLING,
                    probability_model=ProbabilityModel.EXPECTATION_VALUE,
                    flip_probabilities=[0.99, 0.01],
                    annealing_steps=[0.00, 0.50],
                    sampling_runs=96)

                qubo = objective + M * penalty
                qubo.user_data[KEYWORD_HIDE_SAMPLING_INFO] = True

                solution_list = solver.minimize(qubo)

                summary[M][seed_id + '_data'] = {
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

                print(N, M, seed_id, len(summary[M][seed_id]), 'done')

                with open(filename, 'w') as fp:
                    json.dump(summary, fp)
