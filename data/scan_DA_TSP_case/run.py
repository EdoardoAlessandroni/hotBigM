from cases.att48 import att48
from cases.bayg29 import bayg29
from cases.bays29 import bays29
from cases.berlin52 import berlin52
from cases.br17 import br17
from cases.brazil58 import brazil58
from cases.burma14 import burma14
from cases.dantzig42 import dantzig42
from cases.fri26 import fri26
from cases.ft53 import ft53
from cases.ft70 import ft70
from cases.ftv33 import ftv33
from cases.ftv35 import ftv35
from cases.ftv38 import ftv38
from cases.ftv44 import ftv44
from cases.ftv47 import ftv47
from cases.ftv55 import ftv55
from cases.ftv64 import ftv64
from cases.ftv70 import ftv70
from cases.gr17 import gr17
from cases.gr21 import gr21
from cases.gr48 import gr48
from cases.hk48 import hk48
from cases.p43 import p43
from cases.pr76 import pr76
from cases.ry48p import ry48p
from cases.st70 import st70
from cases.swiss42 import swiss42
from cases.ulysses16 import ulysses16
from cases.ulysses22 import ulysses22

cases = [
    burma14,  # 14
    ulysses16,  # 16
    br17,  # 17
    gr17,  # 17
    gr21,  # 21
    ulysses22,  # 22
    fri26,  # 26
    bayg29,  # 29
    bays29,  # 29
    ftv33,  # 34
    ftv35,  # 36
    ftv38,  # 39
    dantzig42,  # 42
    swiss42,  # 42
    p43,  # 43
    ftv44,  # 45
    att48,  # 48
    ftv47,  # 48
    gr48,  # 48
    hk48,  # 48
    ry48p,  # 48
    berlin52,  # 52
    ft53,  # 53
    ftv55,  # 56
    brazil58,  # 58
    ftv64,  # 65
    ft70,  # 70
    st70,  # 70
    ftv70,  # 71
    pr76,  # 76
]

from dadk.QUBOSolverDAv2 import *

print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}");
print(f"DADK {DADK_version} {DADK_flavour}");

number_of_seeds = 4

step = 20_000

Ms = sorted(list(set(list(range(step, 2_000_000 + 1, step)))))

for case in cases:

    id, N, objective, penalty = case()

    if N > 64:
        continue

    factor = (2_000_000 / objective.get_max_abs_coefficient())

    objective *= factor

    for seed in [42]:

        summary = {}

        filename = f'results-{id}.json'

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
