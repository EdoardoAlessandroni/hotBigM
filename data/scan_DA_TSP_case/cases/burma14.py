from dadk.BinPol import *

def burma14() -> BinPol:
    coordinates = [
        (16.47, 96.10),
        (16.47, 94.44),
        (20.09, 92.54),
        (22.39, 93.37),
        (25.23, 97.24),
        (22.00, 96.05),
        (20.47, 97.02),
        (17.20, 96.29),
        (16.30, 97.38),
        (14.05, 98.12),
        (16.53, 97.38),
        (21.52, 95.59),
        (19.41, 97.13),
        (20.09, 94.55)
    ]

    N = len(coordinates)

    distances = [[math.sqrt((coordinates[city_a][0] - coordinates[city_b][0]) ** 2 + (coordinates[city_a][1] - coordinates[city_b][1]) ** 2) for city_b in range(N)] for city_a in range(N)]

    var_shape_set = VarShapeSet(BitArrayShape('x', (N, N), axis_names=['Time', 'City']))

    penalty = BinPol(var_shape_set)
    for t in range(N):
        penalty += (1 - BinPol.sum(Term(1, (('x', t, c),), var_shape_set) for c in range(N))) ** 2
    for c in range(N):
        penalty += (1 - BinPol.sum(Term(1, (('x', t, c),), var_shape_set) for t in range(N))) ** 2

    objective = BinPol(var_shape_set)
    for t in range(N):
        objective += BinPol.sum(Term(distances[c_0][c_1], (('x', t, c_0), ('x', (t + 1) % N, c_1),), var_shape_set)
                                for c_0 in range(N) for c_1 in range(N) if c_0 != c_1)

    return os.path.basename(__file__).replace('.py', ''), N, objective, penalty
