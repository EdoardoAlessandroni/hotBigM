from dadk.BinPol import *

def ulysses16() -> BinPol:
    coordinates = [
        (38.24, 20.42),
        (39.57, 26.15),
        (40.56, 25.32),
        (36.26, 23.12),
        (33.48, 10.54),
        (37.56, 12.19),
        (38.42, 13.11),
        (37.52, 20.44),
        (41.23, 9.10),
        (41.17, 13.05),
        (36.08, -5.21),
        (38.47, 15.13),
        (38.15, 15.35),
        (37.51, 15.17),
        (35.49, 14.32),
        (39.36, 19.56)
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
