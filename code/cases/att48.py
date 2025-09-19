from dadk.BinPol import *

def att48() -> BinPol:
    coordinates = [
        (6734, 1453),
        (2233, 10),
        (5530, 1424),
        (401, 841),
        (3082, 1644),
        (7608, 4458),
        (7573, 3716),
        (7265, 1268),
        (6898, 1885),
        (1112, 2049),
        (5468, 2606),
        (5989, 2873),
        (4706, 2674),
        (4612, 2035),
        (6347, 2683),
        (6107, 669),
        (7611, 5184),
        (7462, 3590),
        (7732, 4723),
        (5900, 3561),
        (4483, 3369),
        (6101, 1110),
        (5199, 2182),
        (1633, 2809),
        (4307, 2322),
        (675, 1006),
        (7555, 4819),
        (7541, 3981),
        (3177, 756),
        (7352, 4506),
        (7545, 2801),
        (3245, 3305),
        (6426, 3173),
        (4608, 1198),
        (23, 2216),
        (7248, 3779),
        (7762, 4595),
        (7392, 2244),
        (3484, 2829),
        (6271, 2135),
        (4985, 140),
        (1916, 1569),
        (7280, 4899),
        (7509, 3239),
        (10, 2676),
        (6807, 2993),
        (5185, 3258),
        (3023, 1942)
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
