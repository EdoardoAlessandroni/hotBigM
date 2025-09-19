from dadk.BinPol import *

def pr76() -> BinPol:
    coordinates = [
        (3600, 2300),
        (3100, 3300),
        (4700, 5750),
        (5400, 5750),
        (5608, 7103),
        (4493, 7102),
        (3600, 6950),
        (3100, 7250),
        (4700, 8450),
        (5400, 8450),
        (5610, 10053),
        (4492, 10052),
        (3600, 10800),
        (3100, 10950),
        (4700, 11650),
        (5400, 11650),
        (6650, 10800),
        (7300, 10950),
        (7300, 7250),
        (6650, 6950),
        (7300, 3300),
        (6650, 2300),
        (5400, 1600),
        (8350, 2300),
        (7850, 3300),
        (9450, 5750),
        (10150, 5750),
        (10358, 7103),
        (9243, 7102),
        (8350, 6950),
        (7850, 7250),
        (9450, 8450),
        (10150, 8450),
        (10360, 10053),
        (9242, 10052),
        (8350, 10800),
        (7850, 10950),
        (9450, 11650),
        (10150, 11650),
        (11400, 10800),
        (12050, 10950),
        (12050, 7250),
        (11400, 6950),
        (12050, 3300),
        (11400, 2300),
        (10150, 1600),
        (13100, 2300),
        (12600, 3300),
        (14200, 5750),
        (14900, 5750),
        (15108, 7103),
        (13993, 7102),
        (13100, 6950),
        (12600, 7250),
        (14200, 8450),
        (14900, 8450),
        (15110, 10053),
        (13992, 10052),
        (13100, 10800),
        (12600, 10950),
        (14200, 11650),
        (14900, 11650),
        (16150, 10800),
        (16800, 10950),
        (16800, 7250),
        (16150, 6950),
        (16800, 3300),
        (16150, 2300),
        (14900, 1600),
        (19800, 800),
        (19800, 10000),
        (19800, 11900),
        (19800, 12200),
        (200, 12200),
        (200, 1100),
        (200, 800)
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
