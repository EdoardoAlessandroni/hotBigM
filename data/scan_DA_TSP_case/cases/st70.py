from dadk.BinPol import *

def st70() -> BinPol:
    coordinates = [
        (64, 96),
        (80, 39),
        (69, 23),
        (72, 42),
        (48, 67),
        (58, 43),
        (81, 34),
        (79, 17),
        (30, 23),
        (42, 67),
        (7, 76),
        (29, 51),
        (78, 92),
        (64, 8),
        (95, 57),
        (57, 91),
        (40, 35),
        (68, 40),
        (92, 34),
        (62, 1),
        (28, 43),
        (76, 73),
        (67, 88),
        (93, 54),
        (6, 8),
        (87, 18),
        (30, 9),
        (77, 13),
        (78, 94),
        (55, 3),
        (82, 88),
        (73, 28),
        (20, 55),
        (27, 43),
        (95, 86),
        (67, 99),
        (48, 83),
        (75, 81),
        (8, 19),
        (20, 18),
        (54, 38),
        (63, 36),
        (44, 33),
        (52, 18),
        (12, 13),
        (25, 5),
        (58, 85),
        (5, 67),
        (90, 9),
        (41, 76),
        (25, 76),
        (37, 64),
        (56, 63),
        (10, 55),
        (98, 7),
        (16, 74),
        (89, 60),
        (48, 82),
        (81, 76),
        (29, 60),
        (17, 22),
        (5, 45),
        (79, 70),
        (9, 100),
        (17, 82),
        (74, 67),
        (10, 68),
        (48, 19),
        (83, 86),
        (84, 94)
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
