from dadk.BinPol import *

def gr17() -> BinPol:
    data = {}
    data[0] = [0]
    data[1] = [633, 0]
    data[2] = [257, 390, 0]
    data[3] = [91, 661, 228, 0]
    data[4] = [412, 227, 169, 383, 0]
    data[5] = [150, 488, 112, 120, 267, 0]
    data[6] = [80, 572, 196, 77, 351, 63, 0]
    data[7] = [134, 530, 154, 105, 309, 34, 29, 0]
    data[8] = [259, 555, 372, 175, 338, 264, 232, 249, 0]
    data[9] = [505, 289, 262, 476, 196, 360, 444, 402, 495, 0]
    data[10] = [353, 282, 110, 324, 61, 208, 292, 250, 352, 154, 0]
    data[11] = [324, 638, 437, 240, 421, 329, 297, 314, 95, 578, 435, 0]
    data[12] = [70, 567, 191, 27, 346, 83, 47, 68, 189, 439, 287, 254, 0]
    data[13] = [211, 466, 74, 182, 243, 105, 150, 108, 326, 336, 184, 391, 145, 0]
    data[14] = [268, 420, 53, 239, 199, 123, 207, 165, 383, 240, 140, 448, 202, 57, 0]
    data[15] = [246, 745, 472, 237, 528, 364, 332, 349, 202, 685, 542, 157, 289, 426, 483, 0]
    data[16] = [121, 518, 142, 84, 297, 35, 29, 36, 236, 390, 238, 301, 55, 96, 153, 336, 0]

    N = len(data)

    distances = [[data[max(city_a, city_b)][min(city_a, city_b)] for city_b in range(N)] for city_a in range(N)]

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
