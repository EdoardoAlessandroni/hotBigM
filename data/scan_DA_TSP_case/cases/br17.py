from dadk.BinPol import *

def br17() -> BinPol:
    distances = []
    distances.append([0, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 0, 3, 5, 8, 8, 5])
    distances.append([3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5])
    distances.append([5, 3, 0, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 0, 48, 48, 24])
    distances.append([48, 48, 74, 0, 0, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12])
    distances.append([48, 48, 74, 0, 0, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12])
    distances.append([8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8])
    distances.append([8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8])
    distances.append([5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 0])
    distances.append([5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 0])
    distances.append([3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5])
    distances.append([3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5])
    distances.append([0, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 0, 3, 5, 8, 8, 5])
    distances.append([3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5])
    distances.append([5, 3, 0, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 0, 48, 48, 24])
    distances.append([8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8])
    distances.append([8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8])
    distances.append([5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 0])

    N = len(distances)

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
