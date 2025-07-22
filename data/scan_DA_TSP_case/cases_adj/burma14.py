import numpy as np

def burma14():
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

    distances = [[np.sqrt((coordinates[city_a][0] - coordinates[city_b][0]) ** 2 + (coordinates[city_a][1] - coordinates[city_b][1]) ** 2) for city_b in range(N)] for city_a in range(N)]
    return distances