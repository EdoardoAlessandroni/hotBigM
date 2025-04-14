import numpy as np
from typing import Tuple

f = lambda x, Q: x.T @ Q @ x


def expectation_delta_x(Q: np.ndarray, m: int):
    """Compute the expected energy difference of a bit flip.

    Args:
        Q (np.ndarray): The QUBO matrix.
        m (int): The number of samples taken.

    Returns:
        float: The expected energy difference of a bit flip.
    """
    N = Q.shape[0]
    Q_diag = np.diag(Q)
    Q_full = Q + Q.T
    np.fill_diagonal(Q_full, Q_diag)

    flips = np.random.randint(0, N, (m,))
    x = np.random.randint(0, 2, (N,))
    h = np.sum(Q_full * x, axis=1) + (1 - x) * Q_diag

    r = np.sum(h) / N
    for i in range(m):
        dh = Q_full[flips[i]] * (1 - 2 * x[flips[i]])
        x[flips[i]] = 1 - x[flips[i]]
        dh[flips[i]] = 0
        h -= dh
        r += np.sum(np.abs(h)) / N
    return r / m


def deterministic_temperature(
    Q: np.ndarray,
    start_flip_prob: float = 0.5,
    end_flip_prob: float = 0.01,
    factor: float = 2.0,
) -> Tuple[float, float]:
    """Sample the temperature range for the annealing process.
    Pretty much exactly from https://github.com/jtiosue/qubovert/blob/master/qubovert/sim/_anneal_temperature_range.py

    Args:
        Q (np.ndarray): The QUBO matrix.
        start_flip_prob (float): Flip probability at the start.
        end_flip_prob (float): Flip probability at the end.
        factor (float): Factor for approximating the maximum/minimum energy difference.

    Returns:
        Tuple[float, float]: The start and end time as a tuple.
    """
    min_del_energy = factor * np.min(np.abs(Q)[np.nonzero(Q)])
    Q_full = Q.T + Q
    np.fill_diagonal(Q_full, np.diagonal(Q))
    max_del_energy = factor * np.max(np.sum(np.abs(Q_full), axis=0))

    t_0 = -max_del_energy / np.log(start_flip_prob)
    t_end = -min_del_energy / np.log(end_flip_prob)
    return t_0, t_end


def geometric_temperature_schedule(
    t_0: float, t_end: float, num_t_values: int, generate_inverse: bool = True
) -> np.ndarray:
    """Compute the complete simulated annealing temperature schedule using the parameters.
    Computes the beta schedule, i.e., the inverse temperature values, by default.

    Args:
        t_0 (float): Start temperature.
        t_end (float): End temperature.
        num_t_values (int): Number of iterations.
        generate_inverse (bool, optional): If the inverse (beta schedule) should be computed. Defaults to True.

    Returns:
        np.ndarray: The temperature values.
    """
    epsilon = np.exp(np.log(t_end / t_0) / num_t_values)
    ts = np.zeros(num_t_values)
    for i in range(0, num_t_values):
        ts[i] = t_0 * epsilon**i

    if not generate_inverse:
        return ts

    # Otherwise invert the temperature
    for i in range(len(ts)):
        ts[i] = 1.0 / ts[i]
    return ts


def simulated_annealing(
    Q: np.ndarray,
    t_0: float | None = None,
    t_end: float | None = None,
    num_t_values: int | None = None,
    constant_temperature_steps: int = 1,
    seed: int | None = None,
) -> Tuple[np.ndarray, float]:
    """Simulated annealing with a computational complexity of O(n * t),
    where t is the number of timesteps.
    This is achieved by computing only the updated values which are at most
    n per update step.

    Args:
        Q (np.ndarray): The QUBO matrix.
        t_0 (float | None, optional): Start temperature.
        t_end (float | None, optional): End temperature.
        num_t_values (int | None, optional): Number of update steps (complete steps are num_t_values * constant_temperature_steps). Defaults to the size of the QUBO squared.
        constant_temperature_steps (int | None, optional): Number of update steps where the temperature is kept constants. Defaults to 1.
        seed (int | None, optional): Random seed. Defaults to None.


    Returns:
        Tuple[np.ndarray, float]: The best solutions and its energy.
    """
    rng = np.random.Generator(np.random.PCG64(seed=seed))

    # Create helper matrix
    n = Q.shape[0]
    Q_outer = Q + Q.T
    np.fill_diagonal(Q_outer, 0)

    if num_t_values is None:
        num_t_values = n**2

    # Random initial
    x = rng.integers(0, high=2, size=(n,))
    f_x = f(x, Q)

    # Create the inverted temperature values
    if t_0 is None or t_end is None:
        t_0_, t_end_ = deterministic_temperature(Q)
        if t_0 is None:
            t_0 = t_0_
        if t_end is None:
            t_end = t_end_
    betas = geometric_temperature_schedule(t_0, t_end, num_t_values)

    for beta in betas:
        for _ in range(constant_temperature_steps):
            # Random flip in x
            idx = rng.integers(0, high=n)

            # Compute the difference between the flip and the previous energy
            sign = -(2 * x[idx] - 1)
            f_difference = sign * (np.dot(x, Q_outer[idx]) + Q[idx, idx])
            f_y = f_x + f_difference

            # Accept the new one if better (t is inverted beforehand)
            if f_y <= f_x or (np.exp(-(f_y - f_x) * beta) > rng.uniform(0, 1)):
                x[idx] = 1 - x[idx]
                f_x = f_y

    return x, f_x


def simulated_annealing_rf(
    Q: np.ndarray,
    t_0: float | None = None,
    t_end: float | None = None,
    num_t_values: int | None = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, float]:
    """Rejection-free simulated annealing with parallelized update scheme.

    Args:
        Q (np.ndarray): The QUBO matrix.
        t_0 (float | None, optional): Start temperature.
        t_end (float | None, optional): End temperature.
        num_t_values (int | None, optional): Number of update steps. Defaults to the size of the QUBO squared.
        seed (int | None, optional): Random seed. Defaults to None.


    Returns:
        Tuple[np.ndarray, float]: The best solutions and its energy.
    """
    rng = np.random.Generator(np.random.PCG64(seed=seed))

    # Number of bits
    n = Q.shape[0]

    # For easier computation create a dense matrix
    Q_diag = np.diag(Q)
    Q_full = Q + Q.T
    np.fill_diagonal(Q_full, Q_diag)

    if num_t_values is None:
        num_t_values = n**2

    # Sample the inverse temperature schedule
    if t_0 is None or t_end is None:
        t_0_, t_end_ = deterministic_temperature(Q)
        if t_0 is None:
            t_0 = t_0_
        if t_end is None:
            t_end = t_end_
    ts = geometric_temperature_schedule(t_0, t_end, num_t_values, False)

    # Random initial x
    x = rng.integers(0, high=2, size=(n,))

    # Remember best values
    best_x = np.copy(x)
    f_x = f(x, Q)
    best_energy = f_x

    # ---------------- Start

    # The change of delta E with respect to a bitflip at index [i]
    # Initial flip
    h = np.sum(Q_full * x, axis=1) + (1 - x) * Q_diag

    for t in ts:
        # Compute the differene for all flipped x at once
        delta_E = -(1 - 2 * (1 - x)) * h

        # Compute criteria
        u_s = rng.uniform(0, 1, size=delta_E.shape)
        criteria = np.maximum(0, delta_E) + t * np.log(-np.log(u_s))
        accepted_state_idx = criteria.argmin()

        # Accept the state by flipping x
        x[accepted_state_idx] = 1 - x[accepted_state_idx]

        # Check for best solution
        f_x += delta_E[accepted_state_idx]
        if f_x < best_energy:
            best_x = np.copy(x)
            best_energy = f_x

        # Then update the h
        dh = Q_full[accepted_state_idx] * (1 - 2 * x[accepted_state_idx])
        dh[accepted_state_idx] = 0
        h -= dh

    return best_x, best_energy


def metropolis_hastings_criterion(deltaE: np.ndarray, t: float) -> np.ndarray:
    """Metropolis-Hastings criterion with with inverse.

    Args:
        deltaE (np.ndarray): The energy difference.
        t (float): The time schedule value.

    Returns:
        np.ndarray: The probability criterion array.
    """
    # Clamp the inverted delta values to remove overflow warning
    criterion = np.minimum(0, -deltaE)
    return np.minimum(1, np.exp(criterion * t))


def simulated_annealing_qrf(
    Q: np.ndarray,
    t_0: float | None = None,
    t_end: float | None = None,
    num_t_values: int | None = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, float]:
    """Quasi rejection-free simulated annealing with parallelized update scheme.

    Args:
        Q (np.ndarray): The QUBO matrix.
    t_0: float | None = None,
    t_end: float | None = None,
        num_t_values (int | None, optional): Number of update steps. Defaults to the size of the QUBO squared.
        temperature_sampling_mode (TEMPERATURE_SAMPLING_TYPE): The way of sampling the temperature start and end values. Defaults to deterministic.
        seed (int | None, optional): Random seed. Defaults to None.

    Returns:
        Tuple[np.ndarray, float]: The best solutions and its energy.
    """
    rng = np.random.Generator(np.random.PCG64(seed=seed))

    # Number of bits
    n = Q.shape[0]

    # For easier computation create a dense matrix
    Q_diag = np.diag(Q)
    Q_full = Q + Q.T
    np.fill_diagonal(Q_full, Q_diag)

    if num_t_values is None:
        num_t_values = n**2

    # Create the inverted temperature values
    if t_0 is None or t_end is None:
        t_0_, t_end_ = deterministic_temperature(Q)
        if t_0 is None:
            t_0 = t_0_
        if t_end is None:
            t_end = t_end_
    betas = geometric_temperature_schedule(t_0, t_end, num_t_values)

    offset_increase_rate = expectation_delta_x(Q, 8) / 3

    # Random initial x
    x = rng.integers(0, high=2, size=(n,))

    # Remember best values
    best_x = np.copy(x)
    f_x = f(x, Q)
    best_energy = f_x

    # ---------------- Start

    # The change of delta E with respect to a bitflip at index [i]
    # Initial flip
    h = np.sum(Q_full * x, axis=1) + (1 - x) * Q_diag

    for beta in betas:
        # Compute the differene for all flipped x at once
        delta_E = -(1 - 2 * (1 - x)) * h
        delta = 0.0
        while True:
            # Check for accepted elements
            criteria = metropolis_hastings_criterion(delta_E - delta, beta)
            u_s = rng.uniform(0, 1, size=criteria.shape)

            # Then some acceptance probabilities
            accepted = np.where(criteria > u_s)[0]  # Only care about true elements

            # If at least one element got accepted, jump out
            if len(accepted):
                break
            # Otherwise increase delta
            delta += offset_increase_rate

        # Then take one randomly
        accepted_state_idx = rng.choice(accepted)

        # Accept the state by flipping x
        x[accepted_state_idx] = 1 - x[accepted_state_idx]

        # Check for best solution
        f_x += delta_E[accepted_state_idx]
        if f_x < best_energy:
            best_x = np.copy(x)
            best_energy = f_x

        # Then update the h
        dh = Q_full[accepted_state_idx] * (1 - 2 * x[accepted_state_idx])
        dh[accepted_state_idx] = 0
        h -= dh

    return best_x, best_energy
