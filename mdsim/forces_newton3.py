"""Lennard-Jones forces in pure Python, using Newton's third law.

The force that j puts on i is exactly opposite to the force that i puts
on j (F_ji = -F_ij). So we visit every pair only once (j > i), add the
force to i and subtract it from j. That is half the work of forces_python.py.
"""

import numpy as np

# Potential energy at the cutoff, for sigma_ij = epsilon_ij = 1.
# Because the cutoff is always 2.5 sigma_ij, sigma_ij / r_c = 0.4 for every pair,
# so the shift of a pair is simply epsilon_ij * SHIFT.
SHIFT = 4.0 * (0.4**12 - 0.4**6)


def compute_forces(positions, box, sigmas, epsilons):
    """Forces on all particles and the total potential energy.

    Lennard-Jones potential between particles i and j at distance r:
        V(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ] - (value at the cutoff)
    It pushes particles apart when they are closer than about 1.12 sigma
    and pulls them together when they are a little farther apart.
    Pairs farther than the cutoff 2.5 sigma do not interact at all.

    Returns (forces, potential_energy); forces has shape (N, DIM).
    """
    # Plain Python lists, so the loops below measure pure Python speed.
    pos = positions.tolist()
    box_list = box.tolist()
    sig = sigmas.tolist()
    eps = epsilons.tolist()
    n = len(pos)
    dim = len(box_list)

    forces = [[0.0] * dim for _ in range(n)]
    potential_energy = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            # Separation vector d = r_i - r_j, using the nearest periodic copy of j.
            d = [0.0] * dim
            r2 = 0.0
            for k in range(dim):
                dk = pos[i][k] - pos[j][k]
                dk = dk - box_list[k] * round(dk / box_list[k])
                d[k] = dk
                r2 += dk * dk

            # Mixing rules for two different particles (Lorentz-Berthelot).
            sigma_ij = 0.5 * (sig[i] + sig[j])
            epsilon_ij = (eps[i] * eps[j]) ** 0.5
            cutoff = 2.5 * sigma_ij
            if r2 >= cutoff * cutoff:
                continue

            sr2 = sigma_ij * sigma_ij / r2
            sr6 = sr2 * sr2 * sr2
            sr12 = sr6 * sr6
            potential_energy += 4.0 * epsilon_ij * (sr12 - sr6) - epsilon_ij * SHIFT
            strength = 24.0 * epsilon_ij * (2.0 * sr12 - sr6) / r2
            for k in range(dim):
                forces[i][k] += strength * d[k]
                forces[j][k] -= strength * d[k]

    return np.array(forces), potential_energy
