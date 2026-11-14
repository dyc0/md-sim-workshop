"""Lennard-Jones forces in pure Python: the slow, most readable version.

For every particle i we visit every other particle j, compute the force
that j puts on i, and add it to particle i. Every pair is visited twice
(once as i-j and once as j-i), so the potential energy is halved at the end.
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
        for j in range(n):
            if i == j:
                continue

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

    # Each pair was counted twice, once from each side.
    potential_energy = 0.5 * potential_energy
    return np.array(forces), potential_energy
