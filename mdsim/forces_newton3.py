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
    # ZADATAK 4a: napišite sile koristeći treći Njutnov zakon.
    # Sila kojom j deluje na i je suprotna sili kojom i deluje na j (F_ji = -F_ij),
    # pa svaki par čestica obiđite samo jednom i uštedite pola posla.
    
    # Dok ovo ne uradite, funkcija vraća nulte sile (čestice se kreću pravo,
    # kao da ne osećaju jedna drugu).
    return np.zeros(positions.shape), 0.0
