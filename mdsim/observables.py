"""Quantities we measure on the particles: kinetic energy, temperature, momentum.

All functions take arrays of shape (N, DIM) for the velocities and (N,) for
the masses, so they work in any number of dimensions.
"""

import numpy as np


def kinetic_energy(velocities, masses):
    """Total kinetic energy, KE = sum over particles of 1/2 * m * |v|^2."""
    # ZADATAK 2: izračunajte ukupnu kinetičku energiju svih čestica,
    # KE = zbir po česticama od 1/2 * m * |v|^2.

    # Vratite jedan broj (float). Dok ovo ne uradite, funkcija vraća 0.0,
    # pa je kinetička energija na grafiku ravna linija na nuli.
    return 0.0


def temperature(velocities, masses):
    """Temperature from the kinetic energy: T = 2 * KE / N_f (with k_B = 1).

    Each independent direction a particle can move in (a "degree of freedom")
    carries on average T/2 of kinetic energy. We have N particles in DIM
    dimensions, but we keep the whole system's center of mass at rest, which
    removes DIM degrees of freedom: N_f = DIM * (N - 1).

    This is the only place in the code where temperature is defined.
    """
    n_particles, dim = velocities.shape
    degrees_of_freedom = dim * (n_particles - 1)
    # The kinetic energy is summed here directly instead of calling
    # kinetic_energy(), so that temperature works before Problem 2 is done.
    speed_squared = np.sum(velocities * velocities, axis=1)
    kinetic = 0.5 * np.sum(masses * speed_squared)
    return float(2.0 * kinetic / degrees_of_freedom)


def total_momentum(velocities, masses):
    """Total momentum P = sum of m * v over all particles, shape (DIM,).

    Without outside forces it stays constant (Newton's third law), so it is
    a good check that the forces are correct.
    """
    return np.sum(masses[:, None] * velocities, axis=0)
