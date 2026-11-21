"""Lennard-Jones forces compiled with Numba: the fastest version.

The loop is the same as in forces_newton3.py, but it works on NumPy arrays
and is decorated with @njit. The first call translates the function into
machine code (this takes a few seconds); every later call runs at the
speed of C. `cache=True` saves the machine code on disk for the next run.
"""

import numpy as np
from numba import njit

# Potential energy at the cutoff, for sigma_ij = epsilon_ij = 1 (see forces_python.py).
SHIFT = 4.0 * (0.4**12 - 0.4**6)


def _forces_kernel(positions, box, sigmas, epsilons):
    # ZADATAK 3b: ubrzajte petlju pomoću Numba-e.
    
    # Dok ovo ne uradite, funkcija vraća nulte sile.
    return np.zeros(positions.shape), 0.0


def compute_forces(positions, box, sigmas, epsilons):
    """Forces on all particles and the total potential energy (same physics as forces_python.py).

    Numba compiles a separate machine-code version for every kind of input
    (int or float, contiguous in memory or not). Converting everything to
    contiguous float64 arrays first means it compiles only once.

    Returns (forces, potential_energy); forces has shape (N, DIM).
    """
    forces, potential_energy = _forces_kernel(
        np.ascontiguousarray(positions, dtype=np.float64),
        np.ascontiguousarray(box, dtype=np.float64),
        np.ascontiguousarray(sigmas, dtype=np.float64),
        np.ascontiguousarray(epsilons, dtype=np.float64),
    )
    return forces, float(potential_energy)
