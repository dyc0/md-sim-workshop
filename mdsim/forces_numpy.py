"""Lennard-Jones forces with NumPy: no Python loops over particles.

Instead of visiting pairs one by one, we build tables (matrices) with one
entry per pair (i, j) and let NumPy do the arithmetic on whole tables at
once. That is much faster, but uses much more memory.
"""

import numpy as np

# Potential energy at the cutoff, for sigma_ij = epsilon_ij = 1 (see forces_python.py).
SHIFT = 4.0 * (0.4**12 - 0.4**6)


def compute_forces(positions, box, sigmas, epsilons):
    """Forces on all particles and the total potential energy (same physics as forces_python.py).

    Returns (forces, potential_energy); forces has shape (N, DIM).
    """
    # ZADATAK 3a: napišite iste sile bez Python petlji, pomoću NumPy tabela.
    # Ideja: umesto da obilazite parove jedan po jedan, napravite tabele
    # (matrice) sa po jednim elementom za svaki par (i, j), i računajte sa
    # celim tabelama odjednom. Finalna implementacija
    # ne treba imati ni jednu for petlju.
    
    # Dok ovo ne uradite, funkcija vraća nulte sile.
    return np.zeros(positions.shape), 0.0
