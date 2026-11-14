"""Analysis of saved runs: the concentration profile for the mixing experiment.

This function only uses NumPy arrays, so it works on any saved trajectory.
"""

import numpy as np


def concentration_profile(positions, species, box, n_bins=20, axis=0):
    """Fraction of particles that are species A, in thin slices across the box.

    The box is cut into `n_bins` equal slices along `axis` (0 = x, 1 = y).
    For each slice we count the A particles and divide by all particles in
    that slice. At the start of the mixing experiment this is 1 on the left
    and 0 on the right; as the species mix it flattens towards 0.5.

    positions: (N, DIM) for one frame. Returns (bin_centers, fraction_a);
    slices with no particles get NaN.
    """
    x = positions[:, axis]
    length = box[axis]
    bins = np.floor(x / length * n_bins).astype(int)
    # A position can be exactly equal to the box length after wrapping; put it in the last slice.
    bins = np.clip(bins, 0, n_bins - 1)

    count_all = np.bincount(bins, minlength=n_bins).astype(float)
    count_a = np.bincount(bins[species == 0], minlength=n_bins).astype(float)
    fraction_a = np.full(n_bins, np.nan)
    np.divide(count_a, count_all, out=fraction_a, where=count_all > 0)

    bin_centers = (np.arange(n_bins) + 0.5) * length / n_bins
    return bin_centers, fraction_a
