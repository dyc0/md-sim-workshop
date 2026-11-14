"""The particle system: positions, velocities and particle properties.

A `System` holds everything that describes the particles at one moment.
The `make_*` functions build a starting state: particles on a regular grid
(a lattice) with random velocities at a chosen temperature.
"""

from dataclasses import dataclass

import numpy as np

from mdsim import observables

# The cutoff of the Lennard-Jones interaction is 2.5 sigma. Every side of the
# box must be at least twice the cutoff, so a particle never sees two copies
# of the same neighbour through the periodic boundaries.
MIN_BOX_IN_SIGMAS = 5.0


@dataclass
class System:
    """All particles at one moment in time.

    positions:  (N, DIM) float, kept inside the box, 0 <= x < box
    velocities: (N, DIM) float
    masses:     (N,) float
    sigmas:     (N,) float, particle size (Lennard-Jones sigma)
    epsilons:   (N,) float, how strongly the particle attracts (Lennard-Jones epsilon)
    species:    (N,) int, 0 = A, 1 = B
    box:        (DIM,) float, side lengths of the periodic box
    """

    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    sigmas: np.ndarray
    epsilons: np.ndarray
    species: np.ndarray
    box: np.ndarray

    def __post_init__(self):
        # Check the shapes, so that every System that exists is valid.
        if self.positions.ndim != 2:
            raise ValueError(f"positions must have shape (N, DIM), got {self.positions.shape}")
        n, dim = self.positions.shape
        if self.velocities.shape != (n, dim):
            raise ValueError(f"velocities must have shape {(n, dim)}, got {self.velocities.shape}")
        for name in ("masses", "sigmas", "epsilons", "species"):
            shape = getattr(self, name).shape
            if shape != (n,):
                raise ValueError(f"{name} must have shape {(n,)}, got {shape}")
        if self.box.shape != (dim,):
            raise ValueError(f"box must have shape {(dim,)}, got {self.box.shape}")

        smallest_allowed = MIN_BOX_IN_SIGMAS * np.max(self.sigmas)
        if np.any(self.box < smallest_allowed):
            raise ValueError(
                f"box {self.box} is too small: every side must be at least "
                f"5 * max(sigma) = {smallest_allowed:.3g}. Use more particles "
                f"or a lower density."
            )

    @property
    def n_particles(self):
        return self.positions.shape[0]

    @property
    def dim(self):
        return self.positions.shape[1]


def lattice_positions(n_x, n_y, box):
    """Put n_x * n_y particles on a regular grid that fills the 2D box evenly.

    The grid spacing is box / (n_x, n_y). Every point is moved by half a
    spacing from the corner, so no particle sits exactly on the box edge.
    """
    if len(box) != 2:
        raise ValueError(f"lattice_positions only works in 2D, got a box with {len(box)} sides")
    spacing = np.asarray(box, dtype=float) / np.array([n_x, n_y])
    positions = []
    for i in range(n_x):
        for j in range(n_y):
            positions.append([(i + 0.5) * spacing[0], (j + 0.5) * spacing[1]])
    return np.array(positions)


def remove_center_of_mass_velocity(velocities, masses):
    """Return velocities with the motion of the center of mass removed.

    The center of mass moves with v_cm = sum(m_i * v_i) / sum(m_i). After
    subtracting v_cm from every particle the total momentum is zero, so the
    system as a whole does not drift through the box.

    The weights matter: the plain average of the velocities would only be
    right if all masses were equal. With a heavy and a light particle, the
    heavy one counts more.
    """
    v_cm = np.sum(masses[:, None] * velocities, axis=0) / np.sum(masses)
    return velocities - v_cm


def random_velocities(masses, temperature, rng, dim):
    """Random velocities that match a temperature exactly, with zero total momentum.

    1. Each velocity component is drawn from a normal (Gaussian) distribution
       with standard deviation sqrt(T / m): heavier particles move slower.
    2. The center-of-mass motion is removed.
    3. All velocities are scaled by one factor, so the measured temperature
       is exactly the requested one.
    """
    n = len(masses)
    if n < 2:
        # With one particle and its center of mass at rest, nothing can move:
        # there are no degrees of freedom, so temperature has no meaning.
        raise ValueError(f"need at least 2 particles to set a temperature, got N = {n}")
    if temperature == 0.0:
        return np.zeros((n, dim))
    standard_deviation = np.sqrt(temperature / masses)
    velocities = rng.normal(0.0, 1.0, size=(n, dim)) * standard_deviation[:, None]
    velocities = remove_center_of_mass_velocity(velocities, masses)
    measured = observables.temperature(velocities, masses)
    return velocities * np.sqrt(temperature / measured)


def make_single_species(n_x, n_y, density, temperature, seed, sigma=1.0, epsilon=1.0, mass=1.0):
    """A square box with n_x * n_y identical particles on a lattice.

    The box area is N / density. Velocities are random at the given
    temperature; `seed` makes the random numbers reproducible.
    """
    rng = np.random.default_rng(seed)
    n = n_x * n_y
    side = np.sqrt(n / density)
    box = np.array([side, side])
    masses = np.full(n, float(mass))
    return System(
        positions=lattice_positions(n_x, n_y, box),
        velocities=random_velocities(masses, temperature, rng, dim=2),
        masses=masses,
        sigmas=np.full(n, float(sigma)),
        epsilons=np.full(n, float(epsilon)),
        species=np.zeros(n, dtype=int),
        box=box,
    )


def make_two_species(n_x, n_y, density, temperature, seed, params_a, params_b, layout, aspect=1.0):
    """A box with two kinds of particles, A and B, on a lattice.

    params_a, params_b: dicts with "sigma", "epsilon" and "mass".
    The box is `aspect` times wider than it is tall, with area N / density.
    layout:
      "left_right": A in the left half, B in the right half (for mixing).
      "random":     exactly half A and half B, at random sites (well mixed).
    """
    rng = np.random.default_rng(seed)
    n = n_x * n_y
    box_y = np.sqrt(n / (density * aspect))
    box = np.array([aspect * box_y, box_y])
    positions = lattice_positions(n_x, n_y, box)

    largest_sigma = max(params_a["sigma"], params_b["sigma"])
    spacing = box / np.array([n_x, n_y])
    if np.any(spacing < largest_sigma):
        raise ValueError(
            f"lattice spacing {spacing} is smaller than the largest sigma "
            f"{largest_sigma}: particles would overlap. Lower the density."
        )

    if layout == "left_right":
        species = np.where(positions[:, 0] < box[0] / 2, 0, 1)
    elif layout == "random":
        if n % 2 != 0:
            raise ValueError(f'layout="random" needs an even number of particles, got N = {n}')
        species = rng.permutation(np.repeat([0, 1], n // 2))
    else:
        raise ValueError(f'unknown layout {layout!r}: use "left_right" or "random"')

    masses = np.where(species == 0, params_a["mass"], params_b["mass"]).astype(float)
    sigmas = np.where(species == 0, params_a["sigma"], params_b["sigma"]).astype(float)
    epsilons = np.where(species == 0, params_a["epsilon"], params_b["epsilon"]).astype(float)
    return System(
        positions=positions,
        velocities=random_velocities(masses, temperature, rng, dim=2),
        masses=masses,
        sigmas=sigmas,
        epsilons=epsilons,
        species=species,
        box=box,
    )
