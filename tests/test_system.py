"""Tests for system setup (mdsim.system) and observables (mdsim.observables)."""

import copy
import warnings

import numpy as np
import pytest

from mdsim.observables import temperature, total_momentum
from mdsim.system import (
    System,
    lattice_positions,
    make_single_species,
    make_two_species,
    random_velocities,
    remove_center_of_mass_velocity,
)

SPECIES_A = dict(sigma=1.0, epsilon=1.0, mass=1.0)
SPECIES_B_HEAVY = dict(sigma=1.0, epsilon=1.0, mass=4.0)


# ---------- observables ----------

def test_temperature_and_momentum():
    v = np.array([[1.0, 0.0], [0.0, 2.0]])
    m = np.array([1.0, 2.0])
    assert temperature(v, m) == pytest.approx(4.5)  # N_f = 2 * (2 - 1) = 2
    assert np.allclose(total_momentum(v, m), [1.0, 4.0])


def test_observables_accept_3d_arrays():
    v = np.ones((3, 3))
    m = np.ones(3)
    assert temperature(v, m) == pytest.approx(1.5)  # N_f = 3 * (3 - 1) = 6
    assert total_momentum(v, m).shape == (3,)


# ---------- lattice and velocities ----------

def test_lattice_positions_fill_box_evenly():
    p = lattice_positions(4, 2, np.array([8.0, 4.0]))
    assert p.shape == (8, 2)
    assert np.allclose(p.min(axis=0), [1.0, 1.0])
    assert np.allclose(p.max(axis=0), [7.0, 3.0])
    assert len(np.unique(p, axis=0)) == 8


def test_lattice_positions_rejects_3d_box():
    with pytest.raises(ValueError):
        lattice_positions(2, 2, np.ones(3) * 10)


def test_random_velocities_hit_target_temperature_with_zero_momentum():
    masses = np.tile([1.0, 4.0], 50)
    v = random_velocities(masses, 1.3, np.random.default_rng(0), dim=2)
    assert v.shape == (100, 2)
    assert temperature(v, masses) == pytest.approx(1.3, rel=1e-12)
    assert np.allclose(total_momentum(v, masses), 0.0, atol=1e-12)


def test_random_velocities_at_zero_temperature_are_zero():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = random_velocities(np.ones(10), 0.0, np.random.default_rng(0), dim=2)
    assert np.all(v == 0.0)


def test_center_of_mass_removal_is_mass_weighted():
    v = np.array([[1.0, 0.0], [0.0, 0.0]])
    m = np.array([1.0, 3.0])
    v_new = remove_center_of_mass_velocity(v, m)
    assert np.allclose(total_momentum(v_new, m), 0.0)


# ---------- System validation ----------

def _valid_system_arrays(n=4):
    box = np.array([10.0, 10.0])
    return dict(
        positions=lattice_positions(2, 2, box),
        velocities=np.zeros((n, 2)),
        masses=np.ones(n),
        sigmas=np.ones(n),
        epsilons=np.ones(n),
        species=np.zeros(n, dtype=int),
        box=box,
    )


def test_system_accepts_valid_arrays():
    system = System(**_valid_system_arrays())
    assert system.n_particles == 4
    assert system.dim == 2


def test_system_rejects_box_too_small():
    with pytest.raises(ValueError, match="box"):
        make_single_species(4, 4, 0.7, 1.0, seed=1)  # side 4.78 < 5 sigma


def test_system_rejects_wrong_shapes():
    arrays = _valid_system_arrays()
    arrays["masses"] = np.ones(3)
    with pytest.raises(ValueError, match="masses"):
        System(**arrays)


# ---------- make_single_species ----------

def test_make_single_species():
    system = make_single_species(20, 20, 0.7, 1.0, seed=1)
    assert system.n_particles == 400
    assert system.dim == 2
    assert np.allclose(system.box, np.sqrt(400 / 0.7))
    assert temperature(system.velocities, system.masses) == pytest.approx(1.0)
    assert np.all(system.positions >= 0.0)
    assert np.all(system.positions < system.box)


def test_same_seed_same_system():
    a = make_single_species(10, 10, 0.7, 1.0, seed=5)
    b = make_single_species(10, 10, 0.7, 1.0, seed=5)
    assert np.array_equal(a.velocities, b.velocities)


def test_different_seed_different_system():
    a = make_single_species(10, 10, 0.7, 1.0, seed=5)
    b = make_single_species(10, 10, 0.7, 1.0, seed=6)
    assert not np.array_equal(a.velocities, b.velocities)


# ---------- make_two_species ----------

def test_left_right_layout():
    system = make_two_species(20, 10, 0.7, 1.0, 1, SPECIES_A, SPECIES_B_HEAVY,
                              "left_right", aspect=2.0)
    assert system.box[0] == pytest.approx(2 * system.box[1])
    assert np.sum(system.species == 0) == 100
    assert np.sum(system.species == 1) == 100
    assert np.all(system.positions[system.species == 0, 0] < system.box[0] / 2)
    assert np.all(system.masses[system.species == 1] == 4.0)
    assert np.all(system.masses[system.species == 0] == 1.0)


def test_random_layout_exact_halves():
    system = make_two_species(10, 10, 0.7, 1.0, 1, SPECIES_A, SPECIES_B_HEAVY, "random")
    assert np.sum(system.species == 0) == 50
    assert np.sum(system.species == 1) == 50


def test_random_layout_rejects_odd_n():
    with pytest.raises(ValueError, match="even"):
        make_two_species(5, 5, 0.3, 1.0, 1, SPECIES_A, SPECIES_B_HEAVY, "random")


def test_unknown_layout_rejected():
    with pytest.raises(ValueError, match="layout"):
        make_two_species(10, 10, 0.7, 1.0, 1, SPECIES_A, SPECIES_B_HEAVY, "stripes")


def test_spacing_below_max_sigma_rejected():
    big_b = dict(sigma=1.6, epsilon=1.0, mass=1.0)
    with pytest.raises(ValueError, match="spacing"):
        make_two_species(10, 10, 0.7, 1.0, 1, SPECIES_A, big_b, "random")


def test_no_pair_closer_than_0_9_sigma():
    big_b = dict(sigma=1.4, epsilon=1.0, mass=1.0)
    system = make_two_species(10, 10, 0.5, 1.0, 2, SPECIES_A, big_b, "random")
    d = system.positions[:, None, :] - system.positions[None, :, :]
    d -= system.box * np.round(d / system.box)
    r = np.sqrt((d * d).sum(axis=-1))
    sigma_ij = (system.sigmas[:, None] + system.sigmas[None, :]) / 2
    off_diagonal = ~np.eye(system.n_particles, dtype=bool)
    assert np.all(r[off_diagonal] >= 0.9 * sigma_ij[off_diagonal])


def test_system_can_be_deep_copied():
    system = make_single_species(10, 10, 0.7, 1.0, seed=1)
    other = copy.deepcopy(system)
    other.positions[0, 0] += 1.0
    assert other.positions[0, 0] != system.positions[0, 0]


def test_random_velocities_rejects_single_particle():
    with pytest.raises(ValueError, match="at least 2 particles"):
        random_velocities(np.ones(1), 1.0, np.random.default_rng(0), dim=2)


def test_make_single_species_rejects_one_particle():
    with pytest.raises(ValueError, match="at least 2 particles"):
        make_single_species(1, 1, 0.7, 1.0, seed=1)
