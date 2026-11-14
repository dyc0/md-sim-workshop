"""Testovi za Zadatak 3: metoda velocity Verlet.

Pokrenite ih sa:  pytest tests/test_problem_3.py
Dok zadatak nije urađen, ovi testovi padaju (FAILED). Kad sve prođe (passed),
zadatak je gotov.
"""

import copy

import numpy as np
import pytest

from mdsim.forces_python import compute_forces as python_forces
from mdsim.integrators import euler_step, velocity_verlet_step
from mdsim.observables import total_momentum
from mdsim.system import System, make_single_species, make_two_species

# Measured 2026-11-14 with forces_python (N = 36, 1000 steps, dt = 0.005):
# velocity Verlet drift 6.0e-4, forward Euler drift 9.2e64 (Euler blows up).
VERLET_DRIFT_LIMIT = 1e-3
EULER_TO_VERLET_FACTOR = 10.0


def zero_forces(positions, box, sigmas, epsilons):
    return np.zeros_like(positions), 0.0


def constant_force(positions, box, sigmas, epsilons):
    return np.tile([1.0, 0.0], (len(positions), 1)), 0.0


def one_particle(position, velocity, mass=1.0):
    return System(
        positions=np.array([position], dtype=float),
        velocities=np.array([velocity], dtype=float),
        masses=np.array([mass]),
        sigmas=np.ones(1),
        epsilons=np.ones(1),
        species=np.zeros(1, dtype=int),
        box=np.array([10.0, 10.0]),
    )


def kinetic_energy(velocities, masses):
    """The tests' own kinetic energy, so they don't depend on Problem 2."""
    return 0.5 * np.sum(masses * np.sum(velocities * velocities, axis=1))


def total_energy_drift(system, step, n_steps, dt):
    """Largest relative change of total energy during n_steps steps."""
    forces, pe = python_forces(system.positions, system.box, system.sigmas, system.epsilons)
    e0 = kinetic_energy(system.velocities, system.masses) + pe
    largest = 0.0
    for _ in range(n_steps):
        forces, pe = step(system, forces, dt, python_forces)
        e = kinetic_energy(system.velocities, system.masses) + pe
        largest = max(largest, abs(e - e0) / abs(e0))
    return largest


def test_free_particle_moves_straight_and_wraps():
    system = one_particle([9.9, 5.0], [1.0, 0.0])
    forces, _ = zero_forces(system.positions, system.box, system.sigmas, system.epsilons)
    velocity_verlet_step(system, forces, 0.2, zero_forces)
    assert np.allclose(system.positions, [[0.1, 5.0]])
    assert np.allclose(system.velocities, [[1.0, 0.0]])


def test_constant_force_is_exact():
    # x = x0 + v*t + 1/2*a*t^2 and v = v0 + a*t, with a = F/m = 1/2
    system = one_particle([5.0, 5.0], [0.5, 0.0], mass=2.0)
    forces, _ = constant_force(system.positions, None, None, None)
    velocity_verlet_step(system, forces, 0.1, constant_force)
    assert system.positions[0, 0] == pytest.approx(5.0 + 0.5 * 0.1 + 0.5 * 0.5 * 0.1**2, rel=1e-14)
    assert system.velocities[0, 0] == pytest.approx(0.5 + 0.5 * 0.1, rel=1e-14)


def test_compute_forces_called_once_per_step():
    calls = []

    def counting_forces(positions, box, sigmas, epsilons):
        calls.append(1)
        return zero_forces(positions, box, sigmas, epsilons)

    system = one_particle([5.0, 5.0], [1.0, 0.0])
    velocity_verlet_step(system, np.zeros((1, 2)), 0.1, counting_forces)
    assert len(calls) == 1


def test_returns_forces_at_new_positions():
    system = make_single_species(10, 10, 0.7, 1.0, seed=2)
    start = system.positions.copy()
    forces, _ = python_forces(system.positions, system.box, system.sigmas, system.epsilons)
    new_forces, new_pe = velocity_verlet_step(system, forces, 0.005, python_forces)
    assert not np.array_equal(system.positions, start)
    expected, expected_pe = python_forces(system.positions, system.box, system.sigmas, system.epsilons)
    assert np.array_equal(new_forces, expected)
    assert new_pe == expected_pe


def test_verlet_conserves_energy_and_euler_does_not():
    start = make_single_species(6, 6, 0.7, 1.0, seed=2)
    verlet_system = copy.deepcopy(start)
    verlet_drift = total_energy_drift(verlet_system, velocity_verlet_step, 1000, 0.005)
    euler_drift = total_energy_drift(copy.deepcopy(start), euler_step, 1000, 0.005)
    assert not np.array_equal(verlet_system.positions, start.positions)
    assert verlet_drift < VERLET_DRIFT_LIMIT
    assert euler_drift > EULER_TO_VERLET_FACTOR * verlet_drift


def test_momentum_stays_zero_with_unequal_masses():
    a = dict(sigma=1.0, epsilon=1.0, mass=1.0)
    b = dict(sigma=1.0, epsilon=1.0, mass=4.0)
    system = make_two_species(10, 10, 0.7, 1.0, 3, a, b, "random")
    start = system.velocities.copy()
    forces, _ = python_forces(system.positions, system.box, system.sigmas, system.epsilons)
    for _ in range(500):
        forces, _ = velocity_verlet_step(system, forces, 0.005, python_forces)
    assert not np.array_equal(system.velocities, start)
    assert np.allclose(total_momentum(system.velocities, system.masses), 0.0, atol=1e-10)
