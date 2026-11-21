"""Tests for the integrators (mdsim.integrators) and the main loop (mdsim.simulation)."""

import numpy as np
import pytest

from mdsim.forces_python import compute_forces as python_forces
from mdsim.integrators import euler_step, velocity_verlet_step
from mdsim.observables import total_momentum
from mdsim.simulation import TimeStepTooLarge, Trajectory, energy_exploded, equilibrate, run
from mdsim.system import System, make_single_species, make_two_species

# Measured 2026-11-14 with forces_python (N = 36, 1000 steps, dt = 0.005):
# velocity Verlet drift 6.0e-4.
VERLET_DRIFT_LIMIT = 1e-3


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


# ---------- integrators ----------

def test_free_particle_moves_straight_and_wraps():
    system = one_particle([9.9, 5.0], [1.0, 0.0])
    forces, _ = zero_forces(system.positions, system.box, system.sigmas, system.epsilons)
    euler_step(system, forces, 0.2, zero_forces)
    assert np.allclose(system.positions, [[0.1, 5.0]])
    assert np.allclose(system.velocities, [[1.0, 0.0]])


def test_euler_uses_old_velocity_for_positions():
    system = one_particle([5.0, 5.0], [0.5, 0.0], mass=2.0)
    forces, _ = constant_force(system.positions, None, None, None)
    euler_step(system, forces, 0.1, constant_force)
    assert system.positions[0, 0] == pytest.approx(5.0 + 0.5 * 0.1, rel=1e-14)
    assert system.velocities[0, 0] == pytest.approx(0.5 + 0.5 * 0.1, rel=1e-14)


def test_compute_forces_called_once_per_step():
    calls = []

    def counting_forces(positions, box, sigmas, epsilons):
        calls.append(1)
        return zero_forces(positions, box, sigmas, epsilons)

    system = one_particle([5.0, 5.0], [1.0, 0.0])
    euler_step(system, np.zeros((1, 2)), 0.1, counting_forces)
    assert len(calls) == 1


def test_step_returns_forces_at_new_positions():
    system = make_single_species(10, 10, 0.7, 1.0, seed=2)
    forces, _ = python_forces(system.positions, system.box, system.sigmas, system.epsilons)
    new_forces, new_pe = euler_step(system, forces, 0.005, python_forces)
    expected, expected_pe = python_forces(system.positions, system.box, system.sigmas, system.epsilons)
    assert np.array_equal(new_forces, expected)
    assert new_pe == expected_pe


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


def test_verlet_free_particle_moves_straight_and_wraps():
    system = one_particle([9.9, 5.0], [1.0, 0.0])
    forces, _ = zero_forces(system.positions, system.box, system.sigmas, system.epsilons)
    velocity_verlet_step(system, forces, 0.2, zero_forces)
    assert np.allclose(system.positions, [[0.1, 5.0]])
    assert np.allclose(system.velocities, [[1.0, 0.0]])


def test_verlet_constant_force_is_exact():
    # x = x0 + v*t + 1/2*a*t^2 and v = v0 + a*t, with a = F/m = 1/2
    system = one_particle([5.0, 5.0], [0.5, 0.0], mass=2.0)
    forces, _ = constant_force(system.positions, None, None, None)
    velocity_verlet_step(system, forces, 0.1, constant_force)
    assert system.positions[0, 0] == pytest.approx(5.0 + 0.5 * 0.1 + 0.5 * 0.5 * 0.1**2, rel=1e-14)
    assert system.velocities[0, 0] == pytest.approx(0.5 + 0.5 * 0.1, rel=1e-14)


def test_verlet_calls_compute_forces_once_per_step():
    calls = []

    def counting_forces(positions, box, sigmas, epsilons):
        calls.append(1)
        return zero_forces(positions, box, sigmas, epsilons)

    system = one_particle([5.0, 5.0], [1.0, 0.0])
    velocity_verlet_step(system, np.zeros((1, 2)), 0.1, counting_forces)
    assert len(calls) == 1


def test_verlet_returns_forces_at_new_positions():
    system = make_single_species(10, 10, 0.7, 1.0, seed=2)
    start = system.positions.copy()
    forces, _ = python_forces(system.positions, system.box, system.sigmas, system.epsilons)
    new_forces, new_pe = velocity_verlet_step(system, forces, 0.005, python_forces)
    assert not np.array_equal(system.positions, start)
    expected, expected_pe = python_forces(system.positions, system.box, system.sigmas, system.epsilons)
    assert np.array_equal(new_forces, expected)
    assert new_pe == expected_pe


def test_verlet_conserves_energy():
    start = make_single_species(6, 6, 0.7, 1.0, seed=2)
    system = make_single_species(6, 6, 0.7, 1.0, seed=2)
    assert total_energy_drift(system, velocity_verlet_step, 1000, 0.005) < VERLET_DRIFT_LIMIT
    assert not np.array_equal(system.positions, start.positions)


def test_verlet_momentum_stays_zero_with_unequal_masses():
    a = dict(sigma=1.0, epsilon=1.0, mass=1.0)
    b = dict(sigma=1.0, epsilon=1.0, mass=4.0)
    system = make_two_species(10, 10, 0.7, 1.0, 3, a, b, "random")
    start = system.velocities.copy()
    forces, _ = python_forces(system.positions, system.box, system.sigmas, system.epsilons)
    for _ in range(500):
        forces, _ = velocity_verlet_step(system, forces, 0.005, python_forces)
    assert not np.array_equal(system.velocities, start)
    assert np.allclose(total_momentum(system.velocities, system.masses), 0.0, atol=1e-10)


# ---------- simulation ----------

def test_run_stores_frames():
    system = make_single_species(10, 10, 0.7, 1.0, seed=2)
    initial_positions = system.positions.copy()
    traj = run(system, n_steps=25, dt=0.005, step=euler_step, save_every=10, progress=False)
    assert traj.positions.shape == (3, 100, 2)
    assert np.allclose(traj.times, [0.0, 0.05, 0.1])
    assert np.array_equal(traj.positions[0], initial_positions)
    assert not np.array_equal(traj.positions[-1], system.positions)  # last frame is step 20 of 25
    assert traj.momentum.shape == (3, 2)
    assert traj.dt == 0.005 and traj.save_every == 10


def test_run_with_euler_and_python_forces():
    from mdsim.forces_python import compute_forces as python_forces
    system = make_single_species(6, 6, 0.7, 1.0, seed=2)
    traj = run(system, 4, 0.005, step=euler_step, compute_forces=python_forces,
               save_every=2, progress=False)
    assert traj.positions.shape == (3, 36, 2)


def test_trajectory_save_load_round_trip(tmp_path):
    system = make_single_species(10, 10, 0.7, 1.0, seed=2)
    traj = run(system, n_steps=20, dt=0.005, save_every=10, progress=False)
    traj.save(tmp_path / "t.npz")
    back = Trajectory.load(tmp_path / "t.npz")
    for name in ("times", "positions", "kinetic_energy", "potential_energy", "temperature",
                 "momentum", "species", "sigmas", "masses", "box"):
        assert np.array_equal(getattr(back, name), getattr(traj, name)), name
    assert type(back.dt) is float and back.dt == traj.dt
    assert type(back.save_every) is int and back.save_every == traj.save_every
    assert [p.name for p in tmp_path.iterdir()] == ["t.npz"]


def test_absurd_dt_raises_time_step_too_large():
    system = make_single_species(10, 10, 0.7, 1.0, seed=2)
    with pytest.raises(TimeStepTooLarge, match="dt"):
        run(system, 200, dt=1.0, step=euler_step, progress=False)


def test_energy_exploded():
    assert energy_exploded(float("inf"), -100.0, 100)
    assert energy_exploded(float("nan"), -100.0, 100)
    assert energy_exploded(1e47, -100.0, 100)
    assert not energy_exploded(-90.0, -100.0, 100)
    assert not energy_exploded(-100.0 + 999.0 * 100, -100.0, 100)


def test_explosion_at_moderate_dt_is_caught():
    system = make_single_species(10, 10, 0.7, 1.0, seed=2)
    with pytest.raises(TimeStepTooLarge):
        run(system, 200, dt=0.05, step=euler_step, progress=False)


def test_slow_euler_drift_is_not_an_explosion():
    # Measured: after 50 Euler steps at dt = 0.005 the energy has grown by about 28 %.
    system = make_single_species(10, 10, 0.7, 1.0, seed=2)
    traj = run(system, 50, dt=0.005, step=euler_step, save_every=50, progress=False)
    assert len(traj.times) == 2  # the run finished without TimeStepTooLarge


def test_progress_output(capsys):
    system = make_single_species(10, 10, 0.7, 1.0, seed=2)
    run(system, 100, 0.005, progress=False)
    assert capsys.readouterr().out == ""
    run(system, 100, 0.005, progress=True)
    lines = capsys.readouterr().out.strip().splitlines()
    assert 1 <= len(lines) <= 11
    assert all("%" in line for line in lines)


def test_equilibrate_advances_without_storing():
    system = make_single_species(10, 10, 0.7, 1.0, seed=2)
    start = system.positions.copy()
    assert equilibrate(system, 20, 0.005, step=euler_step) is None
    assert not np.array_equal(system.positions, start)
