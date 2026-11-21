"""Tests for the kinetic energy, the total energy and the energy plot."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mdsim.observables import kinetic_energy  # noqa: E402
from mdsim.simulation import run  # noqa: E402
from mdsim.system import make_single_species  # noqa: E402
from mdsim.viz import plot_energies  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def short_run():
    system = make_single_species(6, 6, 0.7, 1.0, seed=3)
    return run(system, 20, 0.005, save_every=5, progress=False)


# ---------- kinetic_energy (mdsim/observables.py) ----------

def test_kinetic_energy_two_particles():
    v = np.array([[1.0, 0.0], [0.0, 2.0]])
    m = np.array([1.0, 2.0])
    # 1/2 * 1 * 1^2 + 1/2 * 2 * 2^2 = 0.5 + 4 = 4.5
    assert kinetic_energy(v, m) == pytest.approx(4.5)


def test_kinetic_energy_uses_both_velocity_components():
    v = np.array([[3.0, 4.0]])  # |v| = 5
    m = np.array([2.0])
    assert kinetic_energy(v, m) == pytest.approx(25.0)


def test_kinetic_energy_works_in_3d():
    v = np.ones((3, 3))
    m = np.ones(3)
    assert kinetic_energy(v, m) == pytest.approx(4.5)


def test_kinetic_energy_returns_one_number():
    v = np.array([[1.0, 0.0], [0.0, 2.0]])
    m = np.array([1.0, 2.0])
    assert np.ndim(kinetic_energy(v, m)) == 0


# ---------- Trajectory.total_energy (mdsim/simulation.py) ----------

def test_total_energy_is_kinetic_plus_potential():
    traj = short_run()
    assert traj.total_energy.shape == traj.times.shape
    assert np.allclose(traj.total_energy, traj.kinetic_energy + traj.potential_energy)
    assert np.all(traj.kinetic_energy > 0)


# ---------- plot_energies (mdsim/viz.py) ----------

def test_plot_energies_draws_three_labelled_lines():
    traj = short_run()
    ax = plot_energies(traj)
    lines = ax.get_lines()
    assert len(lines) == 3
    assert all(line.get_label() and not line.get_label().startswith("_") for line in lines)
    assert np.allclose(lines[2].get_ydata(), traj.total_energy)


def test_plot_energies_has_axis_labels_and_legend():
    ax = plot_energies(short_run())
    assert ax.get_xlabel() != "" and ax.get_ylabel() != ""
    assert ax.get_legend() is not None
