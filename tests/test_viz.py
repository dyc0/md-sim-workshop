"""Tests for mdsim.viz (drawing, live viewer, playback) with the headless Agg backend."""

import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backend_bases import KeyEvent  # noqa: E402

from mdsim import viz  # noqa: E402
from mdsim.integrators import euler_step  # noqa: E402
from mdsim.simulation import run  # noqa: E402
from mdsim.system import make_single_species  # noqa: E402
from mdsim.viz import (  # noqa: E402
    animate_live,
    draw_particles,
    play_trajectory,
    save_figure,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def small_system():
    return make_single_species(6, 6, 0.7, 1.0, seed=4)  # box side 7.2


def press(fig, key):
    fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, key=key))


def test_save_figure_writes_into_output_of_working_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    path = save_figure(plt.figure(), "test.png")
    assert (tmp_path / "output" / "test.png").is_file()
    assert path.name == "test.png"
    assert "test.png" in capsys.readouterr().out


def test_draw_particles():
    system = small_system()
    fig, ax = plt.subplots()
    collection = draw_particles(ax, system.positions, system.sigmas, system.species, system.box)
    assert len(collection.get_offsets()) == 36
    assert ax.get_xlim() == pytest.approx((0, system.box[0]))
    assert ax.get_ylim() == pytest.approx((0, system.box[1]))


def test_draw_particles_rejects_3d():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        draw_particles(ax, np.ones((4, 3)), np.ones(4), np.zeros(4, dtype=int), np.ones(3) * 10)


def test_animate_live_headless_advances_and_titles():
    system = small_system()
    start_positions = system.positions.copy()
    fig, anim = animate_live(system, dt=0.001, step=euler_step)
    assert not np.array_equal(system.positions, start_positions)
    assert "T =" in fig.axes[0].get_title()


def test_animate_live_without_energy_panel():
    fig, anim = animate_live(small_system(), dt=0.005, show_energy=False)
    assert len(fig.axes) == 1


def test_restart_key_restores_initial_state():
    system = small_system()
    start_positions = system.positions.copy()
    start_velocities = system.velocities.copy()
    fig, anim = animate_live(system, dt=0.005)
    press(fig, "r")
    assert np.array_equal(system.positions, start_positions)
    assert np.array_equal(system.velocities, start_velocities)
    assert all(len(line.get_xdata()) <= 1 for line in fig.axes[1].get_lines())


def test_space_key_toggles_pause_without_error():
    fig, anim = animate_live(small_system(), dt=0.005)
    press(fig, " ")
    press(fig, " ")


def test_start_paused_does_not_move_particles():
    system = small_system()
    start_positions = system.positions.copy()
    animate_live(system, dt=0.005, start_paused=True)
    assert np.array_equal(system.positions, start_positions)


def test_dark_style_has_dark_background():
    fig, anim = animate_live(small_system(), dt=0.005, style="dark")
    assert np.mean(fig.get_facecolor()[:3]) < 0.3


def test_live_viewer_survives_explosion():
    fig, anim = animate_live(small_system(), dt=1.0, step=euler_step)  # must not raise
    assert "time step too large" in fig.axes[0].get_title().lower()


def test_energy_history_is_bounded(monkeypatch):
    monkeypatch.setattr(viz, "_ENERGY_HISTORY_LENGTH", 5)
    monkeypatch.setattr(viz, "_HEADLESS_FRAMES", 20)
    fig, anim = animate_live(small_system(), dt=0.005, steps_per_frame=1)
    lines = fig.axes[1].get_lines()
    assert len(lines) == 3
    assert all(len(line.get_xdata()) <= 5 for line in lines)


def test_play_trajectory_headless():
    traj = run(small_system(), 20, 0.005, save_every=5, progress=False)
    fig, anim = play_trajectory(traj)
    assert "t =" in fig.axes[0].get_title()


def test_importing_mdsim_has_no_side_effects(tmp_path):
    code = (
        "import mdsim.system, mdsim.observables, mdsim.forces_python, mdsim.forces_newton3, "
        "mdsim.forces_numpy, mdsim.forces_numba, mdsim.integrators, mdsim.simulation, "
        "mdsim.analysis, mdsim.viz; import matplotlib.pyplot as plt; "
        "assert plt.get_fignums() == []"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []


def test_play_script_reports_missing_file(tmp_path):
    import os
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo / "examples" / "play.py"), "nope.npz"],
        cwd=tmp_path, capture_output=True, text=True, env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 1
    assert "File not found" in result.stdout
    assert "Usage" in result.stdout
    assert "Traceback" not in result.stderr


def run_play_script(tmp_path, *args):
    import os
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(repo / "examples" / "play.py"), *args],
        cwd=tmp_path, capture_output=True, text=True, env={**os.environ, "MPLBACKEND": "Agg"},
    )


def save_small_run(tmp_path):
    traj = run(small_system(), 20, 0.005, save_every=5, progress=False)
    traj.save(tmp_path / "small.npz")
    return "small.npz"


def test_play_script_uses_default_interval(tmp_path):
    result = run_play_script(tmp_path, save_small_run(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "30 ms between pictures" in result.stdout


def test_play_script_accepts_interval_argument(tmp_path):
    result = run_play_script(tmp_path, save_small_run(tmp_path), "100")
    assert result.returncode == 0, result.stderr
    assert "100 ms between pictures" in result.stdout


@pytest.mark.parametrize("bad", ["abc", "0", "-5", "2.5"])
def test_play_script_rejects_bad_interval(tmp_path, bad):
    result = run_play_script(tmp_path, save_small_run(tmp_path), bad)
    assert result.returncode == 1
    assert "positive whole number" in result.stdout
    assert "Usage" in result.stdout
    assert "Traceback" not in result.stderr
