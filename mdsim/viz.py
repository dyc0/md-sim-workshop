"""Pictures and animations: drawing particles, the live viewer, playback, energy plots.

This is the only module that uses matplotlib. Importing it does not open
any window; windows appear only when a script calls plt.show().
"""

import copy
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Rectangle

from mdsim import observables
from mdsim.forces_python import compute_forces as python_compute_forces
from mdsim.integrators import velocity_verlet_step
from mdsim.simulation import energy_exploded

# Colors. Species A and B use the Okabe-Ito palette, which color-blind
# people can tell apart. "dark" is for showing the viewer next to dark slides.
_STYLES = {
    "light": {
        "background": "white",
        "foreground": "black",
        "species": ["#0072B2", "#E69F00"],
        "energies": ["#009E73", "#CC79A7", "black"],  # kinetic, potential, total
    },
    "dark": {
        "background": "#000000",
        "foreground": "#FFFFFF",
        "species": ["#56B4E9", "#E69F00"],
        "energies": ["#009E73", "#CC79A7", "#FFFFFF"],
    },
}

# Without a window (headless, e.g. MPLBACKEND=Agg) the animations are advanced
# by hand this many frames, so the saved picture shows a real state.
_HEADLESS_FRAMES = 30

# The live energy panel keeps only the most recent points, so the viewer stays
# fast even when it runs for a whole lecture.
_ENERGY_HISTORY_LENGTH = 2000

_NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def _is_interactive_backend():
    """True if matplotlib can open a window (False for Agg and other file-only backends)."""
    backend = matplotlib.get_backend().lower()
    return backend not in _NON_INTERACTIVE_BACKENDS and "inline" not in backend


def _do_nothing():
    return []


def _apply_style(fig, axes, style):
    colors = _STYLES[style]
    fig.set_facecolor(colors["background"])
    for ax in axes:
        ax.set_facecolor(colors["background"])
        ax.tick_params(colors=colors["foreground"])
        for spine in ax.spines.values():
            spine.set_color(colors["foreground"])
        ax.xaxis.label.set_color(colors["foreground"])
        ax.yaxis.label.set_color(colors["foreground"])
        ax.title.set_color(colors["foreground"])


def draw_particles(ax, positions, sigmas, species, box, style="light"):
    """Draw every particle as a circle that is sigma wide, colored by species.

    Uses one EllipseCollection, so even thousands of particles draw quickly.
    A particle that sticks out over the edge is drawn cut off (the copy on
    the other side of the periodic box is not drawn).
    Returns the collection, so an animation can move it with set_offsets.
    """
    if positions.shape[1] != 2 or len(box) != 2:
        raise ValueError(f"draw_particles only draws 2D systems, got DIM = {positions.shape[1]}")
    colors = np.array(_STYLES[style]["species"])[species]
    collection = EllipseCollection(
        widths=sigmas,
        heights=sigmas,
        angles=0,
        units="xy",
        offsets=positions,
        offset_transform=ax.transData,
        facecolors=colors,
        edgecolors="none",
    )
    ax.add_collection(collection)
    ax.add_patch(Rectangle((0, 0), box[0], box[1], fill=False,
                           edgecolor=_STYLES[style]["foreground"], linewidth=1))
    ax.set_xlim(0, box[0])
    ax.set_ylim(0, box[1])
    ax.set_aspect("equal")
    return collection


def animate_live(system, dt, step=velocity_verlet_step, compute_forces=python_compute_forces,
                 steps_per_frame=10, show_energy=True, start_paused=False, style="light"):
    """Run the simulation and show it while it runs.

    Every frame of the animation advances the simulation by `steps_per_frame`
    steps and redraws the particles. The right panel (if show_energy) plots
    kinetic, potential and total energy over time.

    Keys: space = pause / resume, r = restart from the beginning,
    q = close the window, f = fullscreen (built into matplotlib).

    Returns (figure, animation). Keep both in variables until plt.show(),
    otherwise Python deletes the animation.
    """
    colors = _STYLES[style]
    start = copy.deepcopy(system)  # for restarting with "r"
    n_particles = system.n_particles

    if show_energy:
        fig, (ax, ax_energy) = plt.subplots(1, 2, figsize=(12, 6))
        axes = [ax, ax_energy]
    else:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        axes = [ax]
    _apply_style(fig, axes, style)
    particles = draw_particles(ax, system.positions, system.sigmas, system.species, system.box, style)

    state = {"paused": start_paused, "time": 0.0}
    history = {"time": [], "kinetic": [], "potential": [], "total": []}

    if show_energy:
        energy_lines = []
        for label, color in zip(["kinetic", "potential", "total"], colors["energies"]):
            (line,) = ax_energy.plot([], [], color=color, label=f"{label} energy")
            energy_lines.append(line)
        ax_energy.set_xlabel("time (τ)")
        ax_energy.set_ylabel("energy (ε)")
        legend = ax_energy.legend(loc="center right", facecolor=colors["background"],
                                  edgecolor=colors["foreground"])
        for text in legend.get_texts():
            text.set_color(colors["foreground"])

    def compute_start_forces():
        state["forces"], state["potential_energy"] = compute_forces(
            system.positions, system.box, system.sigmas, system.epsilons
        )
        kinetic = observables.kinetic_energy(system.velocities, system.masses)
        state["start_energy"] = kinetic + state["potential_energy"]

    def show_state():
        kinetic = observables.kinetic_energy(system.velocities, system.masses)
        potential = state["potential_energy"]
        particles.set_offsets(system.positions)
        temperature = observables.temperature(system.velocities, system.masses)
        ax.set_title(f"t = {state['time']:.2f}    T = {temperature:.2f}")
        if show_energy:
            history["time"].append(state["time"])
            history["kinetic"].append(kinetic)
            history["potential"].append(potential)
            history["total"].append(kinetic + potential)
            for values in history.values():
                del values[:-_ENERGY_HISTORY_LENGTH]
            for line, key in zip(energy_lines, ["kinetic", "potential", "total"]):
                line.set_data(history["time"], history[key])
            ax_energy.relim()
            ax_energy.autoscale_view()

    def advance(frame):
        if state["paused"]:
            return
        for _ in range(steps_per_frame):
            state["forces"], state["potential_energy"] = step(system, state["forces"], dt, compute_forces)
            state["time"] += dt
            kinetic = observables.kinetic_energy(system.velocities, system.masses)
            if energy_exploded(kinetic + state["potential_energy"], state["start_energy"], n_particles):
                state["paused"] = True
                ax.set_title(f"Stopped at t = {state['time']:.2f}: time step too large (dt = {dt})")
                return
        show_state()

    def restart():
        system.positions[:] = start.positions
        system.velocities[:] = start.velocities
        state["time"] = 0.0
        for values in history.values():
            values.clear()
        compute_start_forces()
        show_state()
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "r":
            restart()
        elif event.key == "q":
            plt.close(fig)

    compute_start_forces()
    show_state()
    fig.canvas.mpl_connect("key_press_event", on_key)
    # init_func does nothing, so the first drawing of the window shows the
    # starting state instead of already running a frame.
    anim = FuncAnimation(fig, advance, init_func=_do_nothing, interval=1, cache_frame_data=False)

    if not _is_interactive_backend():
        for frame in range(_HEADLESS_FRAMES):
            advance(frame)
        fig.canvas.draw()
    return fig, anim


def play_trajectory(trajectory, interval_ms=30, every=1):
    """Play back a saved run. `every` = 2 shows every second frame, and so on.

    Returns (figure, animation).
    """
    fig, ax = plt.subplots(figsize=(6.5 * trajectory.box[0] / trajectory.box[1] + 0.5, 6.5))
    particles = draw_particles(ax, trajectory.positions[0], trajectory.sigmas,
                               trajectory.species, trajectory.box)
    frames = range(0, len(trajectory.times), every)

    def show_frame(index):
        particles.set_offsets(trajectory.positions[index])
        ax.set_title(f"t = {trajectory.times[index]:.2f}")

    show_frame(0)
    anim = FuncAnimation(fig, show_frame, frames=frames, init_func=_do_nothing,
                         interval=interval_ms, cache_frame_data=False)

    if not _is_interactive_backend():
        for index in list(frames)[:_HEADLESS_FRAMES]:
            show_frame(index)
        fig.canvas.draw()
    return fig, anim


def plot_energies(trajectory, ax=None):
    """Plot kinetic, potential and total energy of a run against time. Returns the axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    colors = _STYLES["light"]["energies"]
    # ZADATAK 2: nacrtajte tri krive u odnosu na vreme (trajectory.times):
    # kinetičku (trajectory.kinetic_energy), potencijalnu
    # (trajectory.potential_energy) i ukupnu energiju (trajectory.total_energy).

    # Dok ovo ne uradite, grafik ostaje prazan.
    return ax


def save_figure(fig, name):
    """Save a figure as output/<name> (the folder is created if needed) and print where."""
    folder = Path("output")
    folder.mkdir(exist_ok=True)
    path = folder / name
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved {path}")
    return path
