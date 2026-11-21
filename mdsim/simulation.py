"""The main simulation loop and the Trajectory that stores its results."""

from dataclasses import dataclass

import numpy as np

from mdsim import observables
from mdsim.forces_python import compute_forces as python_compute_forces
from mdsim.integrators import velocity_verlet_step


@dataclass
class Trajectory:
    """Everything a run stores: one entry per saved frame (F frames).

    Positions are wrapped into the box. The particle properties (species,
    sigmas, masses) and the box are stored too, so a saved run can be
    analysed or played back on its own.
    """

    times: np.ndarray             # (F,)
    positions: np.ndarray         # (F, N, DIM)
    kinetic_energy: np.ndarray    # (F,)
    potential_energy: np.ndarray  # (F,)
    temperature: np.ndarray       # (F,)
    momentum: np.ndarray          # (F, DIM)
    species: np.ndarray           # (N,)
    sigmas: np.ndarray            # (N,)
    masses: np.ndarray            # (N,)
    box: np.ndarray               # (DIM,)
    dt: float
    save_every: int

    @property
    def total_energy(self):
        """Total energy E = KE + PE of every frame."""
        return self.kinetic_energy + self.potential_energy

    def save(self, path):
        """Save the whole run into one .npz file, every field as a named array."""
        np.savez(
            path,
            times=self.times,
            positions=self.positions,
            kinetic_energy=self.kinetic_energy,
            potential_energy=self.potential_energy,
            temperature=self.temperature,
            momentum=self.momentum,
            species=self.species,
            sigmas=self.sigmas,
            masses=self.masses,
            box=self.box,
            dt=self.dt,
            save_every=self.save_every,
        )

    @staticmethod
    def load(path):
        """Read a run saved with `save`."""
        with np.load(path) as data:
            return Trajectory(
                times=data["times"],
                positions=data["positions"],
                kinetic_energy=data["kinetic_energy"],
                potential_energy=data["potential_energy"],
                temperature=data["temperature"],
                momentum=data["momentum"],
                species=data["species"],
                sigmas=data["sigmas"],
                masses=data["masses"],
                box=data["box"],
                dt=float(data["dt"]),
                save_every=int(data["save_every"]),
            )


class TimeStepTooLarge(RuntimeError):
    """The simulation exploded: particles came too close in one time step."""


# A real simulation never changes its energy by this much per particle (in units
# of epsilon). If it does, particles have crashed into each other: dt is too large.
MAX_ENERGY_CHANGE_PER_PARTICLE = 1000.0


def energy_exploded(total_energy, start_energy, n_particles):
    """True if the simulation has blown up: the energy is infinite or not a number,
    or it changed by more than MAX_ENERGY_CHANGE_PER_PARTICLE per particle."""
    if not np.isfinite(total_energy):
        return True
    return abs(total_energy - start_energy) > MAX_ENERGY_CHANGE_PER_PARTICLE * n_particles


def _check_energy(total_energy, start_energy, n_particles, step_number, dt):
    if energy_exploded(total_energy, start_energy, n_particles):
        raise TimeStepTooLarge(
            f"The simulation exploded at step {step_number} with dt = {dt}: "
            f"particles came too close in one step. Use a smaller time step."
        )


def _store_frame(frames, index, time, system, potential_energy):
    frames["times"][index] = time
    frames["positions"][index] = system.positions
    frames["kinetic_energy"][index] = observables.kinetic_energy(system.velocities, system.masses)
    frames["potential_energy"][index] = potential_energy
    frames["temperature"][index] = observables.temperature(system.velocities, system.masses)
    frames["momentum"][index] = observables.total_momentum(system.velocities, system.masses)


def run(system, n_steps, dt, step=velocity_verlet_step, compute_forces=python_compute_forces,
        save_every=10, progress=True):
    """Run the simulation for n_steps steps and return a Trajectory.

    Frame 0 is the starting state; after that a frame is stored every
    `save_every` steps. The system is changed in place.
    Raises TimeStepTooLarge if the simulation explodes (see energy_exploded).
    """
    n_frames = n_steps // save_every + 1
    n, dim = system.positions.shape
    frames = {
        "times": np.zeros(n_frames),
        "positions": np.zeros((n_frames, n, dim)),
        "kinetic_energy": np.zeros(n_frames),
        "potential_energy": np.zeros(n_frames),
        "temperature": np.zeros(n_frames),
        "momentum": np.zeros((n_frames, dim)),
    }
    report_every = max(1, n_steps // 10)

    forces, potential_energy = compute_forces(
        system.positions, system.box, system.sigmas, system.epsilons
    )
    _store_frame(frames, 0, 0.0, system, potential_energy)
    start_energy = frames["kinetic_energy"][0] + potential_energy

    for i in range(1, n_steps + 1):
        forces, potential_energy = step(system, forces, dt, compute_forces)
        energy = observables.kinetic_energy(system.velocities, system.masses) + potential_energy
        _check_energy(energy, start_energy, n, i, dt)
        if i % save_every == 0:
            _store_frame(frames, i // save_every, i * dt, system, potential_energy)
        if progress and i % report_every == 0:
            print(f"{100 * i // n_steps:3d} % done")

    return Trajectory(
        **frames,
        species=system.species.copy(),
        sigmas=system.sigmas.copy(),
        masses=system.masses.copy(),
        box=system.box.copy(),
        dt=float(dt),
        save_every=int(save_every),
    )


def equilibrate(system, n_steps, dt, step=velocity_verlet_step, compute_forces=python_compute_forces):
    """Run n_steps steps without storing anything, e.g. to let the lattice melt."""
    forces, potential_energy = compute_forces(
        system.positions, system.box, system.sigmas, system.epsilons
    )
    start_energy = observables.kinetic_energy(system.velocities, system.masses) + potential_energy
    for i in range(1, n_steps + 1):
        forces, potential_energy = step(system, forces, dt, compute_forces)
        energy = observables.kinetic_energy(system.velocities, system.masses) + potential_energy
        _check_energy(energy, start_energy, system.n_particles, i, dt)
