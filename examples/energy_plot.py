"""Part 2: is energy really conserved?

Runs the simulation with velocity Verlet and plots kinetic, potential and
total energy. Look for: kinetic and potential energy change a lot while the
crystal melts, but the total energy stays flat. The script also prints the
total energy at the start and at the end of the run.
"""

# ---- Parameters (edit these) ----
# Units: σ = particle diameter (length), τ = time unit,
#        ε = attraction strength (energy, and temperature with k_B = 1).
from mdsim.forces_python import compute_forces  # or forces_newton3, forces_numpy, forces_numba

N_X = 10  # particles along x in the starting grid (N = N_X * N_Y)
N_Y = 10  # particles along y in the starting grid
DENSITY = 0.7  # particles per σ²; sets the box size (area = N / DENSITY)
START_TEMPERATURE = 1.3  # in ε/k_B; drops to about 1.0 once the starting grid melts
DT = 0.005  # time step, in τ
N_STEPS = 1000  # number of time steps (run length = N_STEPS * DT, in τ)
SAVE_EVERY = 10  # store one frame every this many steps
SEED = 1  # random seed: the same seed gives exactly the same run
# ---- End of parameters ----

import matplotlib.pyplot as plt
import numpy as np

from mdsim.integrators import velocity_verlet_step
from mdsim.simulation import run
from mdsim.system import make_single_species
from mdsim.viz import plot_energies, save_figure

system = make_single_species(N_X, N_Y, DENSITY, START_TEMPERATURE, SEED)
trajectory = run(system, N_STEPS, DT, step=velocity_verlet_step, compute_forces=compute_forces,
                 save_every=SAVE_EVERY)

energy = trajectory.total_energy
second_half = len(trajectory.times) // 2
print(f"Mean temperature (second half of the run): {np.mean(trajectory.temperature[second_half:]):.3f}")
print(f"Total energy at the start: {energy[0]:.1f} ε, at the end: {energy[-1]:.1f} ε")

ax = plot_energies(trajectory)
ax.set_title(f"N = {system.n_particles}, dt = {DT}")
save_figure(ax.figure, "energy_plot.png")
plt.show()
