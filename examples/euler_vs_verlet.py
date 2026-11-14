"""Part 3: forward Euler against velocity Verlet.

Both integrators start from exactly the same state. Look for: with forward
Euler the total energy keeps growing (the simulation slowly heats up by
itself), with velocity Verlet it stays flat.
"""

# ---- Parameters (edit these) ----
# Units: σ = particle diameter (length), τ = time unit,
#        ε = attraction strength (energy, and temperature with k_B = 1).
from mdsim.forces_python import compute_forces  # or forces_newton3, forces_numpy, forces_numba

N_X = 10  # particles along x in the starting grid (N = N_X * N_Y)
N_Y = 10  # particles along y in the starting grid
DENSITY = 0.7  # particles per σ²; sets the box size (area = N / DENSITY)
START_TEMPERATURE = 1.3  # in ε/k_B; drops to about 1.0 once the starting grid melts
DT = 0.001  # time step, in τ; Euler drifts so fast that a larger dt soon explodes
N_STEPS = 1000  # number of time steps (run length = N_STEPS * DT, in τ)
SAVE_EVERY = 5  # store one frame every this many steps
SEED = 1  # random seed: the same seed gives exactly the same run
# ---- End of parameters ----

import copy

import matplotlib.pyplot as plt
import numpy as np

from mdsim.integrators import euler_step, velocity_verlet_step
from mdsim.simulation import run
from mdsim.system import make_single_species
from mdsim.viz import save_figure

start = make_single_species(N_X, N_Y, DENSITY, START_TEMPERATURE, SEED)
euler = run(copy.deepcopy(start), N_STEPS, DT, step=euler_step,
            compute_forces=compute_forces, save_every=SAVE_EVERY)
verlet = run(copy.deepcopy(start), N_STEPS, DT, step=velocity_verlet_step,
             compute_forces=compute_forces, save_every=SAVE_EVERY)

for name, trajectory in [("forward Euler", euler), ("velocity Verlet", verlet)]:
    energy = trajectory.total_energy
    drift = abs(energy[-1] - energy[0]) / abs(energy[0])
    print(f"{name:16s} energy change |E_end - E_start| / |E_start| = {drift:.2e}")
second_half = len(verlet.times) // 2
print(f"Mean temperature with velocity Verlet (second half): {np.mean(verlet.temperature[second_half:]):.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(euler.times, euler.total_energy, label="forward Euler")
ax.plot(verlet.times, verlet.total_energy, label="velocity Verlet")
ax.set_xlabel("time (τ)")
ax.set_ylabel("total energy (ε)")
ax.set_title(f"Same start, two integrators (dt = {DT})")
ax.legend()
save_figure(fig, "euler_vs_verlet.png")
plt.show()
