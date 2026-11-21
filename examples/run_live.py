"""Part 1: watch a Lennard-Jones fluid move, live.

The particles start on a regular grid (a crystal) and quickly melt into a
liquid. The right panel shows kinetic, potential and total energy: kinetic
and potential energy trade places, and the total should stay constant.

Keys: space = pause / resume, r = restart, q = quit, f = fullscreen.
"""

# ---- Parameters (edit these) ----
# Units: σ = particle diameter (length), τ = time unit,
#        ε = attraction strength (energy, and temperature with k_B = 1).
# ZADATAK 3b: kad napišete forces_numba.py, zamenite forces_python sa
# forces_numba u redu ispod i probajte više čestica (npr. N_X = N_Y = 20 ili 40).
from mdsim.forces_python import compute_forces  # or forces_newton3, forces_numpy, forces_numba
from mdsim.integrators import velocity_verlet_step as step

N_X = 10  # particles along x in the starting grid (N = N_X * N_Y)
N_Y = 10  # particles along y in the starting grid
DENSITY = 0.7  # particles per σ²; sets the box size (area = N / DENSITY)
START_TEMPERATURE = 1.3  # in ε/k_B; drops to about 1.0 once the starting grid melts
DT = 0.005  # time step, in τ
STEPS_PER_FRAME = 10  # simulation steps between two pictures
SEED = 1  # random seed: the same seed gives exactly the same run
START_PAUSED = False  # True: open paused, press space to start
STYLE = "light"  # "light" or "dark" (for dark slides)
# ---- End of parameters ----

import matplotlib.pyplot as plt

from mdsim.system import make_single_species
from mdsim.viz import animate_live, save_figure

system = make_single_species(N_X, N_Y, DENSITY, START_TEMPERATURE, SEED)
fig, anim = animate_live(system, DT, step=step, compute_forces=compute_forces,
                         steps_per_frame=STEPS_PER_FRAME, start_paused=START_PAUSED, style=STYLE)
save_figure(fig, "run_live.png")
plt.show()
