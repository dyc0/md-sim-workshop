"""Part 4: where does the time go? Profiling the pure Python version.

cProfile records how much time every function takes. Look for:
compute_forces takes almost all of the time - that is the loop to speed up.
"""

# ---- Parameters (edit these) ----
# Units: σ = particle diameter (length), τ = time unit,
#        ε = attraction strength (energy, and temperature with k_B = 1).
from mdsim.forces_python import compute_forces  # profile this force version

N_X = 10  # particles along x in the starting grid (N = N_X * N_Y)
N_Y = 10  # particles along y in the starting grid
DENSITY = 0.7  # particles per σ²; sets the box size (area = N / DENSITY)
START_TEMPERATURE = 1.3  # in ε/k_B; drops to about 1.0 once the starting grid melts
DT = 0.005  # time step, in τ
N_STEPS = 20  # number of time steps to profile
SEED = 1  # random seed: the same seed gives exactly the same run
N_TOP = 15  # how many functions to list
# ---- End of parameters ----

import cProfile
import pstats

import numpy as np

from mdsim.simulation import run
from mdsim.system import make_single_species

system = make_single_species(N_X, N_Y, DENSITY, START_TEMPERATURE, SEED)

profiler = cProfile.Profile()
profiler.enable()
trajectory = run(system, N_STEPS, DT, compute_forces=compute_forces, save_every=1, progress=False)
profiler.disable()

print(f"Mean temperature: {np.mean(trajectory.temperature):.3f}")
print()
pstats.Stats(profiler).sort_stats("cumulative").print_stats(N_TOP)
