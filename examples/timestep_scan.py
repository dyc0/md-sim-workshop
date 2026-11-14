"""Part 3: how large can the time step be?

Runs velocity Verlet with several time steps, each for the same physical
time. Look for: the energy fluctuation grows like dt^2 (the dashed line),
and a too-large time step makes the simulation explode.
"""

# ---- Parameters (edit these) ----
# Units: σ = particle diameter (length), τ = time unit,
#        ε = attraction strength (energy, and temperature with k_B = 1).
from mdsim.forces_python import compute_forces  # or forces_newton3, forces_numpy, forces_numba

N_X = 10  # particles along x in the starting grid (N = N_X * N_Y)
N_Y = 10  # particles along y in the starting grid
DENSITY = 0.7  # particles per σ²; sets the box size (area = N / DENSITY)
START_TEMPERATURE = 1.3  # in ε/k_B; drops to about 1.0 once the starting grid melts
DTS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]  # time steps to try
TOTAL_TIME = 2.0  # physical time of every run, in τ
SEED = 1  # random seed: the same seed gives exactly the same run
# ---- End of parameters ----

import matplotlib.pyplot as plt
import numpy as np

from mdsim.simulation import TimeStepTooLarge, run
from mdsim.system import make_single_species
from mdsim.viz import save_figure

good_dts = []
fluctuations = []
exploded_dts = []
for dt in DTS:
    system = make_single_species(N_X, N_Y, DENSITY, START_TEMPERATURE, SEED)
    n_steps = round(TOTAL_TIME / dt)
    save_every = max(1, round(0.05 / dt))  # one frame every 0.05 time units
    try:
        trajectory = run(system, n_steps, dt, compute_forces=compute_forces,
                         save_every=save_every, progress=False)
    except TimeStepTooLarge:
        print(f"dt = {dt:<6}  exploded")
        exploded_dts.append(dt)
        continue
    energy = trajectory.total_energy
    fluctuation = np.std(energy) / abs(np.mean(energy))
    second_half = len(trajectory.times) // 2
    mean_temperature = np.mean(trajectory.temperature[second_half:])
    print(f"dt = {dt:<6}  std(E)/|mean(E)| = {fluctuation:.2e}   mean T = {mean_temperature:.3f}")
    good_dts.append(dt)
    fluctuations.append(fluctuation)

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(good_dts, fluctuations, "o-", label="velocity Verlet")
guide = np.array(DTS)
ax.loglog(guide, fluctuations[0] * (guide / good_dts[0]) ** 2, "--", color="gray", label="∝ dt²")
if exploded_dts:
    top = max(fluctuations) * 10
    ax.loglog(exploded_dts, [top] * len(exploded_dts), "x", color="red", markersize=12,
              markeredgewidth=3, label="exploded")
ax.set_xlabel("time step dt (τ)")
ax.set_ylabel("relative energy fluctuation")
ax.set_title(f"Same physical time ({TOTAL_TIME} τ) for every dt")
ax.legend()
save_figure(fig, "timestep_scan.png")
plt.show()
