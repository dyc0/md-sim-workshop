"""Part 6A: two kinds of particles mixing.

Species A (blue) starts in the left half of the box, species B (orange) in
the right half. As the particles move around, they mix. The figure shows
the fraction of A particles across the box at several times: a sharp step
at the start that gets smoother and smoother.

Try: give B a larger mass (MASS_B) or size (SIGMA_B). With PLAY = True the
run is played back as an animation at the end.
"""

# ---- Parameters (edit these) ----
# Units: σ = particle diameter (length), τ = time unit,
#        ε = attraction strength (energy, and temperature with k_B = 1).
from mdsim.forces_numba import compute_forces  # or forces_python, forces_newton3, forces_numpy

N_Y = 15  # particles along y in the starting grid
N_X = 2 * N_Y  # particles along x (the box is twice as wide as it is tall)
ASPECT = 2.0  # box width / box height
DENSITY = 0.7  # particles per σ²; sets the box size (area = N / DENSITY)
START_TEMPERATURE = 1.3  # in ε/k_B; drops to about 1.0 once the starting grid melts
MASS_B = 1.0  # mass of a B particle, in units of A's mass (A has 1)
SIGMA_B = 1.0  # diameter of a B particle, in σ (A has 1)
DT = 0.005  # time step, in τ
N_STEPS = 30000  # number of time steps (run length = N_STEPS * DT, in τ)
SAVE_EVERY = 10  # store a frame every this many steps (often enough for smooth playback in play.py)
N_PROFILES = 5  # how many moments to show in the figure
SEED = 1  # random seed: the same seed gives exactly the same run
PLAY = False  # True: play the run as an animation at the end
# ---- End of parameters ----

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mdsim.analysis import concentration_profile
from mdsim.simulation import run
from mdsim.system import make_two_species
from mdsim.viz import play_trajectory, save_figure

species_a = dict(sigma=1.0, epsilon=1.0, mass=1.0)
species_b = dict(sigma=SIGMA_B, epsilon=1.0, mass=MASS_B)
system = make_two_species(N_X, N_Y, DENSITY, START_TEMPERATURE, SEED,
                          species_a, species_b, layout="left_right", aspect=ASPECT)
trajectory = run(system, N_STEPS, DT, compute_forces=compute_forces, save_every=SAVE_EVERY)

Path("output").mkdir(exist_ok=True)
trajectory.save("output/mixing.npz")
print("Saved output/mixing.npz")
second_half = len(trajectory.times) // 2
print(f"Mean temperature (second half of the run): {np.mean(trajectory.temperature[second_half:]):.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
frames = np.linspace(0, len(trajectory.times) - 1, N_PROFILES).round().astype(int)
colors = plt.cm.viridis(np.linspace(0, 0.9, N_PROFILES))
for frame, color in zip(frames, colors):
    centers, fraction_a = concentration_profile(trajectory.positions[frame], trajectory.species, trajectory.box)
    ax.plot(centers, fraction_a, "o-", color=color, label=f"t = {trajectory.times[frame]:.0f}")
# The box is periodic: its right edge touches its left edge. So A and B meet
# twice, in the middle and at the edges, and the profile smooths out at both places.
ax.set_xlabel("x (σ)")
ax.set_ylabel("fraction of A particles")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Mixing: fraction of A across the box")
ax.legend()
save_figure(fig, "mixing.png")

if PLAY:
    playback_fig, anim = play_trajectory(trajectory)
plt.show()
