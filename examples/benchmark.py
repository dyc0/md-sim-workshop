"""Part 4: how fast is each force version?

Times one force calculation for every version at several particle numbers.
Look for: pure Python is slowest, using Newton's third law makes it about
2x faster, NumPy is much faster, Numba is fastest. The time grows like N^2
(every particle looks at every other particle).

This one may take a few minutes.
"""

# ---- Parameters (edit these) ----
# Units: σ = particle diameter (length), τ = time unit,
#        ε = attraction strength (energy, and temperature with k_B = 1).
N_VALUES = [100, 400, 1600]  # particle numbers (each must be a square number)
DENSITY = 0.7  # particles per σ²; sets the box size (area = N / DENSITY)
MAX_N_PYTHON = 1600  # skip the pure Python versions above this N (too slow)
MAX_N_NUMPY = 1600  # skip NumPy above this N (its N x N tables need too much memory)
N_CALLS = 3  # time this many calls and take the median
SEED = 1  # random seed for the starting velocities
# ---- End of parameters ----

import math
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from mdsim import forces_numba, forces_newton3, forces_numpy, forces_python
from mdsim.system import make_single_species
from mdsim.viz import save_figure

VERSIONS = {
    "pure Python": (forces_python.compute_forces, MAX_N_PYTHON),
    "Newton 3": (forces_newton3.compute_forces, MAX_N_PYTHON),
    "NumPy": (forces_numpy.compute_forces, MAX_N_NUMPY),
    "Numba": (forces_numba.compute_forces, max(N_VALUES)),
}


def make_system(n):
    side = math.isqrt(n)
    if side * side != n:
        sys.exit(f"N = {n} is not a square number; use values like 100, 400, 1600.")
    return make_single_species(side, side, DENSITY, 1.0, SEED)


def arguments(system):
    return system.positions, system.box, system.sigmas, system.epsilons


# 1. Check that every version gives the same answer as pure Python, before
#    timing anything. A version that doesn't (for example one you haven't
#    written yet) is left out of the timing.
system = make_system(min(N_VALUES))
reference, reference_energy = forces_python.compute_forces(*arguments(system))
for name, (compute_forces, max_n) in list(VERSIONS.items()):
    start = time.perf_counter()
    forces, energy = compute_forces(*arguments(system))
    # On the starting grid the forces are almost zero (by symmetry), so compare the energy too.
    if not (np.allclose(forces, reference, rtol=1e-8, atol=1e-10)
            and np.isclose(energy, reference_energy, rtol=1e-8)):
        print(f"{name} gives different forces than pure Python (not written yet?) - skipped.")
        del VERSIONS[name]
    elif name == "Numba":
        print(f"Numba first call (includes compiling): {time.perf_counter() - start:.2f} s")
print(f"Timing: {', '.join(VERSIONS)}\n")

# 2. Time every version at every N.
times = {name: {} for name in VERSIONS}
for n in N_VALUES:
    system = make_system(n)
    for name, (compute_forces, max_n) in VERSIONS.items():
        if n > max_n:
            continue
        call_times = []
        for _ in range(N_CALLS):
            start = time.perf_counter()
            compute_forces(*arguments(system))
            call_times.append(time.perf_counter() - start)
        times[name][n] = float(np.median(call_times))
        print(f"  {name:12s} N = {n:5d}: {times[name][n]:.4g} s")

# 3. Table: time per call and speedup against pure Python.
print()
print(f"{'version':12s} {'N':>6s} {'time per call (s)':>18s} {'speedup':>10s}")
for n in N_VALUES:
    for name in VERSIONS:
        if n not in times[name]:
            continue
        python_time = times["pure Python"].get(n)
        speedup = f"{python_time / times[name][n]:.0f}x" if python_time else "-"
        print(f"{name:12s} {n:6d} {times[name][n]:18.4g} {speedup:>10s}")

# 4. Plot time against N on log-log axes.
fig, ax = plt.subplots(figsize=(7, 5))
for name in VERSIONS:
    ns = sorted(times[name])
    ax.loglog(ns, [times[name][n] for n in ns], "o-", label=name)
ax.set_xlabel("number of particles N")
ax.set_ylabel("time per force calculation (s)")
ax.set_title("Force calculation: four versions")
ax.legend()
save_figure(fig, "benchmark.png")
plt.show()
