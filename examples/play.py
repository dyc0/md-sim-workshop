"""Any part: play back a saved run.

Usage (from the repository folder):
    python examples/play.py output/mixing.npz
    python examples/play.py output/mixing.npz 100    (100 ms between pictures: slower)
"""

# ---- Parameters (edit these) ----
INTERVAL_MS = 30  # milliseconds between pictures: a bigger number plays slower
EVERY = 1  # show every frame (2 = every second frame, ...)
# ---- End of parameters ----

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from mdsim.simulation import Trajectory
from mdsim.viz import play_trajectory, save_figure

USAGE = "Usage: python examples/play.py output/<file>.npz [milliseconds between pictures]"

if len(sys.argv) < 2:
    print(USAGE)
    sys.exit(1)
path = Path(sys.argv[1])
if not path.is_file():
    print(f"File not found: {path}")
    print(USAGE)
    sys.exit(1)

# An optional second argument replaces INTERVAL_MS, e.g. 100 for slower playback.
interval_ms = INTERVAL_MS
if len(sys.argv) >= 3:
    if not (sys.argv[2].isdigit() and int(sys.argv[2]) > 0):
        print(f"The time between pictures must be a positive whole number of milliseconds, got {sys.argv[2]!r}")
        print(USAGE)
        sys.exit(1)
    interval_ms = int(sys.argv[2])

trajectory = Trajectory.load(path)
print(f"Playing {path}: {len(trajectory.times)} frames, {interval_ms} ms between pictures")
fig, anim = play_trajectory(trajectory, interval_ms=interval_ms, every=EVERY)
save_figure(fig, f"play_{path.stem}.png")
plt.show()
