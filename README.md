# mdsim

A small two-dimensional molecular dynamics (MD) simulation of a Lennard-Jones
fluid, written for a workshop on Git, fast Python and AI-assisted coding.

Particles interact through the Lennard-Jones potential (they repel at short
distance and attract a little at medium distance) in a periodic box. The code
moves them with Newton's equations and measures energy, temperature and how two
kinds of particles mix. There are four versions of the force calculation, from pure
Python to Numba, so you can compare their speed.

Everything is in reduced Lennard-Jones units: σ = ε = m = k_B = 1 for the
particles of species A.

## Installation

You need Python 3.12 with NumPy, Numba and matplotlib. The setup guide uses
conda (Miniforge); if you followed it, you already have the `mdsim`
environment. Otherwise create it with:

```
conda create -n mdsim -y python=3.12 numpy=2.5.4 numba=0.68.1 matplotlib-base=3.11.2 pytest=9.2.0
```

Then open a terminal in this repository's folder and run:

```
conda activate mdsim
pip install -e .
```

**Without conda** you can use a plain virtual environment instead. This does
not work on Intel Macs, where pip has no Numba package; use conda there.

**Windows** (PowerShell):

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

**macOS / Linux**:

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Check that everything works:

```
pytest
```

Until you have solved the workshop problems, the tests in `tests/test_problem_*.py`
fail (that is expected); all other tests should pass.

## Workshop problems

Some parts of the code are missing: you write them during the workshop, one
problem at a time. Every place you need to change starts with a comment
`# ZADATAK N:` (N = problem number) with short instructions. Until you write
your part, the code still runs, it just does nothing useful there (for
example the kinetic energy is zero, or a force version returns zero forces).

| Problem | What you write | Where | Test |
|---|---|---|---|
| 2 | Kinetic and total energy, energy plot | `mdsim/observables.py`, `mdsim/simulation.py`, `mdsim/viz.py` | `pytest tests/test_problem_2.py` |
| 3a | Forces with NumPy | `mdsim/forces_numpy.py` | `pytest tests/test_problem_3a.py` |
| 3b | Forces with Numba | `mdsim/forces_numba.py`, `examples/run_live.py` | `pytest tests/test_problem_3b.py` |

A problem is done when its test file passes. When the next problem starts,
the solution of the previous one appears in the folder `solutions/`. To get
it and put it into your code (here for Problem 2):

```
git pull
python apply_solution.py 2
```

This replaces your version of the files that problem touches with the
solution (it asks first if you have changes that are not committed yet).
Then run the problem's test and commit.

## Running the scripts

Run every script from the repository folder, for example
`python examples/energy_plot.py` (or open it in VS Code and press Run).
Each script starts with a block of parameters that you can edit. Figures are
shown in a window and also saved to the folder `output/`.

| Script | Workshop part | What it shows |
|---|---|---|
| `examples/run_live.py` | 1 | The fluid moving, live, with an energy plot. Keys: space = pause, r = restart, q = quit, f = fullscreen. |
| `examples/energy_plot.py` | 2 | Kinetic, potential and total energy of a run; the total energy stays constant. |
| `examples/python_internals.py` | 4 | Python bytecode and memory use of Python numbers vs. NumPy. |
| `examples/profile_forces.py` | 4 | Profiling: the force calculation takes almost all the time. |
| `examples/benchmark.py` | 4 | Speed of the four force versions (versions you haven't written yet are skipped). |
| `examples/mixing.py` | end | Two species mixing: the concentration profile smooths out. Works only once Problem 3b is done. |
| `examples/play.py` | any | Plays back a saved run: `python examples/play.py output/mixing.npz`. Add a number to play slower, e.g. `... output/mixing.npz 100` (100 ms between pictures). |

## The code

| File | What it does |
|---|---|
| `mdsim/system.py` | The particles (`System`) and starting states on a grid |
| `mdsim/forces_python.py` | Forces in pure Python (every pair twice) |
| `mdsim/forces_newton3.py` | Forces in pure Python, each pair once (Newton's third law) |
| `mdsim/forces_numpy.py` | Forces with NumPy, no Python loops |
| `mdsim/forces_numba.py` | Forces compiled with Numba |
| `mdsim/integrators.py` | Velocity Verlet, the rule that moves the particles one time step |
| `mdsim/observables.py` | Kinetic energy, temperature, momentum |
| `mdsim/simulation.py` | The main loop (`run`) and saved runs (`Trajectory`) |
| `mdsim/analysis.py` | Concentration profile (mixing) |
| `mdsim/viz.py` | Drawing, live viewer, playback, energy plot |

## Measured results

Measured on an Apple M5 laptop (macOS, Python 3.12, NumPy 2.5.4, Numba 0.68.1).
A typical school laptop is a few times slower.

**Temperature.** The crystal melts within the first fraction of a time unit and
the temperature drops. With `START_TEMPERATURE = 1.3` at density 0.7 the fluid
settles at T ≈ 0.99.

**Speed of the force versions** (`benchmark.py`, density 0.7, median of 3 calls).
The N = 6400 row needs `N_VALUES` extended and `MAX_N_NUMPY = 6400`; NumPy then
uses several GB of memory.

| N | pure Python | Newton 3 | NumPy | Numba |
|---:|---:|---:|---:|---:|
| 100 | 2.4 ms | 1.2 ms (2x) | 0.32 ms (8x) | 0.0097 ms (251x) |
| 400 | 38 ms | 19 ms (2x) | 5.2 ms (7x) | 0.11 ms (333x) |
| 1600 | 607 ms | 302 ms (2x) | 89 ms (7x) | 1.7 ms (361x) |
| 6400 | – | – | 1505 ms | 26 ms |
