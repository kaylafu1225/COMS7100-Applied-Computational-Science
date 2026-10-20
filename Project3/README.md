# Project 3 — Planetary Motion

Two Python scripts simulate planetary motion using Euler, Euler-Cromer, and midpoint Runge-Kutta (RK2).

- **P3-1.py (Project 3.I):** Simulates each of the eight planets independently in 2D around a fixed Sun, calculates orbital parameters, and compares them with reference values.
- **P3-2.py (Project 3.II):** Simulates all eight planets together in 3D with the Sun's gravity and planet–planet gravitational interactions, starting from NASA JPL HORIZONS data.

## Installation

Requires Python 3, NumPy, and Matplotlib:

```bash
python -m pip install numpy matplotlib
```

## Project folder

```text
Project3/
├── P3-1.py
├── P3-2.py
├── README.md
├── files/
│   ├── Mercury.txt
│   ├── Venus.txt
│   ├── Earth.txt
│   ├── Mars.txt
│   ├── Jupiter.txt
│   ├── Saturn.txt
│   ├── Uranus.txt
│   └── Neptune.txt
└── results/                  # Automatically created
    ├── Project3I_.../
    └── Project3II_.../
```

The scripts locate folders relative to their own location. You **do not** need to enter input or output paths.

## Project 3.I: Independent planetary orbits

Run:

```bash
python P3-1.py
```

The program asks for:

1. **Numerical method:** `1` Euler, `2` Euler-Cromer, `3` Midpoint RK2, `4` All methods (default).
2. **Planet:** `1`–`8` for an individual planet or `9` for all planets (default).
3. **Number of steps per orbit:** default `100000`, minimum `100`. For example, `5000` means 5,000 integration steps over the reference period of each selected planet.

P3-1 uses perihelion starting positions and speeds and a fixed Sun. It does not include interactions between planets.

Saved results include CSV orbital-parameter tables, a TXT report with reference comparisons, PNG orbital plots, and `run_log.txt`.

## Project 3.II: Eight-planet N-body simulation

Run:

```bash
python P3-2.py
```

The program asks for:

1. **Numerical method:** `1` Euler, `2` Euler-Cromer, `3` RK2 (default), `4` All methods.
2. **Simulation duration in years:** total simulated time, default `165.0` years.
3. **Time step in days:** nominal time advanced per integration step, default `0.25` days (= 6 hours).

Press **Enter** to accept a default. Example quick test:

```text
Choice [3]: 3
Simulation duration in years [165.0]: 2
Time step in days (0.25 recommended) [0.25]: 0.25
```

This models 2 years of motion using approximately 0.25-day steps. For the full assignment, choose 165 years and 0.25 days. A time step of 23 days is too large for accurate inner-planet motion: Mercury takes about 88 days to orbit the Sun.

### Input data

P3-2 automatically reads the eight `files/<Planet>.txt` files. They must contain parseable NASA JPL HORIZONS Cartesian position and velocity data at a common initial epoch. The model assumes **heliocentric (Sun-centered)** vectors, with the Sun fixed at the origin; do not use barycentric vectors unchanged. Missing or malformed files produce an input error.

### Saved results

P3-2 writes CSV planetary positions, PNG XY orbital plots (all planets, inner planets, outer planets), MOL2 trajectory and planet-orbit files, a TXT report, and `run_log.txt`. The model omits relativity, moons, and other Solar System bodies.

## Results organization

**Every run creates a new timestamped subfolder under `results/`**. Previous runs are not overwritten.

```text
results/
├── Project3I_AllPlanets_AllMethods_N5000_20261019_184500_123456/
│   ├── run_log.txt
│   ├── Project3I_Parameters_...csv
│   ├── Project3I_Report_...txt
│   └── Project3I_Orbit_...png
└── Project3II_AllPlanets_RK2_165Years_dt0.25Days_20261019_190000_123456/
    ├── run_log.txt
    ├── Project3II_..._Positions.csv
    ├── Project3II_..._Report.txt
    ├── Project3II_..._OrbitsXY.png
    └── Project3II_..._Trajectory.mol2
```

The terminal is intentionally quiet: it shows only the interactive prompts and a final status message, with errors displayed when necessary. Detailed progress and results are saved in the output folder rather than printed to the terminal.

## Accuracy notes

Compare Euler, Euler-Cromer, and RK2 using the same initial conditions and numerical resolution. To check convergence, rerun with more steps (P3-1) or a smaller time step (P3-2). P3-1 and P3-2 model different physical systems, so their numerical results need not match exactly.
