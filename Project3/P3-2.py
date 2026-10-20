# (1) Read 8 txt files
# (2) Initial positions and velocities
# (3) Calculate gravitational accelerations
# Apply RK2 (or othere methods) to all 8 planets
# Repeat for 165 years
# Plot system
# Save results








from pathlib import Path
from datetime import datetime
import csv
import sys
import math
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = 6.67430e-11
M_SUN = 1.98847e30
DAY = 86400.0
YEAR = 365.25 * DAY
KM = 1000.0
MILLION_KM = 1e9
NAMES = ('Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune')
# Used only when a HORIZONS file has no parseable mass field.
FALLBACK_MASSES = np.array([0.33011, 4.8675, 5.9724, 0.64171, 1898.2,
                            568.34, 86.813, 102.41]) * 1e24
PERIOD_DAYS = np.array([87.97, 224.70, 365.26, 686.98, 4332.59,
                        10759.22, 30685.4, 60189.0])
METHODS = {'1': 'Euler', '2': 'EulerCromer', '3': 'RK2', '4': 'All'}
COLORS = ('gray', 'orange', 'royalblue', 'firebrick', 'peru', 'goldenrod', 'teal', 'navy')
MOL_TYPES = ('H', 'Mn', 'Fe', 'O', 'K', 'Zn', 'N', 'F')
RUN_LOG = None

def log(message):
    if RUN_LOG is not None:
        RUN_LOG.write(str(message) + '\n')
        RUN_LOG.flush()

NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?'


def choose(prompt, options, default):
    while True:
        print('\n' + prompt)
        for key, value in options.items():
            print(f'{key}. {value}')
        answer = input(f'Choice [{default}]: ').strip() or default
        if answer in options:
            return answer
        print('Invalid choice. Try again.')


def positive_number(prompt, default, cast=float):
    while True:
        answer = input(f'{prompt} [{default}]: ').strip()
        try:
            value = cast(answer) if answer else cast(default)
            if math.isfinite(value) and value > 0:
                return value
        except (ValueError, OverflowError):
            pass
        print('Enter a positive number.')


def parse_mass(text):
    """Read mass values such as 'Mass (10^23 kg) = 3.302' or 'Mass = 5.97E24 kg'."""
    for line in text.splitlines():
        if not re.search(r'\bMass\b', line, re.I) or 'ratio' in line.lower():
            continue
        # Some HORIZONS mass lines contain multiple columns; read only the Mass field.
        match = re.search(r'\bMass\s*(?:\(([^)]*)\))?\s*=\s*(' + NUMBER + r')', line, re.I)
        if not match:
            continue
        units = (match.group(1) or '').replace(' ', '')
        mass = float(match.group(2).replace('D', 'E').replace('d', 'e'))
        power = re.search(r'10\s*\^\s*([+-]?\d+)', units)
        if power:
            mass *= 10 ** int(power.group(1))
        if 'g' in units.lower() and 'kg' not in units.lower():
            mass *= 0.001
        if mass > 1e20:
            return mass
    return None


def read_horizons(path, name, fallback_mass):
    """Read first state between $$SOE and $$EOE; units km and km/s."""
    text = path.read_text(encoding='utf-8', errors='replace')
    if '$$SOE' not in text or '$$EOE' not in text:
        raise ValueError(f'{path}: missing $$SOE/$$EOE markers')
    header, rest = text.split('$$SOE', 1)
    block = rest.split('$$EOE', 1)[0]
    values = {}
    for key, raw in re.findall(r'\b(VX|VY|VZ|X|Y|Z)\s*=\s*(' + NUMBER + r')', block, re.I):
        values.setdefault(key.upper(), float(raw.replace('D', 'E').replace('d', 'e')))
    required = ('X', 'Y', 'Z', 'VX', 'VY', 'VZ')
    if any(key not in values for key in required):
        raise ValueError(f'{path}: missing position or velocity component')
    # These example files are heliocentric; a barycentric state would need a Sun state.
    center = re.search(r'Center\s+(?:body\s+)?(?:name|target)\s*:\s*([^\n]+)', header, re.I)
    if center and 'sun' not in center.group(1).lower() and not re.search(r'\b10\b', center.group(1)):
        raise ValueError(f'{path}: non-Sun coordinate center: {center.group(1).strip()}')
    mass = parse_mass(header)
    if mass is None:
        mass = float(fallback_mass)
        log(f'  WARNING: {name} mass not parsed; using documented fallback {mass:.6e} kg')
    position = np.array([values[k] for k in ('X', 'Y', 'Z')]) * KM
    velocity = np.array([values[k] for k in ('VX', 'VY', 'VZ')]) * KM
    return mass, position, velocity


def load_initial_conditions(directory):
    masses, positions, velocities = [], [], []
    for name, fallback in zip(NAMES, FALLBACK_MASSES):
        path = directory / f'{name}.txt'
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}. Put all eight files in the data folder.')
        mass, r, v = read_horizons(path, name, fallback)
        masses.append(mass)
        positions.append(r)
        velocities.append(v)
        log(f'  {name:8s} | r={np.linalg.norm(r)/MILLION_KM:9.3f} million km'
              f' | v={np.linalg.norm(v)/KM:7.3f} km/s')
    return np.array(masses), np.array(positions), np.array(velocities)


def accelerations(positions, masses):
    """Sun gravity plus seven other planets, computed at one shared time."""
    norms = np.linalg.norm(positions, axis=1)
    if np.any(norms <= 0):
        raise ValueError('Planet reached Sun')
    a = -G * M_SUN * positions / norms[:, None]**3
    for i in range(8):
        delta = positions - positions[i]  # delta[j] = r_j - r_i
        d = np.linalg.norm(delta, axis=1)
        d[i] = np.inf
        if np.any(d == 0):
            raise ValueError('Planet collision')
        a[i] += G * np.sum((masses / d**3)[:, None] * delta, axis=0)
    return a


def advance(r, v, masses, dt, method):
    a = accelerations(r, masses)
    if method == 'Euler':
        return r + dt*v, v + dt*a
    if method == 'EulerCromer':
        v_next = v + dt*a
        return r + dt*v_next, v_next
    # Midpoint RK2: all eight planets use the SAME midpoint snapshot.
    r_mid = r + 0.5*dt*v
    v_mid = v + 0.5*dt*a
    return r + dt*v_mid, v + dt*accelerations(r_mid, masses)


def simulate(masses, r0, v0, years, dt_days, method, max_frames=2500):
    total_time = years * YEAR
    steps = math.ceil(total_time / (dt_days * DAY))
    dt = total_time / steps  # reach requested final time exactly
    stride = max(1, math.ceil(steps / (max_frames - 1)))
    r, v = r0.copy(), v0.copy()
    times, locations = [0.0], [r.copy()]
    for i in range(1, steps + 1):
        r, v = advance(r, v, masses, dt, method)
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
            raise FloatingPointError(f'{method} diverged at step {i}')
        if i % stride == 0 or i == steps:
            times.append(i * dt / YEAR)
            locations.append(r.copy())
        if i % max(1, steps // 10) == 0:
            log(f'  {method}: {i}/{steps} steps ({100*i/steps:.0f}%)')
    return np.array(times), np.array(locations), steps, dt / DAY


def save_csv(path, times, locations, method):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'time_years', 'planet', 'x_million_km',
                         'y_million_km', 'z_million_km'])
        for t, frame in zip(times, locations):
            for name, r in zip(NAMES, frame):
                writer.writerow([method, f'{t:.9f}', name, *(f'{x/MILLION_KM:.9f}' for x in r)])


def plot_orbits(path, locations, method, years, group=None):
    indexes = list(range(8)) if group is None else list(group)
    fig, ax = plt.subplots(figsize=(9, 9))
    for i in indexes:
        ax.plot(locations[:, i, 0] / MILLION_KM,
                locations[:, i, 1] / MILLION_KM,
                label=NAMES[i], lw=0.8, color=COLORS[i])
    ax.scatter(0, 0, s=90, c='gold', edgecolors='orange', label='Sun', zorder=10)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (million km)')
    ax.set_ylabel('Y (million km)')
    ax.set_title(f'Project 3.II | {method} | {years:g} years | XY projection')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_mol2_frame(f, name, points, labels, atom_types, bonds=None):
    """Coordinates use 10^9 meters (million km), as in the assignment example."""
    bonds = bonds or []
    f.write('@<TRIPOS>MOLECULE\n')
    f.write(f'{name}\n{len(points)} {len(bonds)} 1\nSMALL\nNO_CHARGES\n\n')
    f.write('@<TRIPOS>ATOM\n')
    for i, (point, label, atom_type) in enumerate(zip(points, labels, atom_types), 1):
        x, y, z = point / MILLION_KM
        f.write(f'{i:6d} {label:<9s} {x: .8E} {y: .8E} {z: .8E} '
                f'{atom_type:3s} 1 RES1 0.000\n')
    if bonds:
        f.write('@<TRIPOS>BOND\n')
        for k, (a, b) in enumerate(bonds, 1):
            f.write(f'{k:6d} {a:6d} {b:6d} 1\n')


def write_trajectory_mol2(path, times, locations, method):
    with path.open('w', encoding='utf-8') as f:
        for t, frame in zip(times, locations):
            points = np.vstack((np.zeros(3), frame))
            write_mol2_frame(f, f'Project3II_{method}_t{t:.5f}yr', points,
                             ['Sun', *NAMES], ['S', *MOL_TYPES])


def write_orbit_mol2(path, positions, name, atom_type):
    # A single molecule with connected sample points, useful for PyMOL.
    if len(positions) < 2:
        return
    bonds = [(i, i+1) for i in range(1, len(positions))]
    with path.open('w', encoding='utf-8') as f:
        write_mol2_frame(f, f'ORBIT OF {name}', positions,
                         [name] * len(positions), [atom_type] * len(positions), bonds)


def main():
    print('PROJECT 3.II - EIGHT-PLANET N-BODY SIMULATION')
    choice = choose('Select numerical method:', METHODS, '3')
    methods = ['Euler', 'EulerCromer', 'RK2'] if choice == '4' else [METHODS[choice]]
    years = positive_number('Simulation duration in years', 165.0)
    dt_days = positive_number('Time step in days (0.25 recommended)', 0.25)
    # Fixed project layout: Project3/files contains the eight HORIZONS inputs.
    # All generated files are automatically written under Project3/results.
    project_dir = Path(__file__).resolve().parent
    folder = project_dir / 'files'
    results_root = project_dir / 'results'
    method_tag = methods[0] if len(methods) == 1 else 'AllMethods'
    run_folder = (f'Project3II_AllPlanets_{method_tag}_{years:g}Years_'
                  f'dt{dt_days:g}Days_{datetime.now():%Y%m%d_%H%M%S_%f}')
    output = results_root / run_folder
    output.mkdir(parents=True, exist_ok=False)
    global RUN_LOG
    RUN_LOG = (output / 'run_log.txt').open('w', encoding='utf-8')
    log(f'Input folder: {folder}')
    log('Loading NASA HORIZONS initial conditions...')
    try:
        masses, r0, v0 = load_initial_conditions(folder)
    except (OSError, ValueError) as error:
        log(f'INPUT ERROR: {error}')
        RUN_LOG.close()
        print(f'Input error: {error}. See {output / "run_log.txt"}')
        return
    log(f'Output folder: {output}')
    log(f'Simulating {years:g} years, nominal dt={dt_days:g} days.')
    for method in methods:
        tag = f'Project3II_AllPlanets_{method}_{years:g}Years_dt{dt_days:g}Days'
        log(f'Running {method}...')
        try:
            times, locations, steps, actual_dt = simulate(
                masses, r0, v0, years, dt_days, method)
        except (ValueError, FloatingPointError) as error:
            log(f'SIMULATION ERROR: {error}')
            continue
        save_csv(output / f'{tag}_Positions.csv', times, locations, method)
        plot_orbits(output / f'{tag}_OrbitsXY.png', locations, method, years)
        plot_orbits(output / f'{tag}_InnerOrbitsXY.png', locations, method, years, range(4))
        plot_orbits(output / f'{tag}_OuterOrbitsXY.png', locations, method, years, range(4, 8))
        write_trajectory_mol2(output / f'{tag}_Trajectory.mol2', times, locations, method)
        # First complete orbit approximately, using the reference sidereal period.
        for i, name in enumerate(NAMES):
            cutoff = min(years, PERIOD_DAYS[i] / 365.25)
            orbit = locations[times <= cutoff + 1e-10, i]
            if len(orbit) > 1:
                write_orbit_mol2(output / f'{tag}_{name}_Orbit.mol2',
                                 orbit, name, MOL_TYPES[i])
        with (output / f'{tag}_Report.txt').open('w', encoding='utf-8') as f:
            f.write('PROJECT 3.II: Eight-planet N-body model\n')
            f.write(f'Method: {method}\nDuration: {years:g} years\n')
            f.write(f'Steps: {steps}\nActual dt: {actual_dt:.10f} days\n')
            f.write(f'Frames saved: {len(times)}\n')
            f.write('Sun fixed at origin; heliocentric J2000 state vectors assumed.\n')
            f.write('Sun gravity + mutual gravity of all eight planets.\n')
            f.write('No relativistic effects or other Solar System bodies.\n')
            for i, name in enumerate(NAMES):
                f.write(f'{name}: mass={masses[i]:.9e} kg, '
                        f'final distance={np.linalg.norm(locations[-1, i])/MILLION_KM:.6f} million km\n')
        log(f'Completed {method}: {steps} steps, {len(times)} frames.')
        log(f'Saved outputs with prefix: {tag}')
    log('Finished. Open the results folder for plots, CSV, TXT and PyMOL MOL2 files.')
    RUN_LOG.close()
    print(f'Finished. Results: {output}')


if __name__ == '__main__':
    main()
