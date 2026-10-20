
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

G = 6.67430e-11
M_SUN = 1.98847e30
DAY = 86400.0
MILLION_KM = 1e9  # meters

@dataclass(frozen=True)
class Planet:
    name: str
    mass_1e24kg: float
    perihelion_mkm: float
    perihelion_kms: float
    period_days: float 







PLANETS = [
    Planet('Mercury', 0.33011, 46.00, 58.98, 87.97),
    Planet('Venus', 4.8675, 107.48, 35.26, 224.70),
    Planet('Earth', 5.9724, 147.09, 30.29, 365.26),
    Planet('Mars', 0.64171, 206.62, 26.50, 686.98),
    Planet('Jupiter', 1898.2, 740.52, 13.72, 4332.59),
    Planet('Saturn', 568.34, 1352.56, 10.18, 10759.22),
    Planet('Uranus', 86.813, 2741.30, 7.11, 30685.4),
    Planet('Neptune', 102.41, 4444.45, 5.50, 60189.0),
]




# aphelion (million km), aphelion speed (km/s), semimajor axis (million km), period (days), eccentricity, mean speed (km/s).
REFERENCE = {
    'Mercury': (69.82, 38.86, 57.91, 87.97, 0.2056, 47.36),
    'Venus': (108.94, 34.79, 108.21, 224.70, 0.0067, 35.02),
    'Earth': (152.10, 29.29, 149.60, 365.26, 0.0167, 29.78),
    'Mars': (249.23, 21.97, 227.92, 686.98, 0.0935, 24.07),
    'Jupiter': (816.62, 12.44, 778.57, 4332.59, 0.0489, 13.06),
    'Saturn': (1514.50, 9.09, 1433.53, 10759.22, 0.0565, 9.68),
    'Uranus': (3003.63, 6.49, 2872.46, 30685.4, 0.0457, 6.80),
    'Neptune': (4545.67, 5.37, 4495.06, 60189.0, 0.0113, 5.43),
}
METHODS = {'1': 'euler', '2': 'cromer', '3': 'rk2', '4': 'all'}
METHOD_LABELS = {'euler': 'Euler', 'cromer': 'Euler-Cromer', 'rk2': 'Midpoint (RK2)'}
COLORS = {'Mercury': 'green', 'Venus': 'deeppink', 'Earth': 'blue', 'Mars': 'red', 'Jupiter': 'black', 'Saturn': 'saddlebrown', 'Uranus': 'purple', 'Neptune': 'darkcyan'}
FIELDS = ['planet', 'method', 'aphelion_distance_million_km', 'aphelion_speed_km_s', 'semimajor_axis_million_km', 'orbital_period_days', 'eccentricity', 'mean_orbital_speed_km_s', 'relative_error_percent', 'period_detected']


def acceleration(r):
    distance = np.linalg.norm(r)
    if distance <= 0:
        raise ValueError('Planet reached the Sun; decrease the time step.')
    return -G * M_SUN * r / distance**3


def euler_step(r, v, dt):
    a = acceleration(r)
    return r + v * dt, v + a * dt


def cromer_step(r, v, dt):
    v_new = v + acceleration(r) * dt
    return r + v_new * dt, v_new


def rk2_step(r, v, dt):
    a = acceleration(r)
    r_mid = r + 0.5 * dt * v
    v_mid = v + 0.5 * dt * a
    return r + dt * v_mid, v + dt * acceleration(r_mid)


STEP_FUNCTIONS = {'euler': euler_step, 'cromer': cromer_step, 'rk2': rk2_step}


def select_method():
    print('\nSelect numerical method:')
    print('1. Euler\n2. Euler-Cromer\n3. Midpoint (RK2)\n4. All methods')
    while True:
        choice = input('Enter your choice (1-4) [4]: ').strip() or '4'
        if choice in METHODS:
            return list(STEP_FUNCTIONS) if METHODS[choice] == 'all' else [METHODS[choice]]
        print('Invalid choice. Enter 1, 2, 3 or 4.')


def select_planets():
    print('\nSelect planet:')
    for i, planet in enumerate(PLANETS, 1):
        print(f'{i}. {planet.name}')
    print('9. All planets')
    while True:
        choice = input('Enter your choice (1-9) [9]: ').strip() or '9'
        if choice == '9':
            return PLANETS
        if choice in {str(i) for i in range(1, 9)}:
            return [PLANETS[int(choice) - 1]]
        print('Invalid choice. Enter 1-9.')


def select_steps():
    while True:
        raw = input('Enter number of steps per orbit [100000]: ').strip()
        try:
            steps = int(raw) if raw else 100000
            if steps >= 100:
                return steps
        except ValueError:
            pass
        print('Enter an integer >= 100.')


def simulate_orbit(planet, method, steps):
    # Integrate for ~1.1 reference periods, so a completed orbit can be detected.
    dt = planet.period_days * DAY / steps
    count = math.ceil(1.1 * steps)
    positions = np.empty((count + 1, 2), dtype=float)
    velocities = np.empty((count + 1, 2), dtype=float)
    positions[0] = [planet.perihelion_mkm * MILLION_KM, 0.0]
    velocities[0] = [0.0, planet.perihelion_kms * 1000.0]
    step = STEP_FUNCTIONS[method]
    for i in range(count):
        positions[i + 1], velocities[i + 1] = step(positions[i], velocities[i], dt)
        if not np.all(np.isfinite(positions[i + 1])) or not np.all(np.isfinite(velocities[i + 1])):
            raise ValueError(f'Nonfinite orbit for {planet.name} ({method}); increase steps.')
    return positions, velocities, dt


def find_period(positions, velocities, dt):
    y = positions[:, 1]
    for i in range(1, len(y) - 1):
        if y[i] <= 0 < y[i + 1] and velocities[i + 1, 1] > 0 and positions[i + 1, 0] > 0:
            fraction = -y[i] / (y[i + 1] - y[i])
            return (i + fraction) * dt
    return None


def calculate_orbital_parameters(positions, velocities, dt, planet, method):
    period = find_period(positions, velocities, dt)
    if period is None:
        # Period not measurable: explicitly mark it as missing, not as reference value.
        end_index = len(positions)
    else:
        end_index = min(len(positions), math.ceil(period / dt) + 1)
    r = np.linalg.norm(positions[:end_index], axis=1)
    speed = np.linalg.norm(velocities[:end_index], axis=1)
    apo_idx = int(np.argmax(r))
    r_apo = float(r[apo_idx] / MILLION_KM)
    v_apo = float(speed[apo_idx] / 1000.0)
    r_peri = float(np.min(r) / MILLION_KM)
    semimajor = (r_peri + r_apo) / 2
    eccentricity = (r_apo - r_peri) / (r_apo + r_peri)
    # Time average over the detected full orbit (trapezoidal integration).
    if period is None:
        mean_speed = float('nan')
    else:
        n = int(period // dt)
        frac = period / dt - n
        partial_speed = speed[n] + frac * (speed[n + 1] - speed[n])
        integral = float(np.trapezoid(speed[:n + 1], dx=dt)) if hasattr(np, 'trapezoid') else float(np.trapz(speed[:n + 1], dx=dt))
        integral += 0.5 * (speed[n] + partial_speed) * (frac * dt)
        mean_speed = integral / period / 1000.0
    values = [r_apo, v_apo, semimajor, period / DAY if period else float('nan'), eccentricity, mean_speed]
    reference = REFERENCE[planet.name]
    errors = [abs(a - b) / abs(b) * 100 for a, b in zip(values, reference) if math.isfinite(a) and b != 0]
    return dict(zip(FIELDS, [planet.name, method, *values, float(np.mean(errors)) if errors else float('nan'), period is not None]))


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_orbits(results, path, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    for (name, method), (positions, velocities, dt) in results.items():
        coords = positions / MILLION_KM
        ax.plot(coords[:, 0], coords[:, 1], label=f'{name} ({METHOD_LABELS[method]})', color=COLORS[name], linewidth=1.1, alpha=0.85, linestyle={'euler': ':', 'cromer': '--', 'rk2': '-'}[method])
    ax.scatter([0], [0], s=100, color='orange', label='Sun', zorder=5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (million km)')
    ax.set_ylabel('y (million km)')
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path, rows, steps):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('PROJECT 3.I - PLANETARY MOTION\n')
        f.write(f'Steps per reference orbit: {steps}\n')
        f.write('Orbital parameters are computed from the simulated trajectory.\n')
        f.write('Reference Table 1 values are used only for error comparison.\n')
        f.write('Reference periods set the numerical time step, not output periods.\n')
        f.write('Methods used: ' + ', '.join(METHOD_LABELS[m] for m in dict.fromkeys(r['method'] for r in rows)) + '\n')
        f.write('Relative error: arithmetic mean of available six parameter percent errors.\n')
        f.write('Missing periods and mean speeds are reported as nan.\n\n')
        for row in rows:
            f.write(f"Planet: {row['planet']} | Method: {METHOD_LABELS[row['method']]} | Steps: {steps}\n")
            for field in FIELDS[2:]:
                f.write(f'  {field}: {row[field]}\n')
            f.write('\n')
        if len({r['method'] for r in rows}) > 1:
            f.write('Best method per planet (smallest available mean relative error):\n')
            for planet in dict.fromkeys(r['planet'] for r in rows):
                valid = [r for r in rows if r['planet'] == planet and math.isfinite(r['relative_error_percent']) and r['period_detected']]
                if valid:
                    best = min(valid, key=lambda r: r['relative_error_percent'])
                    f.write(f"  {planet}: {METHOD_LABELS[best['method']]} ({best['relative_error_percent']:.5f}%)\n")


def main():
    print('PROJECT 3.I - PLANETARY MOTION')
    methods = select_method()
    planets = select_planets()
    steps = select_steps()
    results_root = Path(__file__).resolve().parent / 'results'
    planet_tag = planets[0].name if len(planets) == 1 else 'AllPlanets'
    method_tag = methods[0].upper() if len(methods) == 1 else 'AllMethods'
    run_tag = f'{planet_tag}_{method_tag}_N{steps}'
    run_folder = f'Project3I_{run_tag}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output = results_root / run_folder
    output.mkdir(parents=True, exist_ok=False)
    log = (output / 'run_log.txt').open('w', encoding='utf-8')
    log.write(f'Project 3.I | Planets: {planet_tag} | Methods: {method_tag} | N={steps}\n')
    rows = []
    results = {}
    for method in methods:
        for planet in planets:
            log.write(f'Simulating {planet.name} | Method: {METHOD_LABELS[method]} | Steps: {steps} | dt: {planet.period_days * DAY / steps:.6f} s\n')
            try:
                positions, velocities, dt = simulate_orbit(planet, method, steps)
                row = calculate_orbital_parameters(positions, velocities, dt, planet, method)
            except (ValueError, OverflowError, FloatingPointError) as exc:
                log.write(f'  Failed: {exc}\n')
                continue
            rows.append(row)
            results[(planet.name, method)] = (positions, velocities, dt)
            log.write(f"  Aphelion: {row['aphelion_distance_million_km']:.4f} million km; period: {row['orbital_period_days']:.4f} days; mean error: {row['relative_error_percent']:.4f}%\n")
    if not rows:
        log.write('No successful simulations. Try increasing the number of steps.\n')
        log.close()
        print('Simulation failed. See results/run_log.txt.')
        return
    saved_files = []
    def save_csv(filename, data):
        write_csv(output / filename, data)
        saved_files.append(filename)

    save_csv(f'Project3I_Parameters_{run_tag}.csv', rows)
    for method in methods:
        method_rows = [row for row in rows if row['method'] == method]
        if method_rows and len(methods) > 1:
            save_csv(f'Project3I_Parameters_{planet_tag}_{method.upper()}_N{steps}.csv', method_rows)

    report_name = f'Project3I_Report_{run_tag}.txt'
    write_report(output / report_name, rows, steps)
    saved_files.append(report_name)

    for method in methods:
        subset = {key: val for key, val in results.items() if key[1] == method}
        if not subset:
            continue
        method_label = METHOD_LABELS[method]
        plot_name = f'Project3I_Orbit_{planet_tag}_{method.upper()}_N{steps}.png'
        plot_orbits(subset, output / plot_name,
                    f'Project 3.I | {planet_tag} | {method_label} | N={steps}')
        saved_files.append(plot_name)
        if len(planets) > 1:
            for group, names in [('InnerPlanets', {'Mercury', 'Venus', 'Earth', 'Mars'}),
                                 ('OuterPlanets', {'Jupiter', 'Saturn', 'Uranus', 'Neptune'})]:
                group_data = {key: val for key, val in subset.items() if key[0] in names}
                if group_data:
                    name = f'Project3I_Orbit_{group}_{method.upper()}_N{steps}.png'
                    plot_orbits(group_data, output / name,
                                f'Project 3.I | {group} | {method_label} | N={steps}')
                    saved_files.append(name)
    log.write(f'\nSaved {len(rows)} simulations to {output.resolve()}\nGenerated files:\n')
    for name in saved_files:
        log.write(f'  {name}\n')
    log.close()
    print(f'Completed. Results: {output}')


if __name__ == '__main__':
    main()
