from __future__ import annotations
import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


REFERENCE_ATOMS = [
    ('I1', 0.816559, 0.605285, 0.443681), 
    ('Br2', 0.184553, 0.732849, 0.422528), 
    ('Cl3', 0.538601, 0.187547, 0.425822), 
    ('N4', 0.483883, 0.646545, 0.488147), 
    ('N5', 0.378678, 0.442024, 0.561205), 
    ('N6', 0.649648, 0.394286, 0.452607), 
    ('B7', 0.343047, 0.60423, 0.576299), 
    ('B8', 0.6236, 0.524422, 0.547738), 
    ('B9', 0.510127, 0.380557, 0.407817), 
    ('H10', 0.701261, 0.302105, 0.537015), 
    ('H11', 0.719691, 0.398985, 0.328848), 
    ('H12', 0.487896, 0.44997, 0.255478), 
    ('H13', 0.472473, 0.735457, 0.535232), 
    ('H14', 0.492243, 0.677623, 0.345111), 
    ('H15', 0.609035, 0.488668, 0.714905), 
    ('H16', 0.306873, 0.618111, 0.736486), 
    ('H17', 0.290188, 0.433251, 0.530414), 
    ('H18', 0.391646, 0.378086, 0.690668)]


@dataclass
class CrystalData:
    comment: str
    cell: np.ndarray
    hkl: np.ndarray
    f_abs: np.ndarray
    A: np.ndarray
    B: np.ndarray
    M: np.ndarray
    M_inv: np.ndarray
    volume: float
    s_cart: np.ndarray

@dataclass
class Peak:
    cart: np.ndarray
    frac: np.ndarray
    rho: float
    grad_norm: float
    hessian_eigs: np.ndarray
    iterations: int
    method: str

def banner() -> None:
    print(' PROJECT 2 - Computational Crystallography')

def read_input(filename: Path) -> Tuple[str, np.ndarray, np.ndarray]:
    """input format on pages 14-15"""
    if not filename.exists():
        raise FileNotFoundError(f'Input file not found: {filename}')
    lines = filename.read_text(encoding='utf-8', errors='replace').splitlines()
    if len(lines) < 4:
        raise ValueError('Input file is too short.')
    comment = lines[0].strip()
    cell_tokens = lines[1].split()
    if len(cell_tokens) < 7 or cell_tokens[0].upper() != 'CELL':
        raise ValueError('Line 2 must have the form:\nCELL a b c alpha beta gamma')
    cell = np.array([float(x) for x in cell_tokens[1:7]], dtype=float)
    rows = []
    for line_no, line in enumerate(lines[3:], start=4):
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 6:
            continue
        try:
            h, k, l = (int(parts[0]), int(parts[1]), int(parts[2]))
            f_abs, A, B = map(float, parts[3:6])
        except ValueError:
            continue
        rows.append((h, k, l, f_abs, A, B))
    if not rows:
        raise ValueError('No valid h k l |F| A B reflection rows were found.')
    return (comment, cell, np.asarray(rows, dtype=float))

def build_cell(cell: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    a, b, c, alpha_deg, beta_deg, gamma_deg = cell
    alpha, beta, gamma = np.radians([alpha_deg, beta_deg, gamma_deg])
    ca, cb, cg = (np.cos(alpha), np.cos(beta), np.cos(gamma))
    sg = np.sin(gamma)
    if abs(sg) < 1e-14:
        raise ValueError('Invalid unit cell: sin(gamma) is too close to zero.')
    radicand = 1.0 + 2.0 * ca * cb * cg - ca ** 2 - cb ** 2 - cg ** 2
    if radicand <= 0.0:
        raise ValueError('Invalid unit-cell parameters: non-positive cell volume.')
    volume = a * b * c * math.sqrt(radicand)
    M = np.array([[a, b * cg, c * cb], [0.0, b * sg, c * (ca - cb * cg) / sg], [0.0, 0.0, c * math.sqrt(radicand) / sg]], dtype=float)
    M_inv = np.linalg.inv(M)
    metric = M.T @ M
    volume_from_M = abs(np.linalg.det(M))
    if not np.isclose(volume, volume_from_M, rtol=1e-10, atol=1e-10):
        raise RuntimeError('Internal error: cell-volume consistency check failed.')
    if np.linalg.det(metric) <= 0:
        raise RuntimeError('Internal error: metric tensor is not positive definite.')
    return (volume, M, M_inv)

def prepare_crystal(filename: Path) -> CrystalData:
    comment, cell, rows = read_input(filename)
    volume, M, M_inv = build_cell(cell)
    hkl = rows[:, 0:3]
    f_abs = rows[:, 3]
    A = rows[:, 4]
    B = rows[:, 5]
    s_cart = hkl @ M_inv
    return CrystalData(comment=comment, cell=cell, hkl=hkl, f_abs=f_abs, A=A, B=B, M=M, M_inv=M_inv, volume=volume, s_cart=s_cart)

def rho_grad_hessian(x_cart: np.ndarray, crystal: CrystalData):
    """
    Return rho, Cartesian gradient, and Cartesian Hessian.
    rho = 2/V sum[A cos(theta) + B sin(theta)]
    theta = 2*pi*s.x_cart
    """
    theta = 2.0 * np.pi * (crystal.s_cart @ x_cart)
    cth = np.cos(theta)
    sth = np.sin(theta)
    base = crystal.A * cth + crystal.B * sth
    first = -crystal.A * sth + crystal.B * cth
    pref = 2.0 / crystal.volume
    two_pi = 2.0 * np.pi
    rho = pref * np.sum(base)
    weights_g = pref * two_pi * first
    grad = crystal.s_cart.T @ weights_g
    weights_h = -pref * two_pi ** 2 * base
    H = crystal.s_cart.T @ (crystal.s_cart * weights_h[:, None])
    H = 0.5 * (H + H.T)
    return (float(rho), grad, H)

def density_only(x_cart: np.ndarray, crystal: CrystalData) -> float:
    theta = 2.0 * np.pi * (crystal.s_cart @ x_cart)
    return float(2.0 / crystal.volume * np.sum(crystal.A * np.cos(theta) + crystal.B * np.sin(theta)))

def limit_step(step: np.ndarray, max_step: float) -> np.ndarray:
    norm = np.linalg.norm(step)
    if norm > max_step and norm > 0:
        return step * (max_step / norm)
    return step

def newton_step(g: np.ndarray, H: np.ndarray) -> Optional[np.ndarray]:
    try:
        return np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        try:
            step, *_ = np.linalg.lstsq(H, -g, rcond=1e-12)
            return step
        except np.linalg.LinAlgError:
            return None

def ef_step(g: np.ndarray, H: np.ndarray) -> Optional[np.ndarray]:
    """
    Eigenvector Following for locating a maximum, following handout
    Eqs. (65)-(67).
    H v_i = b_i v_i
    F_i = v_i^T g
    lambda_EF = largest eigenvalue of B'
    h = -sum_i F_i/(b_i-lambda_EF) v_i
    """
    b, V = np.linalg.eigh(H)
    F = V.T @ g
    Bprime = np.zeros((4, 4), dtype=float)
    Bprime[:3, :3] = np.diag(b)
    Bprime[:3, 3] = F
    Bprime[3, :3] = F
    lambdas = np.linalg.eigvalsh(Bprime)
    lam = float(lambdas[-1])
    denom = b - lam
    tiny = 1e-12
    denom = np.where(np.abs(denom) < tiny, np.where(denom >= 0.0, tiny, -tiny), denom)
    coeff = -F / denom
    step = V @ coeff
    if not np.all(np.isfinite(step)):
        return None
    return step

def optimize_from_start(x0_cart: np.ndarray, crystal: CrystalData, method: str, grad_tol: float, max_iter: int, max_step: float) -> Optional[Peak]:
    x = np.array(x0_cart, dtype=float)
    for iteration in range(1, max_iter + 1):
        rho, g, H = rho_grad_hessian(x, crystal)
        grad_norm = float(np.linalg.norm(g))
        if grad_norm < grad_tol:
            eigs = np.linalg.eigvalsh(H)
            if np.all(eigs < 0.0):
                frac = crystal.M_inv @ x
                return Peak(cart=x.copy(), frac=frac, rho=rho, grad_norm=grad_norm, hessian_eigs=eigs, iterations=iteration - 1, method=method)
            return None
        if method == 'NR':
            step = newton_step(g, H)
        elif method == 'EF':
            step = ef_step(g, H)
        else:
            raise ValueError(f'Unknown method: {method}')
        if step is None or not np.all(np.isfinite(step)):
            return None
        step = limit_step(step, max_step)
        x = x + step
        frac_now = crystal.M_inv @ x
        if np.any(frac_now < -0.25) or np.any(frac_now > 1.25):
            return None
    rho, g, H = rho_grad_hessian(x, crystal)
    grad_norm = float(np.linalg.norm(g))
    eigs = np.linalg.eigvalsh(H)
    if grad_norm < grad_tol and np.all(eigs < 0.0):
        return Peak(cart=x.copy(), frac=crystal.M_inv @ x, rho=rho, grad_norm=grad_norm, hessian_eigs=eigs, iterations=max_iter, method=method)
    return None

def make_fractional_grid(crystal: CrystalData, low: float, high: float, spacing_angstrom: float) -> np.ndarray:
    """
    Make starting points in fractional coordinates.
    """
    a, b, c = crystal.cell[:3]
    axis_lengths = np.array([a, b, c], dtype=float)
    axes = []
    for length in axis_lengths:
        span_A = (high - low) * length
        intervals = max(1, int(math.ceil(span_A / spacing_angstrom)))
        axes.append(np.linspace(low, high, intervals + 1))
    X, Y, Z = np.meshgrid(axes[0], axes[1], axes[2], indexing='ij')
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

def periodic_cart_distance(frac1: np.ndarray, frac2: np.ndarray, M: np.ndarray) -> float:
    """Minimum-image distance for two fractional positions."""
    dfrac = frac1 - frac2
    dfrac -= np.round(dfrac)
    return float(np.linalg.norm(M @ dfrac))

def normalize_fractional(frac: np.ndarray) -> np.ndarray:
    """Map a fractional coordinate to [0,1)."""
    return np.mod(frac, 1.0)

def deduplicate_peaks(peaks: List[Peak], crystal: CrystalData, tolerance_A: float) -> List[Peak]:
    ordered = sorted(peaks, key=lambda p: p.rho, reverse=True)
    unique: List[Peak] = []
    for p in ordered:
        p.frac = normalize_fractional(p.frac)
        p.cart = crystal.M @ p.frac
        duplicate = any((periodic_cart_distance(p.frac, q.frac, crystal.M) < tolerance_A for q in unique))
        if not duplicate:
            unique.append(p)
    return unique

def search_peaks(crystal: CrystalData, method: str, grid_frac: np.ndarray, grad_tol: float, max_iter: int, max_step: float, duplicate_tol: float, min_rho: float, progress_every: int=250) -> List[Peak]:
    candidates: List[Peak] = []
    for frac0 in grid_frac:
        x0 = crystal.M @ frac0
        peak = optimize_from_start(x0, crystal, method, grad_tol, max_iter, max_step)
        if peak is not None and peak.rho >= min_rho:
            candidates.append(peak)
    unique = deduplicate_peaks(candidates, crystal, duplicate_tol)
    unique.sort(key=lambda p: p.rho, reverse=True)
    return unique

def match_atoms(peaks: List[Peak], crystal: CrystalData):
    matches = []
    for atom_index, (name, x, y, z) in enumerate(REFERENCE_ATOMS, start=1):
        atom_frac = np.array([x, y, z], dtype=float)
        if not peaks:
            matches.append((atom_index, name, atom_frac, None, math.inf, None))
            continue
        distances = [periodic_cart_distance(atom_frac, p.frac, crystal.M) for p in peaks]
        j = int(np.argmin(distances))
        matches.append((atom_index, name, atom_frac, j + 1, distances[j], peaks[j].rho))
    return matches

def peak_table_text(peaks: List[Peak], method: str) -> str:
    lines = [f'PEAKS FOUND WITH {method}', '-' * 105, 'Peak       xf          yf          zf       rho(e/A^3)   |grad|       Hessian eigenvalues', '-' * 105]
    for i, p in enumerate(peaks, start=1):
        e = p.hessian_eigs
        lines.append(f'{i:4d}  {p.frac[0]:10.6f}  {p.frac[1]:10.6f}  {p.frac[2]:10.6f}  {p.rho:12.5f}  {p.grad_norm:9.2e}   [{e[0]: .3e}, {e[1]: .3e}, {e[2]: .3e}]')
    if not peaks:
        lines.append('(No qualifying maxima found.)')
    return '\n'.join(lines)

def match_table_text(matches) -> str:
    lines = ['ATOMS AND CLOSEST PEAKS', '-' * 104, '#   Atom       xf        yf        zf       Match     Distance(A)      rho(e/A^3)', '-' * 104]
    for atom_index, name, frac, peak_no, dist, rho in matches:
        if peak_no is None:
            match_text = 'none'
            dist_text = 'n/a'
            rho_text = 'n/a'
        else:
            match_text = f'peak {peak_no}'
            dist_text = f'{dist:.4f}'
            rho_text = f'{rho:.5f}'
        star = '*' if dist < 0.1 else ' '
        lines.append(f'{atom_index:2d}  {name:<5s}  {frac[0]:8.5f}  {frac[1]:8.5f}  {frac[2]:8.5f}   {star} {match_text:<9s}  {dist_text:>11s}      {rho_text:>12s}')
    lines.append('\n* = successful match (distance < 0.1 A), as specified in the handout.')
    return '\n'.join(lines)

def write_peak_csv(path: Path, peaks: List[Peak]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['peak', 'method', 'xf', 'yf', 'zf', 'x_cart_A', 'y_cart_A', 'z_cart_A', 'rho_e_per_A3', 'gradient_norm_e_per_A4', 'hessian_eig1', 'hessian_eig2', 'hessian_eig3', 'iterations'])
        for i, p in enumerate(peaks, start=1):
            w.writerow([i, p.method, *[f'{v:.10f}' for v in p.frac], *[f'{v:.10f}' for v in p.cart], f'{p.rho:.10f}', f'{p.grad_norm:.10e}', *[f'{v:.10e}' for v in p.hessian_eigs], p.iterations])

def write_match_csv(path: Path, matches) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['atom_number', 'atom', 'xf', 'yf', 'zf', 'closest_peak', 'distance_A', 'successful_under_0.1A', 'peak_rho_e_per_A3'])
        for atom_index, name, frac, peak_no, dist, rho in matches:
            w.writerow([atom_index, name, *[f'{v:.10f}' for v in frac], '' if peak_no is None else peak_no, '' if not np.isfinite(dist) else f'{dist:.10f}', bool(dist < 0.1), '' if rho is None else f'{rho:.10f}'])

def validate_structure_factors(crystal: CrystalData) -> Tuple[float, float]:
    calc = np.sqrt(crystal.A ** 2 + crystal.B ** 2)
    abs_err = np.abs(calc - crystal.f_abs)
    return (float(np.max(abs_err)), float(np.mean(abs_err)))

def derivative_self_check(crystal: CrystalData) -> Tuple[float, float]:
    """
    Small numerical check of the analytic gradient/Hessian at cell center.
    This is not used in production calculations.
    """
    x = crystal.M @ np.array([0.5, 0.5, 0.5])
    rho, g, H = rho_grad_hessian(x, crystal)
    eps_g = 1e-06
    g_num = np.zeros(3)
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps_g
        g_num[i] = (density_only(x + d, crystal) - density_only(x - d, crystal)) / (2 * eps_g)
    eps_h = 2e-05
    H_num = np.zeros((3, 3))
    for j in range(3):
        d = np.zeros(3)
        d[j] = eps_h
        _, gp, _ = rho_grad_hessian(x + d, crystal)
        _, gm, _ = rho_grad_hessian(x - d, crystal)
        H_num[:, j] = (gp - gm) / (2 * eps_h)
    H_num = 0.5 * (H_num + H_num.T)
    g_scale = max(1.0, np.linalg.norm(g_num))
    H_scale = max(1.0, np.linalg.norm(H_num))
    return (float(np.linalg.norm(g - g_num) / g_scale), float(np.linalg.norm(H - H_num) / H_scale))

def inputfile(path: Path) -> bool:
    """Only for the correct file return ture."""
    if not path.is_file():
        return False
    if path.name == Path(__file__).name:
        return False
    if path.suffix.lower() not in {'.txt', '.dat', '.data', '.inp', '.out', ''}:
        return False
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        if len(lines) < 4:
            return False
        tokens = lines[1].split()
        if len(tokens) < 7 or tokens[0].upper() != 'CELL':
            return False
        [float(v) for v in tokens[1:7]]
        for line in lines[3:]:
            p = line.split()
            if len(p) >= 6:
                try:
                    int(p[0])
                    int(p[1])
                    int(p[2])
                    float(p[3])
                    float(p[4])
                    float(p[5])
                    return True
                except ValueError:
                    pass
    except (OSError, ValueError):
        return False
    return False

def choose_input_file() -> Path:
    candidates = sorted([p for p in Path.cwd().iterdir() if inputfile(p)], key=lambda p: p.name.lower())
    if not candidates:
        print('\nNo valid data files were found in the current directory.')
        while True:
            entered = input('\nEnter the data-file path manually: ').strip().strip('"').strip("'")
            p = Path(entered).expanduser()
            if inputfile(p):
                return p
            print('That file is not a valid Project 2 crystallography input file. Try again.')
    print('\nAvailable data files:')
    for i, p in enumerate(candidates, start=1):
        print(f'  [{i}] {p.name}')
    while True:
        choice = input(f'\nPlease select a file number [1-{len(candidates)}]: ').strip()
        try:
            index = int(choice)
            if 1 <= index <= len(candidates):
                return candidates[index - 1]
        except ValueError:
            pass
        print(f'Invalid selection. Please enter a number from 1 to {len(candidates)}.')

def choose_method() -> str:
    print('\nAvailable optimization methods:')
    print('  [1] Newton-Raphson (NR)')
    print('  [2] Eigenvector Following (EF)')
    print('  [3] Run both methods')
    while True:
        choice = input('\nPlease select a method [1-3]: ').strip()
        if choice == '1':
            return 'NR'
        if choice == '2':
            return 'EF'
        if choice == '3':
            return 'BOTH'
        print('Invalid selection. Please enter 1, 2, or 3.')

def parse_args():
    p = argparse.ArgumentParser(description='Project 2 X-ray crystallography peak search (NR / EF).')
    p.add_argument('input', nargs='?', help='Input diffraction-data file')
    p.add_argument('--method', choices=['NR', 'EF', 'BOTH', 'nr', 'ef', 'both'], default='EF', help='Peak-search method (default: EF)')
    p.add_argument('--output', default='result', help='Output directory (default: result)')
    p.add_argument('--grid-spacing', type=float, default=0.4, help='Maximum starting-grid spacing in Angstrom (default: 0.40)')
    p.add_argument('--grid-low', type=float, default=0.1, help='Lower fractional grid bound (default: 0.10)')
    p.add_argument('--grid-high', type=float, default=0.9, help='Upper fractional grid bound (default: 0.90)')
    p.add_argument('--grad-tol', type=float, default=1e-06, help='Gradient-norm convergence tolerance (default: 1e-6)')
    p.add_argument('--max-iter', type=int, default=20, help='Maximum optimizer steps per starting point (default: 20)')
    p.add_argument('--ef-max-step', type=float, default=0.25, help='EF maximum step in Angstrom (default: 0.25)')
    p.add_argument('--nr-max-step', type=float, default=0.25, help='NR maximum step in Angstrom (default: 0.25)')
    p.add_argument('--duplicate-tol', type=float, default=0.1, help='Distance for treating two peaks as duplicates in A (default: 0.10)')
    p.add_argument('--min-rho', type=float, default=2.0, help='Discard maxima below this density in e/A^3 (default: 2.0)')
    p.add_argument('--skip-self-check', action='store_true', help='Skip analytic-derivative numerical self-check')
    return p.parse_args()

def main() -> int:
    args = parse_args()
    if args.input is None:
        try:
            input_path = choose_input_file()
            args.method = choose_method()
        except (EOFError, KeyboardInterrupt):
            print('\nSelection cancelled.')
            return 2
    else:
        input_path = Path(args.input)
    method_label = {'NR': 'Newton-Raphson (NR)', 'EF': 'Eigenvector Following (EF)', 'BOTH': 'NR + EF'}[args.method.upper()]
    print('\nSelection summary')
    print('-' * 50)
    print(f'  Input file : {input_path.name}')
    print(f'  Method     : {method_label}')
    print('-' * 50)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        crystal = prepare_crystal(input_path)
    except Exception as exc:
        print(f'\nERROR while reading/preparing the input:\n  {exc}')
        return 1
    validate_structure_factors(crystal)
    if not args.skip_self_check:
        gerr, Herr = derivative_self_check(crystal)
        if gerr > 1e-05 or Herr > 0.0001:
            print('      WARNING: derivative self-check is larger than expected.')
    grid_frac = make_fractional_grid(crystal, args.grid_low, args.grid_high, args.grid_spacing)
    print(f'      Starting points: {len(grid_frac):,}')
    method_arg = args.method.upper()
    methods = ['NR', 'EF'] if method_arg == 'BOTH' else [method_arg]
    for method in methods:
        max_step = args.nr_max_step if method == 'NR' else args.ef_max_step
        peaks = search_peaks(crystal=crystal, method=method, grid_frac=grid_frac, grad_tol=args.grad_tol, max_iter=args.max_iter, max_step=max_step, duplicate_tol=args.duplicate_tol, min_rho=args.min_rho)
        matches = match_atoms(peaks, crystal)
        peak_text = peak_table_text(peaks, method)
        match_text = match_table_text(matches)
        stem = input_path.stem
        write_peak_csv(output_dir / f'{stem}_{method}_peaks.csv', peaks)
        write_match_csv(output_dir / f'{stem}_{method}_atom_matches.csv', matches)
        method_report = output_dir / f'{stem}_{method}_results.txt'
        method_report.write_text(peak_text + '\n\n' + match_text + '\n', encoding='utf-8')
    print('\nDone. Results saved in folder:', output_dir)
    return 0
if __name__ == '__main__':
    sys.exit(main())
