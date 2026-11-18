from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.integrate import quad
from scipy.special import eval_genlaguerre, lpmv, sph_harm_y


N_MAX = 4
L_MAX = 3
DEFAULT_OUTPUT = Path("project4_output")
SIGNED_COLORS = LinearSegmentedColormap.from_list(
    "signed_wavefunction", ["#244c9a", "white", "#c3272b"]
)


def validate_quantum_numbers(n: int, ell: int, m: int | None = None) -> None:
    """Raise a clear error if n, ell, m are not an allowed combination."""
    if n < 1 or not 0 <= ell <= n - 1:
        raise ValueError(f"Need n >= 1 and 0 <= ell <= n - 1; got n={n}, ell={ell}")
    if m is not None and not -ell <= m <= ell:
        raise ValueError(f"Need -ell <= m <= ell; got ell={ell}, m={m}")


def radial_function(n: int, ell: int, r: np.ndarray | float) -> np.ndarray:
    """Normalized hydrogen radial function R_nl(r), with a0 = 1."""
    validate_quantum_numbers(n, ell)
    r = np.asarray(r, dtype=float)
    coefficient = (2.0 / n) ** (ell + 1.5)
    coefficient *= math.sqrt(
        math.factorial(n - ell - 1)
        / (2.0 * n * math.factorial(n + ell))
    )
    laguerre = eval_genlaguerre(n - ell - 1, 2 * ell + 1, 2 * r / n)
    return coefficient * r**ell * np.exp(-r / n) * laguerre


def real_spherical_harmonic(
    ell: int,
    m: int,
    theta: np.ndarray | float,
    phi: np.ndarray | float,
) -> np.ndarray:
    """Wavefunction-normalized real spherical harmonic y_l^m(theta, phi).

    theta is the polar angle in [0, pi], and phi is the azimuth in [0, 2 pi].
    This convention follows equations (95)-(100) of the project handout.
    """
    if ell < 0 or abs(m) > ell:
        raise ValueError(f"Need ell >= 0 and |m| <= ell; got ell={ell}, m={m}")
    k = abs(m)
    complex_harmonic = sph_harm_y(ell, k, theta, phi)
    if m > 0:
        return (-1) ** k * math.sqrt(2.0) * complex_harmonic.real
    if m < 0:
        return (-1) ** k * math.sqrt(2.0) * complex_harmonic.imag
    return complex_harmonic.real


def real_spherical_harmonic_from_xyz(
    ell: int,
    m: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Evaluate y_l^m using Cartesian coordinates."""
    radius = np.sqrt(x * x + y * y + z * z)
    safe_radius = np.where(radius == 0.0, 1.0, radius)
    theta = np.arccos(np.clip(z / safe_radius, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)
    angular = real_spherical_harmonic(ell, m, theta, phi)
    if ell > 0:
        angular = np.where(radius == 0.0, 0.0, angular)
    return angular


def wavefunction(
    n: int,
    ell: int,
    m: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Real Cartesian hydrogen wavefunction psi_nlm(x, y, z)."""
    validate_quantum_numbers(n, ell, m)
    radius = np.sqrt(x * x + y * y + z * z)
    return radial_function(n, ell, radius) * real_spherical_harmonic_from_xyz(
        ell, m, x, y, z
    )


def radial_normalization(n: int, ell: int) -> float:
    """Numerically evaluate integral_0^infinity r^2 R_nl(r)^2 dr."""
    value, _ = quad(
        lambda radius: radius**2 * float(radial_function(n, ell, radius)) ** 2,
        0.0,
        np.inf,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=250,
    )
    return value


def angular_normalization(ell: int, m: int) -> float:
    """Numerically evaluate the spherical normalization integral."""
    k = abs(m)
    scale = math.sqrt(
        (1 if k == 0 else 2)
        * (2 * ell + 1)
        * math.factorial(ell - k)
        / (4 * np.pi * math.factorial(ell + k))
    )
    polar, _ = quad(lambda u: float(lpmv(k, ell, u)) ** 2, -1.0, 1.0)
    azimuth = 2 * np.pi if m == 0 else np.pi
    return scale**2 * polar * azimuth


def save_normalization_table(output: Path) -> None:
    """Check every required state and save the results as a CSV table."""
    rows: list[list[object]] = []
    maximum_error = 0.0
    print("\nNormalization checks")
    print(" n  l   m       radial       angular          full")
    print("------------------------------------------------------")
    for n in range(1, N_MAX + 1):
        for ell in range(n):
            radial = radial_normalization(n, ell)
            for m in range(-ell, ell + 1):
                angular = angular_normalization(ell, m)
                full = radial * angular
                maximum_error = max(maximum_error, abs(full - 1.0))
                print(f" {n:1d}  {ell:1d}  {m:2d}   {radial:12.9f}  {angular:12.9f}  {full:12.9f}")
                rows.append([n, ell, m, radial, angular, full, abs(full - 1.0)])

    table_path = output / "normalization_checks.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["n", "l", "m", "radial_norm", "angular_norm", "full_norm", "absolute_error"]
        )
        writer.writerows(rows)

    print(f"Maximum full-normalization error: {maximum_error:.3e}")
    if maximum_error > 1e-8:
        raise RuntimeError("A normalization check failed. Inspect normalization_checks.csv.")


def plot_part_a(output: Path) -> None:
    """Create the 10 required radial plots."""
    folder = output / "part_A_radial"
    folder.mkdir(parents=True, exist_ok=True)
    for n in range(1, N_MAX + 1):
        for ell in range(n):
            r_max = 3 * n**2 + 3
            radius = np.linspace(0.0, r_max, 1600)
            radial = radial_function(n, ell, radius)
            probability = radius**2 * radial**2

            figure, axis = plt.subplots(figsize=(7.2, 4.8))
            axis.plot(radius, radial, color="#2455a4", lw=2.2, label=r"$R_{n\ell}(r)$")
            axis.plot(
                radius,
                probability,
                color="#bd292f",
                lw=2.2,
                label=r"$r^2|R_{n\ell}(r)|^2$",
            )
            axis.axhline(0.0, color="0.55", lw=0.8)
            axis.set(xlabel=r"$r$ (Bohr radii)", title=fr"Radial function: $n={n},\ \ell={ell}$")
            axis.set_ylabel("Amplitude or radial probability density")
            axis.grid(alpha=0.22)
            axis.legend(frameon=False)
            figure.tight_layout()
            figure.savefig(folder / f"radial_n{n}_l{ell}.png", dpi=180)
            plt.close(figure)
    print(f"Part A: saved 10 radial plots in {folder}")


def plot_part_b(output: Path) -> None:
    """Create 16 red/blue isosurfaces of exp(-r) y_l^m at +/- 0.09."""
    folder = output / "part_B_angular"
    folder.mkdir(parents=True, exist_ok=True)

    theta = np.linspace(0.0, np.pi, 121)
    phi = np.linspace(0.0, 2 * np.pi, 241)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    for ell in range(L_MAX + 1):
        for m in range(-ell, ell + 1):
            harmonic = real_spherical_harmonic(ell, m, theta_grid, phi_grid)
            figure = plt.figure(figsize=(6.3, 6.0))
            axis = figure.add_subplot(111, projection="3d")
            isovalue = 0.09
            for sign, color in ((1, "#c71f25"), (-1, "#163e9b")):
                signed_harmonic = sign * harmonic
                valid = signed_harmonic >= isovalue
                surface_radius = np.full_like(signed_harmonic, np.nan)
                surface_radius[valid] = -np.log(isovalue / signed_harmonic[valid])
                x = surface_radius * np.sin(theta_grid) * np.cos(phi_grid)
                y = surface_radius * np.sin(theta_grid) * np.sin(phi_grid)
                z = surface_radius * np.cos(theta_grid)
                axis.plot_surface(
                    x, y, z, color=color, linewidth=0, antialiased=True, shade=True,
                    rcount=91, ccount=181,
                )
            extent = 2.35
            axis.plot([-extent, extent], [0, 0], [0, 0], color="0.25", lw=0.8)
            axis.plot([0, 0], [-extent, extent], [0, 0], color="0.25", lw=0.8)
            axis.plot([0, 0], [0, 0], [-extent, extent], color="0.25", lw=0.8)
            axis.text(extent, 0, 0, "x")
            axis.text(0, extent, 0, "y")
            axis.text(0, 0, extent, "z")
            axis.set_box_aspect((1, 1, 1))
            axis.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
            axis.set_axis_off()
            axis.view_init(elev=23, azim=38)
            axis.set_title(
                fr"Real spherical harmonic $y_{{{ell}}}^{{{m}}}$"
                + "\nred $>0$, blue $<0$"
            )
            figure.tight_layout()
            figure.savefig(folder / f"angular_l{ell}_m{m:+d}.png", dpi=180)
            plt.close(figure)
    print(f"Part B: saved 16 angular plots in {folder}")


def sample_plane(
    n: int, ell: int, m: int, plane: str, coordinate: np.ndarray
) -> np.ndarray:
    """Evaluate one Cartesian coordinate-plane slice."""
    a, b = np.meshgrid(coordinate, coordinate, indexing="xy")
    zero = np.zeros_like(a)
    if plane == "xy":
        return wavefunction(n, ell, m, a, b, zero)
    if plane == "xz":
        return wavefunction(n, ell, m, a, zero, b)
    if plane == "yz":
        return wavefunction(n, ell, m, zero, a, b)
    raise ValueError(f"Unknown plane: {plane}")


def plot_part_c(output: Path) -> None:
    """Create 30 figures, each containing the xy, xz and yz slices."""
    folder = output / "part_C_complete_wavefunctions"
    folder.mkdir(parents=True, exist_ok=True)

    labels = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}
    for n in range(1, N_MAX + 1):
        extent = 2 * n**2 + 2
        coordinate = np.linspace(-extent, extent, 321)
        for ell in range(n):
            for m in range(-ell, ell + 1):
                planes = {
                    name: sample_plane(n, ell, m, name, coordinate)
                    for name in ("xy", "xz", "yz")
                }
                peak = max(float(np.max(np.abs(values))) for values in planes.values())
                peak = peak if peak > 0 else 1.0
                norm = TwoSlopeNorm(vmin=-peak, vcenter=0.0, vmax=peak)

                figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.25), constrained_layout=True)
                image = None
                for axis, plane in zip(axes, ("xy", "xz", "yz")):
                    values = planes[plane]
                    image = axis.imshow(
                        values,
                        origin="lower",
                        extent=(-extent, extent, -extent, extent),
                        cmap=SIGNED_COLORS,
                        norm=norm,
                        interpolation="bilinear",
                    )
                    axis.axhline(0.0, color="0.3", lw=0.45)
                    axis.axvline(0.0, color="0.3", lw=0.45)
                    axis.set_aspect("equal")
                    axis.set_title(f"{plane} plane")
                    axis.set_xlabel(fr"${labels[plane][0]}$ (Bohr radii)")
                    axis.set_ylabel(fr"${labels[plane][1]}$ (Bohr radii)")
                    if np.allclose(values, 0.0, atol=1e-13):
                        axis.text(
                            0.5,
                            0.5,
                            "nodal plane\n$\\psi=0$",
                            ha="center",
                            va="center",
                            transform=axis.transAxes,
                            fontsize=11,
                        )
                figure.colorbar(image, ax=axes, shrink=0.84, label=r"signed amplitude $\psi_{n\ell m}$")
                figure.suptitle(fr"Hydrogen wavefunction: $n={n},\ \ell={ell},\ m={m}$", fontsize=14)
                figure.savefig(folder / f"psi_n{n}_l{ell}_m{m:+d}.png", dpi=180)
                plt.close(figure)
    print(f"Part C: saved 30 three-panel figures in {folder}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        choices=("A", "B", "C", "all", "check"),
        default="all",
        help="Generate one part, all parts, or normalization checks only (default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    arguments.output.mkdir(parents=True, exist_ok=True)
    save_normalization_table(arguments.output)
    if arguments.part in ("A", "all"):
        plot_part_a(arguments.output)
    if arguments.part in ("B", "all"):
        plot_part_b(arguments.output)
    if arguments.part in ("C", "all"):
        plot_part_c(arguments.output)
    print(f"\nFinished. Results are in: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()