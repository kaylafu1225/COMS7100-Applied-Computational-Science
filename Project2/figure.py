import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"

EF_PEAKS = RESULT_DIR / "proj2_data_EF_peaks.csv"
EF_MATCHES = RESULT_DIR / "proj2_data_EF_atom_matches.csv"
NR_PEAKS = RESULT_DIR / "proj2_data_NR_peaks.csv"
NR_MATCHES = RESULT_DIR / "proj2_data_NR_atom_matches.csv"

ef_peaks = pd.read_csv(EF_PEAKS)
ef_atoms = pd.read_csv(EF_MATCHES)
nr_peaks = pd.read_csv(NR_PEAKS)
nr_atoms = pd.read_csv(NR_MATCHES)

for df in [ef_peaks, ef_atoms, nr_peaks, nr_atoms]:
    df.columns = df.columns.str.replace("**", "", regex=False).str.strip()

print("EF peaks:", len(ef_peaks))
print("EF atom matches:", len(ef_atoms))
print("NR peaks:", len(nr_peaks))
print("NR atom matches:", len(nr_atoms))


# 1. 3D atom and peak comparison
def plot_3d_atoms_vs_peaks(peaks, atoms, method):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    rho = peaks["rho_e_per_A3"].to_numpy()
    peak_sizes = 20 + 3 * rho

    ax.scatter(peaks["xf"], peaks["yf"], peaks["zf"], s=peak_sizes, alpha=0.45, label=f"{method} calculated peaks")
    ax.scatter(atoms["xf"], atoms["yf"], atoms["zf"], s=45, marker="x", label="Reference atoms")

    for _, atom in atoms.iterrows():
        ax.text(atom["xf"], atom["yf"], atom["zf"], atom["atom"], fontsize=8)

        peak_number = int(atom["closest_peak"])
        peak = peaks.loc[peaks["peak"] == peak_number].iloc[0]

        ax.plot([atom["xf"], peak["xf"]], [atom["yf"], peak["yf"]], [atom["zf"], peak["zf"]], linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Fractional x")
    ax.set_ylabel("Fractional y")
    ax.set_zlabel("Fractional z")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_title(f"{method}: Reference Atoms vs Electron-Density Peaks")
    ax.legend()

    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{method}_3D_atom_peak_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


# 2. Electron-density peak heights
def plot_density(peaks, atoms, method):
    labels = []
    densities = []

    for _, atom in atoms.iterrows():
        peak_number = int(atom["closest_peak"])
        peak = peaks.loc[peaks["peak"] == peak_number].iloc[0]
        labels.append(atom["atom"])
        densities.append(peak["rho_e_per_A3"])

    plt.figure(figsize=(11, 6))
    plt.bar(labels, densities)
    plt.xlabel("Atom")
    plt.ylabel(r"Electron Density (e/$\AA^3$)")
    plt.title(f"{method}: Electron-Density Peak Height")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{method}_electron_density.png", dpi=300, bbox_inches="tight")
    plt.show()


# 3. Position reconstruction error
def plot_errors(atoms, method):
    plt.figure(figsize=(11, 6))
    plt.bar(atoms["atom"], atoms["distance_A"])
    plt.axhline(y=0.1, linestyle="--", label=r"0.1 $\AA$ success threshold")
    plt.xlabel("Atom")
    plt.ylabel(r"Distance to Closest Peak ($\AA$)")
    plt.title(f"{method}: Atomic Position Reconstruction Error")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{method}_position_error.png", dpi=300, bbox_inches="tight")
    plt.show()


# 4. Number of iterations
def plot_iterations(peaks, atoms, method):
    labels = []
    iterations = []

    for _, atom in atoms.iterrows():
        peak_number = int(atom["closest_peak"])
        peak = peaks.loc[peaks["peak"] == peak_number].iloc[0]
        labels.append(atom["atom"])
        iterations.append(peak["iterations"])

    plt.figure(figsize=(11, 6))
    plt.bar(labels, iterations)
    plt.xlabel("Atom")
    plt.ylabel("Iterations")
    plt.title(f"{method}: Iterations Required to Locate Each Peak")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{method}_iterations.png", dpi=300, bbox_inches="tight")
    plt.show()


# EF
plot_3d_atoms_vs_peaks(ef_peaks, ef_atoms, "EF")
plot_density(ef_peaks, ef_atoms, "EF")
plot_errors(ef_atoms, "EF")
plot_iterations(ef_peaks, ef_atoms, "EF")

# NR
plot_3d_atoms_vs_peaks(nr_peaks, nr_atoms, "NR")
plot_density(nr_peaks, nr_atoms, "NR")
plot_errors(nr_atoms, "NR")
plot_iterations(nr_peaks, nr_atoms, "NR")
