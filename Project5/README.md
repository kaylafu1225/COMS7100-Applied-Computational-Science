# Project 5 — Hydrogen Atom Variational Solution


## Requirements

- Python 3.9 or newer 
- `numpy` and `mpmath`
- Tkinter (included with many desktop Python installations; on Ubuntu/Debian it may require `python3-tk`)
- A graphical desktop session (or correctly configured remote GUI forwarding)

Install the Python dependencies:

```bash
python3 -m pip install numpy mpmath
```

On Ubuntu/Debian, if Tkinter is missing:

```bash
sudo apt install python3-tk
```

## Project Files

```text
Project5/
├── P5.py                 # Tkinter graphical interface
├── hydrogen_math.py      # Variational solver and wavefunction functions
├── H_STO-12G.txt         # 12-Gaussian basis data
├── H_STO-20G.txt         # 20-Gaussian basis data
├── H_STO-23G.txt         # 23-Gaussian basis data
└── README.md
```

Additional files named `H_STO-2G.txt` through `H_STO-6G.txt` can be placed in the same directory. `STO-1G` and default `STO-2G` exponents are already defined in `hydrogen_math.py`.


## Run

Open a terminal in the project directory and run:

```bash
python3 P5.py
```

A desktop window will open. No browser or `localhost` address is needed. For a syntax check:

```bash
python3 -m py_compile P5.py hydrogen_math.py
```

## GUI Features

- **Basis:** Select an available Gaussian basis set (STO-1G, STO-2G, STO-12G, STO-20G, STO-23G, and any supported additional files).
- **Plot:** Switch between Wavefunction, Radial probability, Energy, and Approximation error.
- **Show exact:** Overlay the exact hydrogen 1s solution on applicable plots.
- **Distance r (Bohr):** Move the vertical distance marker.
- **Plot range:** Adjust the displayed radial range.
- **Export CSV:** Save kinetic, potential, total energy, energy error, and virial ratio for all successfully solved basis sets.

## Mathematical Method

In atomic units, the hydrogen Hamiltonian is

\[
\hat H=-\frac12\nabla^2-\frac1r.
\]

The normalized Gaussian primitives are

\[
g_i(r)=\left(\frac{2\alpha_i}{\pi}\right)^{3/4}e^{-\alpha_i r^2},
\]

and the trial wavefunction is

\[
\psi_{\mathrm{trial}}(r)=\sum_{i=1}^{N}c_i g_i(r).
\]

For each basis set, the solver constructs the overlap matrix \(S\), kinetic-energy matrix \(T\), and potential-energy matrix \(V\), then solves the generalized eigenvalue problem

\[
Hc=ESc,\qquad H=T+V.
\]

It uses high-precision `mpmath` arithmetic and Löwdin symmetric orthogonalization. The lowest eigenvalue gives the variational ground-state energy for the selected basis. The supplied basis-file contraction coefficients are **not** used; the expansion coefficients \(c_i\) are optimized by the solver.

The exact reference is

\[
\psi_{1s}(r)=\frac{e^{-r}}{\sqrt\pi},\qquad E_{\mathrm{exact}}=-0.5\ \mathrm{Hartree}.
\]

The radial probability density displayed by the GUI is

\[
P(r)=4\pi r^2|\psi(r)|^2.
\]

