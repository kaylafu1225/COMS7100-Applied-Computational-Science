# Project 2: Computational Crystallography

This project reconstructs an electron-density function from X-ray
diffraction data and locates electron-density maxima using numerical
optimization. The program supports both **Newton-Raphson (NR)** and
**Eigenvector Following (EF)** methods.

## Files

``` text
Project2/
├── P2.py
├── figure.py
├── proj2_data.txt
└── result/
```

-   `P2.py` --- main program for reading diffraction data, calculating
    electron density, and locating peaks.
-   `figure.py` --- generates figures from the output CSV files.
-   `proj2_data.txt` --- input crystallographic data.
-   `result/` --- output directory created automatically.

## Requirements

The program requires Python 3 and NumPy.

``` bash
pip install numpy
```

For `figure.py`, also install pandas and matplotlib:

``` bash
pip install pandas matplotlib
```

## Input Format

The input file contains the unit-cell parameters and reflection data.

``` text
Comment line
CELL a b c alpha beta gamma
...
h k l |F| A B
```

Each reflection contains the Miller indices `(h, k, l)`,
structure-factor magnitude `|F|`, and real and imaginary components `A`
and `B`.

## How to Run

Run the main program from the Project 2 directory:

``` bash
python3 P2.py
```

The program searches the current directory for valid crystallography
input files and asks the user to select one.

It then asks which optimization method to use:

``` text
[1] Newton-Raphson (NR)
[2] Eigenvector Following (EF)
[3] Run both methods
```

The program can also be run directly from the command line:

``` bash
python3 P2.py proj2_data.txt --method EF
python3 P2.py proj2_data.txt --method NR
python3 P2.py proj2_data.txt --method BOTH
```

## Program Logic

### 1. Read the diffraction data

The program reads the unit-cell parameters

\[ a, b, c, `\alpha`{=tex}, `\beta`{=tex}, `\gamma`{=tex} \]

and the reflection data

\[ (h,k,l),`\quad `{=tex}\|F\|,`\quad `{=tex}A,`\quad `{=tex}B. \]

### 2. Build the unit cell

The unit-cell volume is calculated by

\[
V=abc`\sqrt{1+2\cos\alpha\cos\beta\cos\gamma-\cos^2\alpha-\cos^2\beta-\cos^2\gamma}`{=tex}.
\]

A transformation matrix (M) is constructed to convert fractional
coordinates to Cartesian coordinates:

\[ `\mathbf{x}`{=tex}\_c=M`\mathbf{x}`{=tex}\_f. \]

The inverse transformation is

\[ `\mathbf{x}`{=tex}\_f=M\^{-1}`\mathbf{x}`{=tex}\_c. \]

### 3. Calculate the electron density

At a position (`\mathbf{x}`{=tex}), the electron density is calculated
using Fourier synthesis:

\[
`\rho`{=tex}(`\mathbf{x}`{=tex})=`\frac{2}{V}`{=tex}`\sum`{=tex}\_i`\left[A_i\cos(\theta_i)+B_i\sin(\theta_i)\right]`{=tex},
\]

where

\[
`\theta`{=tex}\_i=2`\pi`{=tex},`\mathbf{s}`{=tex}\_i`\cdot`{=tex}`\mathbf{x}`{=tex}.
\]

The factor of 2 is used because the supplied data contain only half of
the Friedel reflections.

The program also analytically calculates the gradient

\[ `\nabla`{=tex}`\rho`{=tex} \]

and Hessian

\[ H=`\nabla`{=tex}\^2`\rho`{=tex}. \]

### 4. Generate starting points

The program creates a three-dimensional grid of starting positions in
fractional coordinates. The default fractional range is

\[ 0.10`\le `{=tex}x,y,z`\le0.90`{=tex}, \]

and the maximum axial spacing is `0.40 Å`.

The number of starting points is therefore generated automatically from
the unit-cell dimensions and grid-spacing parameter. The starting points
are not manually selected.

Each fractional starting point is converted to Cartesian coordinates
before optimization.

### 5. Search for electron-density maxima

Starting from every grid point, the selected optimization method
iteratively searches for a stationary point.

For Newton-Raphson,

\[ H`\mathbf{h}`{=tex}=-`\mathbf{g}`{=tex}, \]

so the step is

\[ `\mathbf{h}`{=tex}=-H\^{-1}`\mathbf{g}`{=tex}. \]

For Eigenvector Following, the Hessian is diagonalized and the
optimization step is calculated in the Hessian eigenvector directions.

The step length is limited to prevent excessively large optimization
steps.

### 6. Check convergence

A point is accepted when the gradient satisfies

\[ \|`\nabla`{=tex}`\rho`{=tex}\|\<10\^{-6} \]

by default.

The Hessian eigenvalues must also all be negative:

\[
`\lambda`{=tex}\_1\<0,`\qquad `{=tex}`\lambda`{=tex}\_2\<0,`\qquad `{=tex}`\lambda`{=tex}\_3\<0.
\]

This confirms that the stationary point is a local maximum of the
electron density.

### 7. Remove duplicate peaks

Many different starting points can converge to the same electron-density
maximum.

The program compares converged maxima using their periodic Cartesian
distance. Peaks closer than the duplicate tolerance (`0.10 Å` by
default) are treated as the same peak.

The higher-density solution is retained.

### 8. Match peaks to reference atoms

After the peak search is complete, each supplied reference atom is
matched to its nearest calculated peak.

The reference atomic coordinates are used **only at this final matching
stage**. They are not used to generate the starting grid or guide the
optimization.

A match is considered successful when

\[ d\<0.10 `\text{Å}`{=tex}. \]

## Output Files

For each selected method, the program creates three files in the
`result/` directory.

For EF:

``` text
proj2_data_EF_peaks.csv
proj2_data_EF_atom_matches.csv
proj2_data_EF_results.txt
```

For NR:

``` text
proj2_data_NR_peaks.csv
proj2_data_NR_atom_matches.csv
proj2_data_NR_results.txt
```

### Peaks CSV

The peak file contains information such as:

``` text
peak
method
xf, yf, zf
x_cart_A, y_cart_A, z_cart_A
rho_e_per_A3
gradient_norm_e_per_A4
hessian_eig1, hessian_eig2, hessian_eig3
iterations
```

### Atom-Matching CSV

The matching file contains:

``` text
atom_number
atom
xf, yf, zf
closest_peak
distance_A
successful_under_0.1A
peak_rho_e_per_A3
```

## Figures

After running `P2.py`, figures can be generated with:

``` bash
python3 figure.py
```

The figures can be used to visualize the calculated electron-density
peaks, reference atom positions, peak densities, reconstruction errors,
and optimization iterations.

## Default Parameters

``` text
Grid range:          0.10 to 0.90
Grid spacing:        0.40 Å
Gradient tolerance:  1e-6
Maximum iterations:  20
Maximum NR step:     0.25 Å
Maximum EF step:     0.25 Å
Duplicate tolerance: 0.10 Å
Minimum density:     2.0 e/Å³
```

These values can be changed using command-line arguments. For example:

``` bash
python3 P2.py proj2_data.txt --method BOTH --grid-spacing 0.30 --max-iter 30
```

## Summary

The overall workflow is

``` text
Diffraction data
      ↓
Unit-cell transformation
      ↓
Fourier electron-density function
      ↓
Generate 3D starting grid
      ↓
NR / EF optimization from every starting point
      ↓
Check gradient and Hessian
      ↓
Remove duplicate maxima
      ↓
Electron-density peaks
      ↓
Match peaks with reference atoms
      ↓
CSV / TXT results and figures
```
