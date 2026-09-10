# Project 1a - Non-Linear Least Squares in Chemical Thermodynamics

This program fits pressure-volume data to the Virial Equation of State using the Levenberg-Marquardt nonlinear least-squares method.

The program is written in standard C++17 and does not require third-party libraries.

## Project Structure

```text
Project1a/
├── P1a.cpp
├── Project 1a.pdf
├── README.md
├── run.sh
├── run.bat
├── file/
│   ├── data1.txt
│   ├── data2.txt
│   └── ...
└── result/
```

- `P1a.cpp` - main C++ source code
- `run.sh` - compile and run on macOS/Linux
- `run.bat` - compile and run on Windows
- `file/` - input data files
- `result/` - generated fitting results

The `result/` folder will be created automatically if it does not exist.

## Requirements

A C++17-compatible compiler is required.

Supported compilers include:

- Clang
- GCC / G++
- MinGW-w64 G++

No external numerical or optimization libraries are required.

## Quick Start

### macOS / Linux

Open a terminal in the `Project1a` directory.

The first time, give the shell script permission to run:

```bash
chmod +x run.sh
```

Then run:

```bash
./run.sh
```

The script automatically compiles `P1a.cpp` and starts the program.

### Windows

A MinGW-w64 C++ compiler with `g++` available in PATH is required.

Run:

```text
run.bat
```

or double-click `run.bat`.

The script automatically compiles `P1a.cpp` and starts the program.

## Manual Compilation

The program can also be compiled manually.

### macOS

```bash
clang++ -std=c++17 P1a.cpp -o project1a
./project1a
```

### Linux

```bash
g++ -std=c++17 P1a.cpp -o project1a
./project1a
```

### Windows

```text
g++ -std=c++17 P1a.cpp -o project1a.exe
project1a.exe
```

## Input Data

Place all input `.txt` files inside the `file/` directory.

The program automatically detects the available data files and displays a numbered list.

Example:

```text
[1] Ar_223.15K_virial_experiment_B2-B6.txt
[2] Ar_273.15K_virial_theory_B2-B7.txt
[3] air_100K_virial_B2.txt
Select file:
```

Enter the number of the data file to be analyzed.

New `.txt` files can be added to the `file/` directory without modifying the source code.

## Input Format

The expected input format is:

```text
Comment
virial N
B2 B3 ... B(N+1)
temp T K
volume_unit pressure_unit
V1 P1
V2 P2
...
```

The `virial` keyword is case-insensitive.

The program supports up to 10 virial coefficients.

## Units

The program converts pressure and molar-volume data to SI units before fitting.

Supported volume units:

```text
m^3/mol
dm^3/mol
L/mol
cm^3/mol
```

Supported pressure units:

```text
Pa
kPa
MPa
bar
kiloBar
kbar
atm
torr
mmHg
```

The optimized virial coefficients are reported in SI units.

## Method

The program uses the Levenberg-Marquardt method for nonlinear least-squares fitting.

The Virial Equation of State is

p = (RT/Vm) [1 + B2/Vm + B3/Vm^2 + ...].

The program implements the fitting procedure directly without using a built-in nonlinear least-squares or optimization library.

## Output

Final optimized coefficients and fitting statistics are displayed in the terminal.

The program also creates files in the `result/` directory:

```text
<filename>_LM_cycles.csv
<filename>_fit.csv
<filename>_fit.svg
```

### LM Cycles

`*_LM_cycles.csv` contains the fitting history:

- Cycle number
- Lambda
- SSE
- Virial coefficients
- ACCEPT/REJECT status

### Fitted Data

`*_fit.csv` contains:

- Molar volume in m^3/mol
- Experimental pressure in Pa
- Fitted pressure in Pa
- Residual pressure in Pa

### Plot

`*_fit.svg` contains the fitted Virial EOS curve superimposed on the experimental data.

The SVG file can be opened directly with a modern web browser.

## Portability

The source code uses the C++17 standard library and does not depend on external numerical libraries.

The source code is intended to be portable across:

- macOS
- Linux
- Windows

The program should be compiled on the target computer before execution.

Executables are operating-system and architecture dependent. For example, an executable compiled on macOS is not expected to run directly on Windows.

## Notes

Run the program from the main `Project1a` directory so that the program can locate the `file/` directory.

If the source code is modified, run `run.sh` or `run.bat` again. The script will recompile the program before execution.