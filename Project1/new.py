import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# ============================================================
# Global constants and display names
# ============================================================

R = 8.314462618  # J/(mol*K)

EOS_NAMES = {
    "vdw": "van der Waals",
    "rk": "Redlich-Kwong",
    "dieterici": "Dieterici",
    "berthelot": "Berthelot",
}

METHOD_NAMES = {
    "GD": "Gradient Descent",
    "GN": "Gauss-Newton",
    "LM": "Levenberg-Marquardt",
}

VALID_EOS = set(EOS_NAMES.keys())


# ============================================================
# User interface helpers
# ============================================================

def print_header(title, char="-"):
    print("\n" + char * 70)
    print(title.center(70))
    print(char * 70)


def choose_data_file(folder="file"):
    """Let the user choose a data file by number."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"The data folder '{folder}' was not found. "
            "Please create it and place your experimental data files inside."
        )

    files = sorted(
        name for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
    )

    if not files:
        raise FileNotFoundError(
            f"No data files were found in the '{folder}' folder."
        )

    print_header("Project 1: EOS Data Fitting Program")
    print("Available data files:\n")

    for i, name in enumerate(files, start=1):
        print(f"  [{i}] {name}")

    while True:
        choice = input("\nSelect a data file by number: ").strip()

        try:
            index = int(choice)
        except ValueError:
            print("Invalid entry. Please enter a number from the list.")
            continue

        if 1 <= index <= len(files):
            filename = files[index - 1]
            filepath = os.path.join(folder, filename)
            print(f"\nSelected file: {filename}")
            return filename, filepath

        print(f"Please enter a number between 1 and {len(files)}.")


def choose_method():
    """Let the user choose one optimization method or all methods."""
    print_header("Non-Linear Optimization Method")
    print("Please select a fitting method:\n")
    print("  [1] Gradient Descent")
    print("  [2] Gauss-Newton")
    print("  [3] Levenberg-Marquardt")
    print("  [4] Run all methods and compare")

    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        if choice in {"1", "2", "3", "4"}:
            return choice
        print("Invalid selection. Please enter 1, 2, 3, or 4.")


# ============================================================
# Input file reading and unit conversion
# ============================================================

def read_data_file(filepath):
    """Read an EOS experimental data file."""
    with open(filepath, "r") as file:
        # Line 1: comment
        comment = file.readline().strip()

        # Line 2: equation of state
        eos = file.readline().strip().lower()

        # Line 3: initial guesses for parameters
        initial_params = np.array(
            [float(x) for x in file.readline().split()],
            dtype=float,
        )

        # Line 4: temperature
        temperature_line = file.readline().split()
        if len(temperature_line) < 2:
            raise ValueError("Temperature line is not in the expected format.")
        temperature = float(temperature_line[1])

        # Line 5: units
        units = file.readline().split()
        if len(units) < 2:
            raise ValueError("Unit line is not in the expected format.")
        volume_unit = units[0].lower()
        pressure_unit = units[1].lower()

        # Line 6 onward: experimental data
        volume = []
        pressure = []

        for line_number, line in enumerate(file, start=6):
            line = line.strip()
            if line == "":
                continue

            values = line.split()
            if len(values) < 2:
                raise ValueError(
                    f"Invalid experimental data on line {line_number}."
                )

            volume.append(float(values[0]))
            pressure.append(float(values[1]))

    volume = np.array(volume, dtype=float)
    pressure = np.array(pressure, dtype=float)

    if len(volume) == 0:
        raise ValueError("The data file contains no experimental data points.")

    return {
        "comment": comment,
        "eos": eos,
        "initial_params": initial_params,
        "temperature": temperature,
        "volume_unit": volume_unit,
        "pressure_unit": pressure_unit,
        "volume": volume,
        "pressure": pressure,
    }


def convert_volume_to_si(volume, unit):
    """Convert molar volume to SI unit: m^3/mol."""
    unit = unit.lower()

    if unit == "m^3/mol":
        factor = 1.0
    elif unit == "dm^3/mol":
        factor = 1e-3
    elif unit == "cm^3/mol":
        factor = 1e-6
    elif unit == "l/mol":
        factor = 1e-3
    else:
        raise ValueError(f"Unknown volume unit: {unit}")

    return volume * factor


def convert_pressure_to_si(pressure, unit):
    """Convert pressure to SI unit: Pa."""
    unit = unit.lower()

    if unit == "pa":
        factor = 1.0
    elif unit == "megapa":
        factor = 1e6
    elif unit == "kilobar":
        factor = 1e8
    elif unit == "bar":
        factor = 1e5
    elif unit == "atm":
        factor = 101325.0
    elif unit == "torr":
        factor = 101325.0 / 760.0
    elif unit == "mmhg":
        factor = 133.322387415
    else:
        raise ValueError(f"Unknown pressure unit: {unit}")

    return pressure * factor


# ============================================================
# Equations of state
# ============================================================

def vdw_model(V, T, params):
    """van der Waals: P = RT/(V-b) - a/V^2."""
    a = params[0]
    b = params[1]
    return R * T / (V - b) - a / V**2


def rk_model(V, T, params):
    """Redlich-Kwong: P = RT/(V-b) - a/[sqrt(T)*V*(V+b)]."""
    a = params[0]
    b = params[1]
    return R * T / (V - b) - a / (np.sqrt(T) * V * (V + b))


def dieterici_model(V, T, params):
    """Dieterici: P = RT/(V-b) * exp[-a/(RTV)]."""
    a = params[0]
    b = params[1]
    return R * T / (V - b) * np.exp(-a / (R * T * V))


def berthelot_model(V, T, params):
    """Berthelot: P = RT/(V-b) - a/(T*V^2)."""
    a = params[0]
    b = params[1]
    return R * T / (V - b) - a / (T * V**2)


def eos_model(V, T, params, eos):
    """Automatically select the equation of state."""
    eos = eos.lower()

    if eos == "vdw":
        return vdw_model(V, T, params)
    elif eos == "rk":
        return rk_model(V, T, params)
    elif eos == "dieterici":
        return dieterici_model(V, T, params)
    elif eos == "berthelot":
        return berthelot_model(V, T, params)

    raise ValueError(f"Unknown equation of state: {eos}")


# ============================================================
# Residuals, SSE, parameter checks
# ============================================================

def calculate_residuals(V, P, T, params, eos):
    predicted = eos_model(V, T, params, eos)
    return P - predicted


def calculate_sse(V, P, T, params, eos):
    residuals = calculate_residuals(V, P, T, params, eos)
    return np.sum(residuals**2)


def valid_parameters(V, params):
    """Check whether EOS parameters are numerically/physically valid."""
    params = np.array(params, dtype=float)

    if len(params) < 2:
        return False

    if not np.all(np.isfinite(params)):
        return False

    b = params[1]
    if np.any(V - b <= 0):
        return False

    return True



# Analytical Jacobian
def vdw_jacobian(V, T, params):
    b = params[1]
    df_da = -1.0 / V**2
    df_db = R * T / (V - b)**2
    return np.column_stack((df_da, df_db))


def rk_jacobian(V, T, params):
    a = params[0]
    b = params[1]
    df_da = -1.0 / (np.sqrt(T) * V * (V + b))
    df_db = (
        R * T / (V - b)**2
        + a / (np.sqrt(T) * V * (V + b)**2)
    )
    return np.column_stack((df_da, df_db))


def dieterici_jacobian(V, T, params):
    a = params[0]
    b = params[1]
    exponential = np.exp(-a / (R * T * V))
    df_da = -exponential / (V * (V - b))
    df_db = R * T * exponential / (V - b)**2
    return np.column_stack((df_da, df_db))


def berthelot_jacobian(V, T, params):
    b = params[1]
    df_da = -1.0 / (T * V**2)
    df_db = R * T / (V - b)**2
    return np.column_stack((df_da, df_db))


def eos_jacobian(V, T, params, eos):
    eos = eos.lower()

    if eos == "vdw":
        return vdw_jacobian(V, T, params)
    elif eos == "rk":
        return rk_jacobian(V, T, params)
    elif eos == "dieterici":
        return dieterici_jacobian(V, T, params)
    elif eos == "berthelot":
        return berthelot_jacobian(V, T, params)

    raise ValueError(f"Unknown EOS: {eos}")


def calculate_quantities(V, P, T, params, eos):
    predicted = eos_model(V, T, params, eos)
    delta_y = P - predicted
    J = eos_jacobian(V, T, params, eos)
    S = delta_y.T @ delta_y
    beta = J.T @ delta_y
    alpha = J.T @ J
    gradient = -2.0 * beta
    return S, delta_y, J, beta, alpha, gradient



# Optimization methods
def gradient_descent(
    V,
    P,
    T,
    initial_params,
    eos,
    max_iterations=5000,
    tolerance=1e-5,
    initial_step=1.0,
):
    params = np.array(initial_params, dtype=float).copy()
    scale = np.array([
        max(abs(params[0]), 1.0),
        max(abs(params[1]), 1e-5),
    ])

    converged = False
    small_change_count = 0
    history = []

    for cycle in range(1, max_iterations + 1):
        S, delta_y, J, beta, alpha, gradient = calculate_quantities(
            V, P, T, params, eos
        )

        scaled_beta = beta * scale
        beta_norm = np.linalg.norm(scaled_beta)

        if beta_norm == 0:
            converged = True
            break

        direction_scaled = scaled_beta / beta_norm
        step = initial_step
        accepted = False

        while step > 1e-20:
            delta_params = step * direction_scaled * scale
            trial_params = params + delta_params

            if not valid_parameters(V, trial_params):
                step *= 0.5
                continue

            trial_S = calculate_sse(V, P, T, trial_params, eos)

            if np.isfinite(trial_S) and trial_S < S:
                accepted = True
                break

            step *= 0.5

        if not accepted:
            print("\nGradient Descent could not find an improving step.")
            break

        decrease = S - trial_S
        fractional_decrease = decrease / max(abs(S), 1e-30)
        params = trial_params

        history.append({
            "cycle": cycle,
            "SSE": trial_S,
            "params": params.copy(),
            "step": step,
        })

        if decrease < tolerance or fractional_decrease < tolerance:
            small_change_count += 1
        else:
            small_change_count = 0

        if small_change_count >= 3:
            converged = True
            print("\nGradient Descent converged.")
            break

    final_S = calculate_sse(V, P, T, params, eos)

    return {
        "method": "GD",
        "params": params,
        "SSE": final_S,
        "converged": converged,
        "iterations": len(history),
        "history": history,
    }


def gauss_newton(
    V,
    P,
    T,
    initial_params,
    eos,
    max_iterations=100,
    tolerance=1e-5,
):
    params = np.array(initial_params, dtype=float).copy()
    converged = False
    small_change_count = 0
    history = []

    for cycle in range(1, max_iterations + 1):
        S, delta_y, J, beta, alpha, gradient = calculate_quantities(
            V, P, T, params, eos
        )

        try:
            delta_params = np.linalg.solve(alpha, beta)
        except np.linalg.LinAlgError:
            print("\nGauss-Newton stopped: alpha matrix is singular.")
            break

        trial_params = params + delta_params

        if not valid_parameters(V, trial_params):
            print("\nGauss-Newton stopped: invalid trial parameters.")
            break

        trial_S = calculate_sse(V, P, T, trial_params, eos)

        if not np.isfinite(trial_S):
            print("\nGauss-Newton stopped: non-finite SSE.")
            break

        decrease = S - trial_S
        fractional_decrease = abs(decrease) / max(abs(S), 1e-30)

        params = trial_params

        history.append({
            "cycle": cycle,
            "SSE": trial_S,
            "params": params.copy(),
        })

        if abs(decrease) < tolerance or fractional_decrease < tolerance:
            small_change_count += 1
        else:
            small_change_count = 0

        if small_change_count >= 3:
            converged = True
            print("\nGauss-Newton converged.")
            break

    final_S = calculate_sse(V, P, T, params, eos)

    return {
        "method": "GN",
        "params": params,
        "SSE": final_S,
        "converged": converged,
        "iterations": len(history),
        "history": history,
    }


def levenberg_marquardt(
    V,
    P,
    T,
    initial_params,
    eos,
    max_cycles=1000,
    tolerance=1e-5,
    initial_lambda=0.001,
):
    params = np.array(initial_params, dtype=float).copy()
    lambda_value = initial_lambda
    converged = False
    small_change_count = 0
    history = []
    S = calculate_sse(V, P, T, params, eos)

    for cycle in range(1, max_cycles + 1):
        S, delta_y, J, beta, alpha, gradient = calculate_quantities(
            V, P, T, params, eos
        )
        lambda_used = lambda_value

        alpha_prime = alpha.copy()
        for m in range(len(params)):
            alpha_prime[m, m] = alpha[m, m] * (1.0 + lambda_used)

        try:
            delta_params = np.linalg.solve(alpha_prime, beta)
        except np.linalg.LinAlgError:
            lambda_value *= 10.0
            history.append({
                "cycle": cycle,
                "lambda": lambda_used,
                "SSE": S,
                "params": params.copy(),
                "status": "REJECT",
            })
            continue

        trial_params = params + delta_params

        if not valid_parameters(V, trial_params):
            lambda_value *= 10.0
            history.append({
                "cycle": cycle,
                "lambda": lambda_used,
                "SSE": S,
                "params": params.copy(),
                "status": "REJECT",
            })
            continue

        trial_S = calculate_sse(V, P, T, trial_params, eos)

        if not np.isfinite(trial_S) or trial_S >= S:
            lambda_value *= 10.0
            history.append({
                "cycle": cycle,
                "lambda": lambda_used,
                "SSE": S,
                "params": params.copy(),
                "status": "REJECT",
            })
            continue

        decrease = S - trial_S
        fractional_decrease = decrease / max(abs(S), 1e-30)

        params = trial_params
        S = trial_S
        lambda_value /= 10.0

        history.append({
            "cycle": cycle,
            "lambda": lambda_used,
            "SSE": S,
            "params": params.copy(),
            "status": "ACCEPT",
        })

        if decrease < tolerance or fractional_decrease < tolerance:
            small_change_count += 1
        else:
            small_change_count = 0

        if small_change_count >= 3:
            converged = True
            print("\nLevenberg-Marquardt converged.")
            break

        if lambda_value > 1e30:
            print("\nLM stopped because lambda became too large.")
            break

    return {
        "method": "LM",
        "params": params,
        "SSE": S,
        "converged": converged,
        "iterations": len(history),
        "lambda": lambda_value,
        "history": history,
    }

# Statistics
def calculate_statistics(V, P, T, params, eos):
    predicted = eos_model(V, T, params, eos)
    residuals = P - predicted

    N = len(P)
    M = len(params)

    chi_square = np.sum(residuals**2)
    degrees_of_freedom = N - M

    if degrees_of_freedom <= 0:
        raise ValueError(
            "Not enough data points to calculate parameter statistics."
        )

    sample_variance = chi_square / degrees_of_freedom

    J = eos_jacobian(V, T, params, eos)
    alpha = J.T @ J

    try:
        C = np.linalg.inv(alpha)
    except np.linalg.LinAlgError:
        C = np.linalg.pinv(alpha)

    covariance_matrix = sample_variance * C
    standard_errors = np.sqrt(np.diag(covariance_matrix))

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation_matrix = covariance_matrix / np.outer(
            standard_errors,
            standard_errors,
        )

    y_mean = np.mean(P)
    SS_total = np.sum((P - y_mean)**2)

    if SS_total == 0:
        R_squared = np.nan
        adjusted_R_squared = np.nan
    else:
        R_squared = 1.0 - chi_square / SS_total
        adjusted_R_squared = 1.0 - (
            (chi_square / (N - M)) / (SS_total / (N - 1))
        )

    denominator = np.sum(np.abs(P))
    R_factor = (
        np.sum(np.abs(residuals)) / denominator
        if denominator != 0
        else np.nan
    )

    RMSE = np.sqrt(chi_square / N)

    return {
        "chi_square": chi_square,
        "degrees_of_freedom": degrees_of_freedom,
        "sample_variance": sample_variance,
        "C": C,
        "covariance_matrix": covariance_matrix,
        "standard_errors": standard_errors,
        "correlation_matrix": correlation_matrix,
        "R_squared": R_squared,
        "adjusted_R_squared": adjusted_R_squared,
        "R_factor": R_factor,
        "RMSE": RMSE,
        "predicted": predicted,
        "residuals": residuals,
    }



# Result reporting and saving
def print_method_comparison(results):
    print_header("Method Comparision")

    print(
        f"{'Method':<25}"
        f"{'a':<18}"
        f"{'b':<18}"
        f"{'SSE':<20}"
        f"{'Iterations':<12}"
        f"{'Converged':<12}"
    )

    for method, result in results.items():
        print(
            f"{METHOD_NAMES[method]:<25}"
            f"{result['params'][0]:<18.6e}"
            f"{result['params'][1]:<18.6e}"
            f"{result['SSE']:<20.6e}"
            f"{result['iterations']:<12}"
            f"{str(result['converged']):<12}"
        )


def determine_best_methods(results, tie_tolerance=1e-8):
    valid_results = {
        method: result
        for method, result in results.items()
        if result["converged"] and np.isfinite(result["SSE"])
    }

    # If a single selected method did not satisfy the convergence flag,
    # still allow its finite result to be reported rather than crashing.
    if len(valid_results) == 0 and len(results) == 1:
        method, result = next(iter(results.items()))
        if np.isfinite(result["SSE"]):
            print(
                "\nWarning: the selected method did not satisfy the "
                "convergence test, but its final finite result will be reported."
            )
            return [method]

    if len(valid_results) == 0:
        raise RuntimeError("No optimization method converged.")

    best_sse = min(result["SSE"] for result in valid_results.values())

    best_methods = []
    for method, result in valid_results.items():
        relative_difference = (
            abs(result["SSE"] - best_sse) / max(abs(best_sse), 1.0)
        )
        if relative_difference <= tie_tolerance:
            best_methods.append(method)

    return best_methods


def print_final_report(best_methods, results, statistics_results, eos, T):
    print_header("Final Report")

    if len(best_methods) > 1:
        print(
            "Several methods produced essentially identical minimum SSE values."
        )

    for method in best_methods:
        result = results[method]
        stats = statistics_results[method]

        print(f"Method                   : {METHOD_NAMES[method]}")
        print(f"Equation of state        : {EOS_NAMES[eos]}")
        print(f"Temperature              : {T:.3f} K")
        print(f"Converged                : {result['converged']}")
        print(f"Iterations               : {result['iterations']}")

        print("\nOptimized EOS parameters:")
        print(f"  a = {result['params'][0]:.8e}")
        print(f"  b = {result['params'][1]:.8e}")

        print("\nGoodness of fit:")
        print(f"  SSE             = {result['SSE']:.8e}")
        print(f"  RMSE            = {stats['RMSE']:.8e}")
        print(f"  Chi-square      = {stats['chi_square']:.8e}")
        print(f"  R squared       = {stats['R_squared']:.8f}")
        print(f"  Adjusted R^2    = {stats['adjusted_R_squared']:.8f}")
        print(f"  R-factor        = {stats['R_factor']:.8e}")
        print(f"  R-factor (%)    = {stats['R_factor'] * 100:.4f}%")

        print("\nParameter uncertainty:")
        print(f"  Standard error of a = {stats['standard_errors'][0]:.8e}")
        print(f"  Standard error of b = {stats['standard_errors'][1]:.8e}")

        # print("\nAdditional statistical information:")
        # print(f"  Degrees of freedom = {stats['degrees_of_freedom']}")
        # print(f"  Sample variance    = {stats['sample_variance']:.8e}")
        print("\n  C matrix:")
        print(stats["C"])
        print("\n  Parameter covariance matrix:")
        print(stats["covariance_matrix"])
        print("\n  Correlation matrix:")
        print(stats["correlation_matrix"])


def create_output_folder(filename, eos, results):
    """Create one result subfolder for the current analysis.

    Naming format:
        filename_EOS_METHOD

    Examples:
        data1_vdw_LM
        data1_rk_GN
        data1_dieterici_GD
        data1_vdw_ALL
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]

    if len(results) == 1:
        method = next(iter(results))
        method_tag = method
    else:
        method_tag = "ALL"

    output_base = f"{base_name}_{eos}_{method_tag}"
    output_folder = os.path.join("result", output_base)

    os.makedirs(output_folder, exist_ok=True)

    return output_folder, output_base


def save_results_csv(output_folder, output_base, eos, T, results, statistics_results):
    output_name = os.path.join(
        output_folder,
        f"{output_base}_results.csv",
    )

    with open(output_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([
            "Method",
            "EOS",
            "Temperature_K",
            "a",
            "b",
            "SSE",
            "RMSE",
            "R_squared",
            "Adjusted_R_squared",
            "R_factor",
            "Std_Error_a",
            "Std_Error_b",
            "Iterations",
            "Converged",
        ])

        for method, result in results.items():
            stats = statistics_results.get(method)

            if stats is None:
                writer.writerow([
                    METHOD_NAMES[method],
                    EOS_NAMES[eos],
                    T,
                    result["params"][0],
                    result["params"][1],
                    result["SSE"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    result["iterations"],
                    result["converged"],
                ])
            else:
                writer.writerow([
                    METHOD_NAMES[method],
                    EOS_NAMES[eos],
                    T,
                    result["params"][0],
                    result["params"][1],
                    result["SSE"],
                    stats["RMSE"],
                    stats["R_squared"],
                    stats["adjusted_R_squared"],
                    stats["R_factor"],
                    stats["standard_errors"][0],
                    stats["standard_errors"][1],
                    result["iterations"],
                    result["converged"],
                ])

    return output_name


def create_plots(output_folder, output_base, V, P, T, eos, results):
    fit_plot_name = os.path.join(
        output_folder,
        f"{output_base}_fit.png",
    )
    residual_plot_name = os.path.join(
        output_folder,
        f"{output_base}_residuals.png",
    )

    # Fit plot
    plt.figure(figsize=(8, 6))
    plt.scatter(V, P, s=20, label="Experimental data")

    sort_index = np.argsort(V)
    V_sorted = V[sort_index]

    for method, result in results.items():
        fitted_pressure = eos_model(V_sorted, T, result["params"], eos)
        plt.plot(
            V_sorted,
            fitted_pressure,
            linewidth=2,
            label=f"{METHOD_NAMES[method]} fit",
        )

    plt.xlabel("Molar Volume (m^3/mol)")
    plt.ylabel("Pressure (Pa)")
    plt.title(f"{EOS_NAMES[eos]} Equation of State Fit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fit_plot_name, dpi=300)
    plt.show()

    # Residual plot
    plt.figure(figsize=(8, 6))

    for method, result in results.items():
        predicted = eos_model(V, T, result["params"], eos)
        residuals = P - predicted
        plt.scatter(
            V,
            residuals,
            s=15,
            label=f"{METHOD_NAMES[method]} residual",
        )

    plt.axhline(y=0, linestyle="--", linewidth=1)
    plt.xlabel("Molar Volume (m^3/mol)")
    plt.ylabel("Residual Pressure (Pa)")
    plt.title(f"{EOS_NAMES[eos]} Residual Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(residual_plot_name, dpi=300)
    plt.show()

    return fit_plot_name, residual_plot_name


# ============================================================
# Main program
# ============================================================

def main():
    try:
        filename, filepath = choose_data_file("file")
        data = read_data_file(filepath)

        if data["eos"] not in VALID_EOS:
            raise ValueError(
                f"Unknown EOS '{data['eos']}'. "
                "Supported EOS options are: vdw, rk, dieterici, berthelot."
            )

        if len(data["initial_params"]) != 2:
            raise ValueError(
                "This program expects exactly two initial EOS parameters: a and b."
            )

        print("Experimental data loaded successfully.")

        print_header("Dataset Infromation", "-")
        print(f"Description        : {data['comment']}")
        print(f"Equation of state  : {EOS_NAMES[data['eos']]}")
        print(f"Temperature        : {data['temperature']:.3f} K")
        print(f"Data points        : {len(data['volume'])}")
        print(f"Volume unit        : {data['volume_unit']}")
        print(f"Pressure unit      : {data['pressure_unit']}")
        print("\nInitial parameter estimates:")
        print(f"  a = {data['initial_params'][0]:.6e}")
        print(f"  b = {data['initial_params'][1]:.6e}")

        volume_si = convert_volume_to_si(
            data["volume"],
            data["volume_unit"],
        )
        pressure_si = convert_pressure_to_si(
            data["pressure"],
            data["pressure_unit"],
        )
        print("Unit conversion completed successfully.")
        # print("  Volume   -> m^3/mol")
        # print("  Pressure -> Pa")

        data["volume_si"] = volume_si
        data["pressure_si"] = pressure_si

        V = data["volume_si"]
        P = data["pressure_si"]
        T = data["temperature"]
        eos = data["eos"]
        initial_params = data["initial_params"]

        initial_sse = calculate_sse(V, P, T, initial_params, eos)
        print(f"\nInitial SSE: {initial_sse:.6e}")

        method_choice = choose_method()

        results = {}

        if method_choice == "1":
            print("\nRunning Gradient Descent:")
            results["GD"] = gradient_descent(
                V, P, T, initial_params, eos
            )

        elif method_choice == "2":
            print("\nRunning Gauss-Newton:")
            results["GN"] = gauss_newton(
                V, P, T, initial_params, eos
            )

        elif method_choice == "3":
            print("\nRunning Levenberg-Marquardt:")
            results["LM"] = levenberg_marquardt(
                V,
                P,
                T,
                initial_params,
                eos,
                initial_lambda=0.001,
            )

        elif method_choice == "4":
            print("\nRunning all optimization methods:")

            results["GD"] = gradient_descent(
                V, P, T, initial_params, eos
            )
            results["GN"] = gauss_newton(
                V, P, T, initial_params, eos
            )
            results["LM"] = levenberg_marquardt(
                V,
                P,
                T,
                initial_params,
                eos,
                initial_lambda=0.001,
            )

        # Only show method comparison when the user selected ALL methods.
        if method_choice == "4":
            print_method_comparison(results)

            best_methods = determine_best_methods(results)

            print("\nBest method(s):")
            for method in best_methods:
                print(f"  - {METHOD_NAMES[method]}")
        else:
            # For a single selected method, no comparison section is needed.
            # Keep the selected method as the method used for the final report.
            best_methods = list(results.keys())

        statistics_results = {}

        # Calculate statistics for every method that produced a finite result.
        for method, result in results.items():
            if np.isfinite(result["SSE"]):
                try:
                    statistics_results[method] = calculate_statistics(
                        V,
                        P,
                        T,
                        result["params"],
                        eos,
                    )
                except Exception as error:
                    print(
                        f"\nWarning: statistics could not be calculated for "
                        f"{METHOD_NAMES[method]}: {error}"
                    )

        missing_best_stats = [
            method
            for method in best_methods
            if method not in statistics_results
        ]
        if missing_best_stats:
            raise RuntimeError(
                "Statistics could not be calculated for the best fitting method."
            )

        print_final_report(
            best_methods,
            results,
            statistics_results,
            eos,
            T,
        )

        output_folder, output_base = create_output_folder(
            filename,
            eos,
            results,
        )

        csv_name = save_results_csv(
            output_folder,
            output_base,
            eos,
            T,
            results,
            statistics_results,
        )

        fit_plot_name, residual_plot_name = create_plots(
            output_folder,
            output_base,
            V,
            P,
            T,
            eos,
            results,
        )

        print_header("Complete the analysis!")
        print(f"\nAll output are saved in: {output_folder}")
        print("\nThank you for using the EOS Data Fitting Program.\n")

    except FileNotFoundError as error:
        print("\nERROR:")
        print(error)

    except ValueError as error:
        print("\nINPUT/DATA ERROR:")
        print(error)

    except RuntimeError as error:
        print("\nANALYSIS ERROR:")
        print(error)

    except Exception as error:
        print("\nUNEXPECTED ERROR:")
        print(error)


if __name__ == "__main__":
    main()
