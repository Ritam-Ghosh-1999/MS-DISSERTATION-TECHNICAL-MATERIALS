"""
Small-matrix four-solver benchmark.

This is the standalone/Anaconda version of the small-matrix experiments.
It reproduces the experimental design used for the dissertation:
Direct, CG, LSQR and LSMR are tested under controlled changes in matrix
size, condition number, regularisation, sparsity, sparsity pattern and spectrum.

No Slurm/HPC configuration is required.

Outputs:
    small_matrix_results.csv
    small_matrix_summary.csv
    matrix_size_summary.csv
    condition_number_summary.csv
    lambda_summary.csv
    sparsity_summary.csv
    pattern_summary.csv
    spectrum_summary.csv
    small_matrix_metadata.txt
"""

from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, lsqr, lsmr, spsolve


SOLVERS = ["Direct", "CG", "LSQR", "LSMR"]
M_VALUES = [400, 500, 1000, 1100, 2100]
N_VALUES = [50, 100, 500, 2000, 2500]
CONDITION_VALUES = [10, 100, 1000, 10000]
LAMBDA_VALUES = [0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1]
SPARSITY_VALUES = [0, 0.25, 0.5, 0.75, 0.9]
PATTERN_VALUES = ["random"]
SPECTRUM_VALUES = ["clustered", "exponential", "linear", "polynomial"]

DEFAULT_M = 500
DEFAULT_N = 50
DEFAULT_CONDITION = 10
DEFAULT_LAMBDA = 0.0
DEFAULT_SPARSITY = 0.0
DEFAULT_PATTERN = "random"
DEFAULT_SPECTRUM = "exponential"

DEFAULT_REPEATS = 10
DEFAULT_TOL = 1e-6
DEFAULT_MAXITER = 1000
DEFAULT_DENSITY = 0.10


def spectrum_values(n, condition_number, spectrum):
    if n == 1:
        return np.ones(1)

    kappa = float(condition_number)

    if spectrum == "clustered":
        values = np.ones(n)
        split = max(1, n // 10)
        values[:split] = 1.0 / kappa
        return values

    if spectrum == "exponential":
        return np.geomspace(1.0, 1.0 / kappa, n)

    if spectrum == "linear":
        return np.linspace(1.0, 1.0 / kappa, n)

    if spectrum == "polynomial":
        x = np.linspace(0.0, 1.0, n)
        return 1.0 / (1.0 + (kappa - 1.0) * x**2)

    raise ValueError(f"Unknown spectrum: {spectrum}")


def orthogonal_matrix(n, rng):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def build_matrix(m, n, condition_number, sparsity, pattern, spectrum, rng):
    if pattern != "random":
        raise ValueError("Only the recorded random sparsity pattern is supported.")

    rank = min(m, n)
    singular = spectrum_values(rank, condition_number, spectrum)

    if m >= n:
        u = orthogonal_matrix(m, rng)[:, :n]
        v = orthogonal_matrix(n, rng)
        a = (u * singular) @ v.T
    else:
        u = orthogonal_matrix(m, rng)
        v = orthogonal_matrix(n, rng)[:, :m]
        a = (u * singular) @ v.T

    if sparsity > 0:
        keep_probability = 1.0 - sparsity
        mask = rng.random(a.shape) < keep_probability
        a = a * mask

        if np.count_nonzero(a) == 0:
            a.flat[rng.integers(0, a.size)] = singular[0]

    scale = np.linalg.norm(a, ord="fro")
    if scale > 0:
        a = a / scale

    return a


def make_rhs(a, rng, noise_level=0.0):
    x_true = rng.standard_normal(a.shape[1])
    x_true /= max(np.linalg.norm(x_true), np.finfo(float).eps)
    b = a @ x_true

    if noise_level > 0:
        b = b + noise_level * rng.standard_normal(a.shape[0])

    return b, x_true


def normal_operator(a, lam):
    n = a.shape[1]

    def mv(x):
        return a.T @ (a @ x) + lam * x

    return LinearOperator((n, n), matvec=mv, dtype=np.float64)


def run_direct(a, b, lam):
    start = time.perf_counter()

    if lam == 0:
        x = np.linalg.lstsq(a, b, rcond=None)[0]
    else:
        ata = a.T @ a
        rhs = a.T @ b
        x = np.linalg.solve(ata + lam * np.eye(a.shape[1]), rhs)

    elapsed = time.perf_counter() - start
    return x, 1, 0, elapsed


def run_cg(a, b, lam, tol, maxiter):
    op = normal_operator(a, lam)
    rhs = a.T @ b
    start = time.perf_counter()
    count = [0]

    def callback(_):
        count[0] += 1

    x, info = cg(op, rhs, rtol=tol, atol=0.0, maxiter=maxiter,
                 callback=callback)
    elapsed = time.perf_counter() - start
    return x, count[0], int(info), elapsed


def run_lsqr(a, b, lam, tol, maxiter):
    if lam == 0:
        start = time.perf_counter()
        result = lsqr(a, b, atol=tol, btol=tol, iter_lim=maxiter)
        elapsed = time.perf_counter() - start
    else:
        n = a.shape[1]

        def matvec(x):
            return np.concatenate((a @ x, np.sqrt(lam) * x))

        def rmatvec(y):
            return a.T @ y[:a.shape[0]] + np.sqrt(lam) * y[a.shape[0]:]

        augmented = LinearOperator(
            (a.shape[0] + n, n),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.float64,
        )

        rhs = np.concatenate((b, np.zeros(n)))

        start = time.perf_counter()
        result = lsqr(augmented, rhs, atol=tol, btol=tol,
                      iter_lim=maxiter)
        elapsed = time.perf_counter() - start

    return result[0], int(result[2]), int(result[1]), elapsed


def run_lsmr(a, b, lam, tol, maxiter):
    if lam == 0:
        start = time.perf_counter()
        result = lsmr(a, b, atol=tol, btol=tol, maxiter=maxiter)
        elapsed = time.perf_counter() - start
    else:
        n = a.shape[1]

        def matvec(x):
            return np.concatenate((a @ x, np.sqrt(lam) * x))

        def rmatvec(y):
            return a.T @ y[:a.shape[0]] + np.sqrt(lam) * y[a.shape[0]:]

        augmented = LinearOperator(
            (a.shape[0] + n, n),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.float64,
        )

        rhs = np.concatenate((b, np.zeros(n)))

        start = time.perf_counter()
        result = lsmr(augmented, rhs, atol=tol, btol=tol,
                      maxiter=maxiter)
        elapsed = time.perf_counter() - start

    return result[0], int(result[2]), int(result[1]), elapsed


def solve_one(solver, a, b, lam, tol, maxiter):
    if solver == "Direct":
        return run_direct(a, b, lam)
    if solver == "CG":
        return run_cg(a, b, lam, tol, maxiter)
    if solver == "LSQR":
        return run_lsqr(a, b, lam, tol, maxiter)
    if solver == "LSMR":
        return run_lsmr(a, b, lam, tol, maxiter)
    raise ValueError(solver)


def metrics(a, x, b, x_true, lam):
    residual = a @ x - b
    rel_residual = np.linalg.norm(residual) / max(np.linalg.norm(b), 1e-15)
    error = np.linalg.norm(x - x_true) / max(np.linalg.norm(x_true), 1e-15)
    objective = np.dot(residual, residual) + lam * np.dot(x, x)
    return rel_residual, error, objective


def run_case(case, repeats, tol, maxiter, density, seed):
    rows = []

    for repeat in range(1, repeats + 1):
        rng = np.random.default_rng(seed + repeat)

        a = build_matrix(
            case["m"], case["n"], case["condition_number"],
            case["sparsity"], case["pattern"], case["spectrum"], rng
        )

        b, x_true = make_rhs(a, rng)

        for solver in SOLVERS:
            try:
                x, iterations, info, runtime = solve_one(
                    solver, a, b, case["lambda"], tol, maxiter
                )
                rel_residual, error, objective = metrics(
                    a, x, b, x_true, case["lambda"]
                )
                converged = info == 0

                rows.append({
                    "Experiment": case["experiment"],
                    "m": case["m"],
                    "n": case["n"],
                    "Condition Number": case["condition_number"],
                    "Lambda": case["lambda"],
                    "Sparsity": case["sparsity"],
                    "Sparsity Pattern": case["pattern"],
                    "Spectrum": case["spectrum"],
                    "Solver": solver,
                    "Runtime (s)": runtime,
                    "Iterations": iterations,
                    "Info": info,
                    "Converged": converged,
                    "Residual": rel_residual,
                    "Relative Error": error,
                    "Objective": objective,
                    "Repeat": repeat,
                    "Tolerance": tol,
                    "Max Iterations": maxiter,
                    "Density": density,
                    "Seed": seed + repeat,
                })
            except Exception as exc:
                rows.append({
                    "Experiment": case["experiment"],
                    "m": case["m"],
                    "n": case["n"],
                    "Condition Number": case["condition_number"],
                    "Lambda": case["lambda"],
                    "Sparsity": case["sparsity"],
                    "Sparsity Pattern": case["pattern"],
                    "Spectrum": case["spectrum"],
                    "Solver": solver,
                    "Runtime (s)": np.nan,
                    "Iterations": np.nan,
                    "Info": -999,
                    "Converged": False,
                    "Residual": np.nan,
                    "Relative Error": np.nan,
                    "Objective": np.nan,
                    "Repeat": repeat,
                    "Tolerance": tol,
                    "Max Iterations": maxiter,
                    "Density": density,
                    "Seed": seed + repeat,
                    "Error Message": str(exc),
                })

    return rows


def experiment_cases():
    cases = []

    for m in M_VALUES:
        for n in N_VALUES:
            cases.append({
                "experiment": "matrix_size",
                "m": m,
                "n": n,
                "condition_number": DEFAULT_CONDITION,
                "lambda": DEFAULT_LAMBDA,
                "sparsity": DEFAULT_SPARSITY,
                "pattern": DEFAULT_PATTERN,
                "spectrum": DEFAULT_SPECTRUM,
            })

    for kappa in CONDITION_VALUES:
        cases.append({
            "experiment": "condition_number",
            "m": DEFAULT_M,
            "n": DEFAULT_N,
            "condition_number": kappa,
            "lambda": DEFAULT_LAMBDA,
            "sparsity": DEFAULT_SPARSITY,
            "pattern": DEFAULT_PATTERN,
            "spectrum": DEFAULT_SPECTRUM,
        })

    for lam in LAMBDA_VALUES:
        cases.append({
            "experiment": "lambda",
            "m": DEFAULT_M,
            "n": DEFAULT_N,
            "condition_number": DEFAULT_CONDITION,
            "lambda": lam,
            "sparsity": DEFAULT_SPARSITY,
            "pattern": DEFAULT_PATTERN,
            "spectrum": DEFAULT_SPECTRUM,
        })

    for sparsity in SPARSITY_VALUES:
        cases.append({
            "experiment": "sparsity",
            "m": DEFAULT_M,
            "n": DEFAULT_N,
            "condition_number": DEFAULT_CONDITION,
            "lambda": DEFAULT_LAMBDA,
            "sparsity": sparsity,
            "pattern": DEFAULT_PATTERN,
            "spectrum": DEFAULT_SPECTRUM,
        })

    for pattern in PATTERN_VALUES:
        cases.append({
            "experiment": "sparsity_pattern",
            "m": DEFAULT_M,
            "n": DEFAULT_N,
            "condition_number": DEFAULT_CONDITION,
            "lambda": DEFAULT_LAMBDA,
            "sparsity": DEFAULT_SPARSITY,
            "pattern": pattern,
            "spectrum": DEFAULT_SPECTRUM,
        })

    for spectrum in SPECTRUM_VALUES:
        cases.append({
            "experiment": "spectrum",
            "m": DEFAULT_M,
            "n": DEFAULT_N,
            "condition_number": DEFAULT_CONDITION,
            "lambda": DEFAULT_LAMBDA,
            "sparsity": DEFAULT_SPARSITY,
            "pattern": DEFAULT_PATTERN,
            "spectrum": spectrum,
        })

    return cases


def save_outputs(results, output_dir, repeats, tol, maxiter):
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / "small_matrix_results.csv", index=False)

    valid = results.dropna(subset=["Runtime (s)"])

    summary = (
        valid.groupby(
            ["Experiment", "m", "n", "Condition Number", "Lambda",
             "Sparsity", "Sparsity Pattern", "Spectrum", "Solver"],
            as_index=False
        )
        .agg(
            Mean_Time=("Runtime (s)", "mean"),
            Std_Time=("Runtime (s)", "std"),
            Mean_Iterations=("Iterations", "mean"),
            Std_Iterations=("Iterations", "std"),
            Convergence_Rate=("Converged", "mean"),
            Mean_Residual=("Residual", "mean"),
            Mean_Relative_Error=("Relative Error", "mean"),
            Mean_Objective=("Objective", "mean"),
        )
    )

    summary.to_csv(output_dir / "small_matrix_summary.csv", index=False)

    for name, experiment in [
        ("matrix_size_summary.csv", "matrix_size"),
        ("condition_number_summary.csv", "condition_number"),
        ("lambda_summary.csv", "lambda"),
        ("sparsity_summary.csv", "sparsity"),
        ("pattern_summary.csv", "sparsity_pattern"),
        ("spectrum_summary.csv", "spectrum"),
    ]:
        summary[summary["Experiment"] == experiment].to_csv(
            output_dir / name, index=False
        )

    metadata = f"""Small-matrix four-solver benchmark

Solvers: Direct, CG, LSQR, LSMR
Matrix-size m values: {M_VALUES}
Matrix-size n values: {N_VALUES}
Condition numbers: {CONDITION_VALUES}
Lambda values: {LAMBDA_VALUES}
Sparsity values: {SPARSITY_VALUES}
Sparsity patterns: {PATTERN_VALUES}
Spectra: {SPECTRUM_VALUES}

Repeats per case: {repeats}
Tolerance: {tol}
Maximum iterations: {maxiter}

The matrix is generated in memory and no Slurm/HPC configuration is required.
Direct solves the original system (or the regularised normal equations when
lambda is non-zero); CG solves the regularised normal equations; LSQR and
LSMR solve the least-squares problem, using an augmented matrix for
lambda > 0.

This script is intended as a transparent standalone reproduction benchmark.
"""
    (output_dir / "small_matrix_metadata.txt").write_text(metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output_small_matrices")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--seed", type=int, default=14339893)
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY)
    parser.add_argument("--experiments", nargs="+",
                        choices=["matrix_size", "condition_number", "lambda",
                                 "sparsity", "sparsity_pattern", "spectrum",
                                 "all"],
                        default=["all"])
    args = parser.parse_args()

    selected = set(args.experiments)
    if "all" in selected:
        selected = {
            "matrix_size", "condition_number", "lambda",
            "sparsity", "sparsity_pattern", "spectrum"
        }

    cases = [c for c in experiment_cases() if c["experiment"] in selected]

    all_rows = []
    total = len(cases)

    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{total}] {case['experiment']} "
            f"m={case['m']} n={case['n']} "
            f"kappa={case['condition_number']} "
            f"lambda={case['lambda']} "
            f"s={case['sparsity']} "
            f"spectrum={case['spectrum']}"
        )
        all_rows.extend(
            run_case(
                case, args.repeats, args.tol, args.maxiter,
                args.density, args.seed + index * 1000
            )
        )

    results = pd.DataFrame(all_rows)
    save_outputs(
        results, Path(args.output), args.repeats, args.tol, args.maxiter
    )

    print("\nBenchmark complete.")
    print(f"Results: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
