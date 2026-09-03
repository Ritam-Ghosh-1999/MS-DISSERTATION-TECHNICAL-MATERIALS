#!/usr/bin/env python3
"""
Maragal_8 real-data benchmark: PCG vs LSQR vs LSMR

Purpose
-------
Find a defensible regularisation regime in which matrix-free Jacobi-PCG
can outperform LSMR on the same real-world SuiteSparse matrix.

For every lambda:
    - the SAME matrix A is used
    - the SAME constructed RHS b is used
    - PCG, LSQR and LSMR solve the SAME ridge-regularised problem

Problem:
    min_x ||A x - b||_2^2 + lambda ||x||_2^2

Methods:
    PCG  : CG on (A^T A + lambda I)x = A^T b,
           with a Jacobi diagonal preconditioner and matrix-free A^T A.
    LSQR : damped least-squares solver, damp = sqrt(lambda).
    LSMR : damped least-squares solver, damp = sqrt(lambda).

IMPORTANT
---------
The RHS is constructed as b = A @ ones(n). It is NOT an observed RHS
from the original real-world application.

The lambda sweep is deliberate: lambda is an experimental parameter here,
because the purpose of this benchmark is to investigate where PCG shines.
No lambda is selected merely by changing it after seeing a single result.
The complete sweep is recorded in the output CSV.
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg, lsqr, lsmr


DATASET_GROUP = "NYPA"
DATASET_NAME = "Maragal_8"

DATA_URL = (
    "https://suitesparse-collection-website.herokuapp.com/"
    f"MM/{DATASET_GROUP}/{DATASET_NAME}.tar.gz"
)

DEFAULT_LAMBDAS = [
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
]

RTOL = 1e-8
MAXITER = 5000
DEFAULT_REPEATS = 3


def find_matrix_file(data_dir: Path) -> Path:
    """Find Maragal_8.mtx below data_dir."""
    matches = list(data_dir.rglob(f"{DATASET_NAME}.mtx"))
    if not matches:
        matches = list(data_dir.rglob(f"{DATASET_NAME.lower()}.mtx"))

    if not matches:
        raise FileNotFoundError(
            f"{DATASET_NAME}.mtx was not found below {data_dir}"
        )

    return matches[0]


def download_data(data_dir: Path) -> Path:
    """Download and extract the SuiteSparse Maragal_8 archive."""
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        return find_matrix_file(data_dir)
    except FileNotFoundError:
        pass

    archive = data_dir / f"{DATASET_NAME}.tar.gz"

    print(f"Downloading {DATASET_GROUP}/{DATASET_NAME}...")
    urllib.request.urlretrieve(DATA_URL, archive)

    print("Extracting dataset...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir)

    return find_matrix_file(data_dir)


def load_matrix(matrix_file: Path) -> csr_matrix:
    """Load the Matrix Market file as float64 CSR."""
    print(f"Loading: {matrix_file}")

    A = mmread(matrix_file)

    if not hasattr(A, "tocsr"):
        A = csr_matrix(A)
    else:
        A = A.tocsr()

    A = A.astype(np.float64)
    A.eliminate_zeros()

    return A


def construct_rhs(A: csr_matrix) -> np.ndarray:
    """
    Construct one deterministic RHS for all solver comparisons:
        b = A @ ones(n)
    """
    x_reference = np.ones(A.shape[1], dtype=np.float64)
    return np.asarray(A @ x_reference).ravel()


def build_pcg_operators(
    A: csr_matrix,
    lambda_: float,
) -> tuple[LinearOperator, LinearOperator]:
    """
    Build:
        H = A^T A + lambda I
        M = diag(H)^(-1)

    H is matrix-free. A^T A is NEVER explicitly formed.
    """
    n = A.shape[1]

    def h_matvec(v):
        return A.T @ (A @ v) + lambda_ * v

    H = LinearOperator(
        shape=(n, n),
        matvec=h_matvec,
        dtype=np.float64,
    )

    # diag(A^T A) = column-wise sum of squares.
    ata_diagonal = np.asarray(
        A.multiply(A).sum(axis=0)
    ).ravel()

    diagonal = ata_diagonal + lambda_

    if np.any(diagonal <= 0):
        raise ValueError(
            "The Jacobi preconditioner contains a non-positive diagonal."
        )

    def m_matvec(v):
        return v / diagonal

    M = LinearOperator(
        shape=(n, n),
        matvec=m_matvec,
        dtype=np.float64,
    )

    return H, M


def common_metrics(
    solver: str,
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    runtime: float,
    iterations: int,
    info: int,
    lambda_: float,
    preconditioner: str,
    matrix_free: bool,
    converged: bool,
) -> dict:
    """Calculate identical comparison metrics for every solver."""
    x = np.asarray(x).ravel()

    residual_vector = np.asarray(A @ x - b).ravel()
    residual = np.linalg.norm(residual_vector)

    b_norm = np.linalg.norm(b)
    relative_residual = (
        residual / b_norm if b_norm > 0 else np.nan
    )

    normal_residual_vector = (
        A.T @ residual_vector + lambda_ * x
    )
    normal_residual = np.linalg.norm(normal_residual_vector)

    atb = np.asarray(A.T @ b).ravel()
    atb_norm = np.linalg.norm(atb)

    relative_normal_residual = (
        normal_residual / atb_norm
        if atb_norm > 0 else np.nan
    )

    objective = (
        residual**2
        + lambda_ * np.dot(x, x)
    )

    return {
        "Solver": solver,
        "Lambda": lambda_,
        "Time (s)": runtime,
        "Iterations": int(iterations),
        "Residual": residual,
        "Relative Residual": relative_residual,
        "Normal Equation Residual": normal_residual,
        "Relative Normal Equation Residual": relative_normal_residual,
        "Objective": objective,
        "Info": int(info),
        "Converged": bool(converged),
        "Preconditioner": preconditioner,
        "Matrix-Free": matrix_free,
    }


def solve_pcg(
    A: csr_matrix,
    b: np.ndarray,
    lambda_: float,
) -> dict:
    """
    Matrix-free Jacobi-preconditioned CG on the ridge normal equations.

    (A^T A + lambda I)x = A^T b
    """
    start = time.perf_counter()

    H, M = build_pcg_operators(A, lambda_)
    rhs = np.asarray(A.T @ b).ravel()

    iterations = [0]

    def callback(_xk):
        iterations[0] += 1

    x, info = cg(
        H,
        rhs,
        M=M,
        rtol=RTOL,
        atol=0.0,
        maxiter=MAXITER,
        callback=callback,
    )

    runtime = time.perf_counter() - start

    return common_metrics(
        solver="PCG",
        A=A,
        b=b,
        x=x,
        runtime=runtime,
        iterations=iterations[0],
        info=info,
        lambda_=lambda_,
        preconditioner="Jacobi diagonal",
        matrix_free=True,
        converged=(info == 0),
    )


def solve_lsqr(
    A: csr_matrix,
    b: np.ndarray,
    lambda_: float,
) -> dict:
    """Damped LSQR for the same ridge objective."""
    start = time.perf_counter()

    result = lsqr(
        A,
        b,
        damp=np.sqrt(lambda_),
        atol=RTOL,
        btol=RTOL,
        iter_lim=MAXITER,
        show=False,
    )

    runtime = time.perf_counter() - start

    x = result[0]
    istop = result[1]
    iterations = result[2]

    # SciPy LSQR uses istop=7 for the iteration limit.
    converged = istop != 7

    return common_metrics(
        solver="LSQR",
        A=A,
        b=b,
        x=x,
        runtime=runtime,
        iterations=iterations,
        info=istop,
        lambda_=lambda_,
        preconditioner="None",
        matrix_free=True,
        converged=converged,
    )


def solve_lsmr(
    A: csr_matrix,
    b: np.ndarray,
    lambda_: float,
) -> dict:
    """Damped LSMR for the same ridge objective."""
    start = time.perf_counter()

    result = lsmr(
        A,
        b,
        damp=np.sqrt(lambda_),
        atol=RTOL,
        btol=RTOL,
        maxiter=MAXITER,
        show=False,
    )

    runtime = time.perf_counter() - start

    x = result[0]
    istop = result[1]
    iterations = result[2]

    # SciPy LSMR uses istop=7 for the iteration limit.
    converged = istop != 7

    return common_metrics(
        solver="LSMR",
        A=A,
        b=b,
        x=x,
        runtime=runtime,
        iterations=iterations,
        info=istop,
        lambda_=lambda_,
        preconditioner="None",
        matrix_free=True,
        converged=converged,
    )


def benchmark(
    data_dir: Path,
    output_dir: Path,
    lambdas: list[float],
    repeats: int,
    download: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every solver for every lambda using the same A and b."""
    if download:
        matrix_file = download_data(data_dir)
    else:
        matrix_file = find_matrix_file(data_dir)

    A = load_matrix(matrix_file)
    b = construct_rhs(A)

    print()
    print("=" * 72)
    print("MARAGAL_8 THREE-SOLVER LAMBDA SWEEP")
    print("=" * 72)
    print(f"Dataset:       {DATASET_GROUP}/{DATASET_NAME}")
    print(f"Shape:         {A.shape[0]} x {A.shape[1]}")
    print(f"Nonzeros:      {A.nnz:,}")
    print(
        f"Density:       "
        f"{A.nnz / (A.shape[0] * A.shape[1]):.6e}"
    )
    print(f"RHS:           constructed as A @ ones")
    print(f"Solvers:       PCG, LSQR, LSMR")
    print(f"PCG:           Jacobi-preconditioned, matrix-free")
    print(f"RTOL:          {RTOL}")
    print(f"Max iterations:{MAXITER}")
    print(f"Repeats:       {repeats}")
    print(f"Lambdas:       {lambdas}")
    print("=" * 72)

    solver_functions = [
        ("PCG", solve_pcg),
        ("LSQR", solve_lsqr),
        ("LSMR", solve_lsmr),
    ]

    rows = []

    for lambda_ in lambdas:
        print()
        print("#" * 72)
        print(f"LAMBDA = {lambda_:.6g}")
        print("#" * 72)

        for run in range(1, repeats + 1):
            for solver_name, solver_function in solver_functions:
                print(
                    f"Running {solver_name:4s} | "
                    f"lambda={lambda_:.6g} | "
                    f"run={run}/{repeats}"
                )

                result = solver_function(
                    A,
                    b,
                    lambda_=lambda_,
                )

                result.update({
                    "Dataset": DATASET_NAME,
                    "Group": DATASET_GROUP,
                    "m": A.shape[0],
                    "n": A.shape[1],
                    "Nonzeros": A.nnz,
                    "Density": (
                        A.nnz / (A.shape[0] * A.shape[1])
                    ),
                    "RHS Type": "constructed: A @ ones",
                    "Run": run,
                    "RTOL": RTOL,
                    "Max Iterations": MAXITER,
                })

                rows.append(result)

                print(
                    f"    time={result['Time (s)']:.6f}s, "
                    f"iters={result['Iterations']}, "
                    f"relres={result['Relative Residual']:.3e}, "
                    f"relnormal="
                    f"{result['Relative Normal Equation Residual']:.3e}, "
                    f"converged={result['Converged']}"
                )

    results = pd.DataFrame(rows)

    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = (
        output_dir / "Maragal_8_lambda_sweep_results.csv"
    )
    results.to_csv(results_path, index=False)

    summary = (
        results
        .groupby(
            ["Dataset", "m", "n", "Nonzeros", "Lambda",
             "Solver", "Preconditioner"],
            as_index=False,
        )
        .agg(
            Mean_Time=("Time (s)", "mean"),
            Std_Time=("Time (s)", "std"),
            Mean_Iterations=("Iterations", "mean"),
            Std_Iterations=("Iterations", "std"),
            Mean_Residual=("Residual", "mean"),
            Mean_Relative_Residual=(
                "Relative Residual",
                "mean",
            ),
            Mean_Normal_Residual=(
                "Normal Equation Residual",
                "mean",
            ),
            Mean_Relative_Normal_Residual=(
                "Relative Normal Equation Residual",
                "mean",
            ),
            Mean_Objective=("Objective", "mean"),
            All_Converged=("Converged", "all"),
        )
    )

    summary_path = (
        output_dir / "Maragal_8_lambda_sweep_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    # Additional direct comparison: PCG time relative to LSMR time.
    mean_times = (
        results
        .groupby(["Lambda", "Solver"])["Time (s)"]
        .mean()
        .unstack("Solver")
        .reset_index()
    )

    if "PCG" in mean_times.columns and "LSMR" in mean_times.columns:
        mean_times["PCG_vs_LSMR_Time_Ratio"] = (
            mean_times["PCG"] / mean_times["LSMR"]
        )
        mean_times["PCG_Faster_Than_LSMR"] = (
            mean_times["PCG"] < mean_times["LSMR"]
        )

    comparison_path = (
        output_dir / "Maragal_8_pcg_lsmr_comparison.csv"
    )
    mean_times.to_csv(comparison_path, index=False)

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(summary.to_string(index=False))

    print()
    print("PCG vs LSMR mean runtime by lambda:")
    print(mean_times.to_string(index=False))

    print()
    print(f"Detailed results: {results_path}")
    print(f"Summary:          {summary_path}")
    print(f"PCG vs LSMR:      {comparison_path}")

    return results, summary


def parse_lambda_list(value: str) -> list[float]:
    """Parse comma-separated lambda values."""
    values = [float(item.strip()) for item in value.split(",")]

    if not values:
        raise ValueError("At least one lambda is required.")

    if any(value < 0 for value in values):
        raise ValueError("Lambda values must be non-negative.")

    return values


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Maragal_8 lambda sweep comparing PCG, LSQR and LSMR."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("suitesparse_data"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("real_data_pcg_output"),
    )

    parser.add_argument(
        "--lambdas",
        type=parse_lambda_list,
        default=DEFAULT_LAMBDAS,
        help=(
            "Comma-separated lambda values. "
            "Default: 1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1"
        ),
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
    )

    parser.add_argument(
        "--download-data",
        action="store_true",
    )

    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lambdas=args.lambdas,
        repeats=args.repeats,
        download=args.download_data,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
