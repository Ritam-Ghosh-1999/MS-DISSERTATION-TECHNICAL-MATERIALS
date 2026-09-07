from __future__ import annotations

import argparse
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


DATASET_GROUP = "Harvard_Seismology"
DATASET_NAME = "JP"

DATA_URL = (
    "https://suitesparse-collection-website.herokuapp.com/"
    f"MM/{DATASET_GROUP}/{DATASET_NAME}.tar.gz"
)

DEFAULT_LAMBDA_MULTIPLIERS = [
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
]

RTOL = 1e-8
MAXITER = 10000
DEFAULT_REPEATS = 3
CONDITION_PARAMETER = "not applicable for real JP matrix"


def find_matrix_file(data_dir: Path) -> Path:
    matches = list(data_dir.rglob(f"{DATASET_NAME}.mtx"))
    if not matches:
        matches = list(data_dir.rglob(f"{DATASET_NAME.lower()}.mtx"))

    if not matches:
        raise FileNotFoundError(
            f"{DATASET_NAME}.mtx was not found below {data_dir}"
        )

    return matches[0]


def download_data(data_dir: Path) -> Path:
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
    x_reference = np.ones(A.shape[1], dtype=np.float64)
    return np.asarray(A @ x_reference).ravel()


def compute_lambda_scale(A: csr_matrix) -> float:
    ata_diagonal = np.asarray(A.multiply(A).sum(axis=0)).ravel()
    scale = float(np.mean(ata_diagonal))

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Could not compute a positive regularisation scale.")

    return scale


def build_normal_operator(
    A: csr_matrix,
    lambda_: float,
) -> LinearOperator:
    n = A.shape[1]

    def h_matvec(v):
        return A.T @ (A @ v) + lambda_ * v

    return LinearOperator(
        shape=(n, n),
        matvec=h_matvec,
        dtype=np.float64,
    )


def build_pcg_operators(
    A: csr_matrix,
    lambda_: float,
) -> tuple[LinearOperator, LinearOperator]:
    H = build_normal_operator(A, lambda_)

    ata_diagonal = np.asarray(A.multiply(A).sum(axis=0)).ravel()
    diagonal = ata_diagonal + lambda_

    if np.any(diagonal <= 0):
        raise ValueError(
            "The Jacobi preconditioner contains a non-positive diagonal."
        )

    def m_matvec(v):
        return v / diagonal

    M = LinearOperator(
        shape=(A.shape[1], A.shape[1]),
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
    lambda_multiplier: float,
    lambda_scale: float,
    preconditioner: str,
    matrix_free: bool,
    converged: bool,
) -> dict:
    x = np.asarray(x).ravel()

    residual_vector = np.asarray(A @ x - b).ravel()
    residual = np.linalg.norm(residual_vector)

    b_norm = np.linalg.norm(b)
    relative_residual = (
        residual / b_norm if b_norm > 0 else np.nan
    )

    normal_residual_vector = A.T @ residual_vector + lambda_ * x
    normal_residual = np.linalg.norm(normal_residual_vector)

    atb = np.asarray(A.T @ b).ravel()
    atb_norm = np.linalg.norm(atb)

    relative_normal_residual = (
        normal_residual / atb_norm
        if atb_norm > 0 else np.nan
    )

    objective = residual**2 + lambda_ * np.dot(x, x)

    return {
        "Solver": solver,
        "Lambda": lambda_,
        "Lambda Multiplier": lambda_multiplier,
        "Lambda Scale": lambda_scale,
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
        "Condition Parameter": CONDITION_PARAMETER,
    }


def solve_cg(
    A: csr_matrix,
    b: np.ndarray,
    lambda_: float,
    lambda_multiplier: float,
    lambda_scale: float,
) -> dict:
    start = time.perf_counter()

    H = build_normal_operator(A, lambda_)
    rhs = np.asarray(A.T @ b).ravel()

    iterations = [0]

    def callback(_xk):
        iterations[0] += 1

    x, info = cg(
        H,
        rhs,
        rtol=RTOL,
        atol=0.0,
        maxiter=MAXITER,
        callback=callback,
    )

    runtime = time.perf_counter() - start

    return common_metrics(
        solver="CG",
        A=A,
        b=b,
        x=x,
        runtime=runtime,
        iterations=iterations[0],
        info=info,
        lambda_=lambda_,
        lambda_multiplier=lambda_multiplier,
        lambda_scale=lambda_scale,
        preconditioner="None",
        matrix_free=True,
        converged=(info == 0),
    )


def solve_pcg(
    A: csr_matrix,
    b: np.ndarray,
    lambda_: float,
    lambda_multiplier: float,
    lambda_scale: float,
) -> dict:
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
        lambda_multiplier=lambda_multiplier,
        lambda_scale=lambda_scale,
        preconditioner="Jacobi diagonal",
        matrix_free=True,
        converged=(info == 0),
    )


def solve_lsqr(
    A: csr_matrix,
    b: np.ndarray,
    lambda_: float,
    lambda_multiplier: float,
    lambda_scale: float,
) -> dict:
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
    converged = istop in (1, 2, 3, 4, 5, 6)

    return common_metrics(
        solver="LSQR",
        A=A,
        b=b,
        x=x,
        runtime=runtime,
        iterations=iterations,
        info=istop,
        lambda_=lambda_,
        lambda_multiplier=lambda_multiplier,
        lambda_scale=lambda_scale,
        preconditioner="None",
        matrix_free=True,
        converged=converged,
    )


def solve_lsmr(
    A: csr_matrix,
    b: np.ndarray,
    lambda_: float,
    lambda_multiplier: float,
    lambda_scale: float,
) -> dict:
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
    converged = istop in (1, 2, 3, 4, 5, 6)

    return common_metrics(
        solver="LSMR",
        A=A,
        b=b,
        x=x,
        runtime=runtime,
        iterations=iterations,
        info=istop,
        lambda_=lambda_,
        lambda_multiplier=lambda_multiplier,
        lambda_scale=lambda_scale,
        preconditioner="None",
        matrix_free=True,
        converged=converged,
    )


def benchmark(
    data_dir: Path,
    output_dir: Path,
    lambda_multipliers: list[float],
    explicit_lambdas: list[float] | None,
    repeats: int,
    download: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if download:
        matrix_file = download_data(data_dir)
    else:
        matrix_file = find_matrix_file(data_dir)

    A = load_matrix(matrix_file)
    b = construct_rhs(A)
    lambda_scale = compute_lambda_scale(A)

    if explicit_lambdas is not None:
        lambda_values = explicit_lambdas
        multiplier_values = [value / lambda_scale for value in lambda_values]
    else:
        multiplier_values = lambda_multipliers
        lambda_values = [
            multiplier * lambda_scale
            for multiplier in lambda_multipliers
        ]

    print()
    print("=" * 72)
    print("JP JAPAN SEISMIC TOMOGRAPHY FOUR-SOLVER LAMBDA SWEEP")
    print("=" * 72)
    print(f"Dataset:          {DATASET_GROUP}/{DATASET_NAME}")
    print("Problem:          Linearized seismic tomography of Japan")
    print(f"Shape:            {A.shape[0]} x {A.shape[1]}")
    print(f"Nonzeros:         {A.nnz:,}")
    print(
        f"Density:          "
        f"{A.nnz / (A.shape[0] * A.shape[1]):.6e}"
    )
    print("RHS:              constructed as A @ ones")
    print("Solvers:          CG, PCG, LSQR, LSMR")
    print("CG/PCG:           matrix-free normal equations")
    print("LSQR/LSMR:        sparse A, no explicit A.T @ A")
    print(f"Lambda scale:     {lambda_scale:.6e}")
    print(f"RTOL:             {RTOL}")
    print(f"Max iterations:   {MAXITER}")
    print(f"Repeats:          {repeats}")
    print(f"Lambda multipliers:{multiplier_values}")
    print(f"Lambda values:    {lambda_values}")
    print("=" * 72)

    solver_functions = [
        ("CG", solve_cg),
        ("PCG", solve_pcg),
        ("LSQR", solve_lsqr),
        ("LSMR", solve_lsmr),
    ]

    rows = []

    for lambda_multiplier, lambda_ in zip(
        multiplier_values,
        lambda_values,
    ):
        print()
        print("#" * 72)
        print(
            f"LAMBDA = {lambda_:.6e} "
            f"(multiplier={lambda_multiplier:.6e})"
        )
        print("#" * 72)

        for run in range(1, repeats + 1):
            for solver_name, solver_function in solver_functions:
                print(
                    f"Running {solver_name:4s} | "
                    f"lambda={lambda_:.6e} | "
                    f"run={run}/{repeats}"
                )

                result = solver_function(
                    A,
                    b,
                    lambda_=lambda_,
                    lambda_multiplier=lambda_multiplier,
                    lambda_scale=lambda_scale,
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
                    "Condition Parameter": CONDITION_PARAMETER,
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

    results_path = output_dir / "JP_lambda_sweep_results.csv"
    results.to_csv(results_path, index=False)

    summary = (
        results
        .groupby(
            [
                "Dataset",
                "m",
                "n",
                "Nonzeros",
                "Lambda",
                "Lambda Multiplier",
                "Solver",
                "Preconditioner",
                "Condition Parameter",
            ],
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
            Converged_Runs=("Converged", "sum"),
            Total_Runs=("Converged", "count"),
        )
    )

    summary["Convergence_Rate"] = (
        summary["Converged_Runs"] / summary["Total_Runs"]
    )

    summary_path = output_dir / "JP_lambda_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    mean_times = (
        results
        .groupby(["Lambda", "Lambda Multiplier", "Solver"])["Time (s)"]
        .mean()
        .unstack("Solver")
        .reset_index()
    )

    for solver in ["LSQR", "CG", "PCG"]:
        if solver in mean_times.columns and "LSMR" in mean_times.columns:
            mean_times[f"LSMR_vs_{solver}_Time_Ratio"] = (
                mean_times["LSMR"] / mean_times[solver]
            )
            mean_times[f"LSMR_Faster_Than_{solver}"] = (
                mean_times["LSMR"] < mean_times[solver]
            )

    comparison_path = output_dir / "JP_solver_runtime_comparison.csv"
    mean_times.to_csv(comparison_path, index=False)

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(summary.to_string(index=False))

    print()
    print("Mean runtime by lambda:")
    print(mean_times.to_string(index=False))

    print()
    print(f"Detailed results:  {results_path}")
    print(f"Summary:           {summary_path}")
    print(f"Solver comparison: {comparison_path}")

    return results, summary


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",")]

    if not values:
        raise ValueError("At least one value is required.")

    if any(value < 0 for value in values):
        raise ValueError("Values must be non-negative.")

    return values


def main():
    parser = argparse.ArgumentParser(
        description=(
            "JP Japan seismic tomography benchmark comparing "
            "CG, PCG, LSQR and LSMR."
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
        default=Path("real_data_jp_output"),
    )

    parser.add_argument(
        "--lambda-multipliers",
        type=parse_float_list,
        default=DEFAULT_LAMBDA_MULTIPLIERS,
        help=(
            "Comma-separated multipliers of the mean diagonal of A.T @ A. "
            "Default: 1e-3,1e-2,1e-1,1,10,100,1000"
        ),
    )

    parser.add_argument(
        "--lambdas",
        type=parse_float_list,
        default=None,
        help=(
            "Optional comma-separated absolute lambda values. "
            "If supplied, these override the adaptive lambda multipliers."
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

    if args.lambdas is not None:
        lambda_multipliers = []
    else:
        lambda_multipliers = args.lambda_multipliers

    benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lambda_multipliers=lambda_multipliers,
        explicit_lambdas=args.lambdas,
        repeats=args.repeats,
        download=args.download_data,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
