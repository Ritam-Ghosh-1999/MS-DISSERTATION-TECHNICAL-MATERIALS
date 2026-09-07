"""
Plotting script for the standalone small-matrix four-solver benchmark.

Reads the CSV files produced by small_matrix_four_solver_benchmark.py and
writes publication-ready PNG figures into the plots/ directory.

The plots compare Direct, CG, LSQR and LSMR using:
    runtime
    iteration count
    relative reconstruction error
    residual
    runtime distributions
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SOLVER_ORDER = ["Direct", "CG", "LSQR", "LSMR"]
OUTPUT_DIR = Path("output_small_matrices")
PLOT_DIR = OUTPUT_DIR / "plots"


def save(fig, filename):
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def ordered(data, column):
    values = list(data[column].dropna().unique())
    if column in {"m", "n", "Condition Number", "Lambda", "Sparsity"}:
        return sorted(values)
    return values


def solver_bar(data, metric, title, ylabel, filename, log=False):
    if data.empty:
        return

    grouped = (
        data.groupby(["Solver"], as_index=False)[metric]
        .mean()
    )
    grouped["Solver"] = pd.Categorical(
        grouped["Solver"], categories=SOLVER_ORDER, ordered=True
    )
    grouped = grouped.sort_values("Solver")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(grouped["Solver"].astype(str), grouped[metric])
    ax.set_title(title)
    ax.set_xlabel("Solver")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    save(fig, filename)


def line_plot(data, x, y, title, xlabel, ylabel, filename, log_y=False):
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for solver in SOLVER_ORDER:
        subset = data[data["Solver"] == solver]
        if subset.empty:
            continue

        curve = (
            subset.groupby(x, as_index=False)[y]
            .mean()
            .sort_values(x)
        )

        ax.plot(curve[x], curve[y], marker="o", label=solver)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.legend()
    save(fig, filename)


def box_plot(data, x, y, title, xlabel, ylabel, filename):
    if data.empty:
        return

    values = []
    labels = []

    for solver in SOLVER_ORDER:
        subset = data.loc[data["Solver"] == solver, y].dropna()
        if len(subset):
            values.append(subset)
            labels.append(solver)

    if not values:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(values, labels=labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save(fig, filename)


def make_experiment_plots(results, experiment, prefix):
    data = results[results["Experiment"] == experiment].copy()
    if data.empty:
        return

    if experiment == "matrix_size":
        data["Matrix"] = (
            data["m"].astype(int).astype(str)
            + " × "
            + data["n"].astype(int).astype(str)
        )

        order = (
            data[["m", "n", "Matrix"]]
            .drop_duplicates()
            .sort_values(["m", "n"])
        )
        matrix_order = order["Matrix"].tolist()
        data["Matrix"] = pd.Categorical(
            data["Matrix"], categories=matrix_order, ordered=True
        )

        grouped = (
            data.groupby(["Matrix", "Solver"], observed=False, as_index=False)
            .agg(
                Runtime=("Runtime (s)", "mean"),
                Iterations=("Iterations", "mean"),
                Error=("Relative Error", "mean"),
            )
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        for solver in SOLVER_ORDER:
            subset = grouped[grouped["Solver"] == solver]
            if not subset.empty:
                ax.plot(
                    subset["Matrix"].astype(str),
                    subset["Runtime"],
                    marker="o",
                    label=solver,
                )
        ax.set_title("Runtime across matrix sizes")
        ax.set_xlabel("Matrix size (m × n)")
        ax.set_ylabel("Mean runtime (s)")
        ax.tick_params(axis="x", rotation=60)
        ax.legend()
        save(fig, f"{prefix}_runtime_line.png")

        fig, ax = plt.subplots(figsize=(12, 6))
        for solver in SOLVER_ORDER:
            subset = grouped[grouped["Solver"] == solver]
            if not subset.empty:
                ax.plot(
                    subset["Matrix"].astype(str),
                    subset["Iterations"],
                    marker="o",
                    label=solver,
                )
        ax.set_title("Iteration count across matrix sizes")
        ax.set_xlabel("Matrix size (m × n)")
        ax.set_ylabel("Mean iterations")
        ax.tick_params(axis="x", rotation=60)
        ax.legend()
        save(fig, f"{prefix}_iterations_line.png")

        return

    x = {
        "condition_number": "Condition Number",
        "lambda": "Lambda",
        "sparsity": "Sparsity",
        "sparsity_pattern": "Sparsity Pattern",
        "spectrum": "Spectrum",
    }[experiment]

    xlabel = {
        "Condition Number": "Condition number",
        "Lambda": "Regularisation parameter λ",
        "Sparsity": "Sparsity",
        "Sparsity Pattern": "Sparsity pattern",
        "Spectrum": "Spectrum",
    }[x]

    values = ordered(data, x)
    data[x] = pd.Categorical(data[x], categories=values, ordered=True)

    line_plot(
        data,
        x,
        "Runtime (s)",
        f"Runtime versus {xlabel.lower()}",
        xlabel,
        "Mean runtime (s)",
        f"{prefix}_runtime_line.png",
        log_y=True,
    )

    line_plot(
        data,
        x,
        "Iterations",
        f"Iterations versus {xlabel.lower()}",
        xlabel,
        "Mean iterations",
        f"{prefix}_iterations_line.png",
    )

    line_plot(
        data,
        x,
        "Relative Error",
        f"Relative reconstruction error versus {xlabel.lower()}",
        xlabel,
        "Mean relative reconstruction error",
        f"{prefix}_error_line.png",
    )

    line_plot(
        data,
        x,
        "Residual",
        f"Residual versus {xlabel.lower()}",
        xlabel,
        "Mean relative residual",
        f"{prefix}_residual_line.png",
    )

    grouped = (
        data.groupby([x, "Solver"], observed=False, as_index=False)["Runtime (s)"]
        .mean()
    )

    pivot = grouped.pivot(index=x, columns="Solver", values="Runtime (s)")
    pivot = pivot.reindex(columns=SOLVER_ORDER)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"Mean runtime comparison: {xlabel}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean runtime (s)")
    ax.tick_params(axis="x", rotation=45)
    save(fig, f"{prefix}_runtime_bars.png")

    box_plot(
        data,
        x,
        "Runtime (s)",
        f"Runtime distribution: {xlabel}",
        "Solver",
        "Runtime (s)",
        f"{prefix}_runtime_boxplot.png",
    )


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(OUTPUT_DIR / "small_matrix_results.csv")

    make_experiment_plots(
        results, "matrix_size", "CT_small_matrix_size"
    )
    make_experiment_plots(
        results, "condition_number", "CT_small_condition_number"
    )
    make_experiment_plots(
        results, "lambda", "CT_small_lambda"
    )
    make_experiment_plots(
        results, "sparsity", "CT_small_sparsity"
    )
    make_experiment_plots(
        results, "sparsity_pattern", "CT_small_sparsity_pattern"
    )
    make_experiment_plots(
        results, "spectrum", "CT_small_spectrum"
    )

    solver_bar(
        results,
        "Runtime (s)",
        "Overall mean runtime",
        "Mean runtime (s)",
        "CT_small_overall_runtime_bar.png",
        log=True,
    )

    solver_bar(
        results,
        "Iterations",
        "Overall mean iteration count",
        "Mean iterations",
        "CT_small_overall_iterations_bar.png",
    )

    solver_bar(
        results,
        "Relative Error",
        "Overall mean relative reconstruction error",
        "Mean relative reconstruction error",
        "CT_small_overall_error_bar.png",
    )

    print(f"Plots written to: {PLOT_DIR.resolve()}")


if __name__ == "__main__":
    main()
