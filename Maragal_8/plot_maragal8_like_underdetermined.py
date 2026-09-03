#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt


# The three CSV files produced by the Maragal_8 benchmark are expected
# inside this directory.
INPUT_DIR = os.environ.get(
    "ERP_OUTPUT_DIR",
    "real_data_pcg_output"
)

PLOT_DIR = os.path.join(INPUT_DIR, "plots")
SUMMARY_DIR = os.path.join(INPUT_DIR, "summaries")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# Direct is not part of this benchmark.
SOLVERS = ["PCG", "LSQR", "LSMR"]


def load_results():
    results_file = os.path.join(
        INPUT_DIR,
        "Maragal_8_lambda_sweep_results.csv"
    )

    summary_file = os.path.join(
        INPUT_DIR,
        "Maragal_8_lambda_sweep_summary.csv"
    )

    comparison_file = os.path.join(
        INPUT_DIR,
        "Maragal_8_pcg_lsmr_comparison.csv"
    )

    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"Could not find: {results_file}"
        )

    if not os.path.exists(summary_file):
        raise FileNotFoundError(
            f"Could not find: {summary_file}"
        )

    results = pd.read_csv(results_file)
    summary = pd.read_csv(summary_file)

    comparison = None
    if os.path.exists(comparison_file):
        comparison = pd.read_csv(comparison_file)

    required_results = [
        "Lambda",
        "Solver",
        "Time (s)",
        "Iterations",
        "Relative Residual"
    ]

    missing = [
        column for column in required_results
        if column not in results.columns
    ]

    if missing:
        raise ValueError(
            "Results CSV is missing required columns: "
            + ", ".join(missing)
        )

    required_summary = [
        "Lambda",
        "Solver",
        "Mean_Time",
        "Std_Time",
        "Mean_Iterations",
        "Mean_Relative_Residual"
    ]

    missing = [
        column for column in required_summary
        if column not in summary.columns
    ]

    if missing:
        raise ValueError(
            "Summary CSV is missing required columns: "
            + ", ".join(missing)
        )

    results = results[
        results["Solver"].isin(SOLVERS)
    ].copy()

    summary = summary[
        summary["Solver"].isin(SOLVERS)
    ].copy()

    return results, summary, comparison


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def solver_order(data):
    present = set(data["Solver"].dropna().unique())
    return [
        solver for solver in SOLVERS
        if solver in present
    ]


def create_summary(results):
    return (
        results
        .groupby(["Lambda", "Solver"], as_index=False)
        .agg(
            Mean_Time=("Time (s)", "mean"),
            Std_Time=("Time (s)", "std"),
            Mean_Iterations=("Iterations", "mean"),
            Std_Iterations=("Iterations", "std"),
            Mean_Residual=("Residual", "mean"),
            Mean_Relative_Residual=("Relative Residual", "mean"),
            Std_Relative_Residual=("Relative Residual", "std"),
            Mean_Normal_Residual=(
                "Normal Equation Residual",
                "mean"
            ),
            Mean_Relative_Normal_Residual=(
                "Relative Normal Equation Residual",
                "mean"
            ),
            Mean_Objective=("Objective", "mean")
        )
    )


def save_summary(data, filename):
    if not data.empty:
        data.to_csv(
            os.path.join(SUMMARY_DIR, filename),
            index=False
        )


def plot_line(
    data,
    parameter,
    metric,
    ylabel,
    filename,
    title
):
    if data.empty:
        return

    plt.figure(figsize=(9, 6))

    for solver in solver_order(data):
        subset = (
            data[data["Solver"] == solver]
            .groupby(parameter, as_index=False)[metric]
            .mean()
            .sort_values(parameter)
        )

        plt.plot(
            subset[parameter],
            subset[metric],
            marker="o",
            label=solver
        )

    plt.xlabel(parameter)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    save_plot(filename)


def plot_bar(
    data,
    parameter,
    metric,
    ylabel,
    filename,
    title
):
    if data.empty:
        return

    pivot = (
        data
        .groupby([parameter, "Solver"])[metric]
        .mean()
        .unstack()
    )

    if pivot.empty:
        return

    pivot = pivot.reindex(
        columns=solver_order(data)
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_xlabel(parameter)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Solver")

    save_plot(filename)


def plot_box(
    data,
    parameter,
    metric,
    ylabel,
    filename,
    title
):
    if data.empty:
        return

    groups = []
    labels = []

    parameter_values = list(
        data[parameter].dropna().unique()
    )

    try:
        parameter_values = sorted(parameter_values)
    except TypeError:
        pass

    for value in parameter_values:
        for solver in solver_order(data):
            values = (
                data[
                    (data[parameter] == value)
                    & (data["Solver"] == solver)
                ][metric]
                .dropna()
                .values
            )

            if len(values) > 0:
                groups.append(values)
                labels.append(
                    f"{value}\n{solver}"
                )

    if not groups:
        return

    plt.figure(figsize=(12, 6))

    plt.boxplot(groups)

    plt.xticks(
        range(1, len(labels) + 1),
        labels,
        rotation=45,
        ha="right"
    )

    plt.xlabel(parameter)
    plt.ylabel(ylabel)
    plt.title(title)

    save_plot(filename)


def plot_runtime_vs_residual(
    data,
    experiment_name
):
    if data.empty:
        return

    plt.figure(figsize=(9, 6))

    for solver in solver_order(data):
        subset = data[
            data["Solver"] == solver
        ]

        plt.scatter(
            subset["Time (s)"],
            subset["Relative Residual"],
            label=solver,
            alpha=0.7
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Relative residual")
    plt.title(
        f"Runtime vs Relative Residual: "
        f"{experiment_name}"
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    filename = (
        "maragal_8_runtime_vs_relative_residual.png"
    )

    save_plot(filename)


def plot_runtime_comparison(comparison):
    if comparison is None or comparison.empty:
        return

    required = [
        "Lambda",
        "PCG",
        "LSMR"
    ]

    if not all(
        column in comparison.columns
        for column in required
    ):
        return

    data = comparison.sort_values("Lambda")

    plt.figure(figsize=(9, 6))

    for solver in ["PCG", "LSMR"]:
        plt.plot(
            data["Lambda"],
            data[solver],
            marker="o",
            label=solver
        )

    plt.xlabel(r"Regularisation parameter $\lambda$")
    plt.ylabel("Mean runtime (s)")
    plt.title(
        "Maragal_8: PCG vs LSMR Runtime"
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    save_plot(
        "maragal_8_pcg_vs_lsmr_runtime.png"
    )


def plot_pcg_speedup(comparison):
    if comparison is None or comparison.empty:
        return

    required = [
        "Lambda",
        "PCG_vs_LSMR_Time_Ratio"
    ]

    if not all(
        column in comparison.columns
        for column in required
    ):
        return

    data = comparison.sort_values("Lambda").copy()

    # LSMR time / PCG time:
    # values above 1 mean PCG is faster.
    data["PCG_Speedup"] = (
        1.0 / data["PCG_vs_LSMR_Time_Ratio"]
    )

    plt.figure(figsize=(9, 6))

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        label="Equal runtime"
    )

    plt.plot(
        data["Lambda"],
        data["PCG_Speedup"],
        marker="o",
        linewidth=2
    )

    plt.xlabel(r"Regularisation parameter $\lambda$")
    plt.ylabel("PCG speedup over LSMR")
    plt.title(
        "Maragal_8: PCG Speedup Relative to LSMR"
    )

    plt.xscale("log")
    plt.legend()

    save_plot(
        "maragal_8_pcg_speedup_over_lsmr.png"
    )


def main():

    print(f"Input directory   : {INPUT_DIR}")
    print(f"Plot directory    : {PLOT_DIR}")
    print(f"Summary directory : {SUMMARY_DIR}")
    print()

    results, summary, comparison = load_results()

    print(
        "Solvers included:",
        ", ".join(solver_order(results))
    )
    print()

    # Recreate the summary directly from the raw results.
    # This uses the exact naming convention of the benchmark CSVs.
    calculated_summary = create_summary(results)

    save_summary(
        calculated_summary,
        "maragal_8_summary_for_plotting.csv"
    )

    metrics = [
        (
            "Time (s)",
            "Mean runtime (s)",
            "runtime"
        ),
        (
            "Iterations",
            "Mean iterations",
            "iterations"
        ),
        (
            "Relative Residual",
            "Mean relative residual",
            "relative_residual"
        )
    ]

    for metric, ylabel, metric_name in metrics:

        plot_line(
            results,
            "Lambda",
            metric,
            ylabel,
            f"maragal_8_{metric_name}_line.png",
            f"{metric} vs Lambda: Maragal_8"
        )

        plot_bar(
            results,
            "Lambda",
            metric,
            ylabel,
            f"maragal_8_{metric_name}_bar.png",
            f"{metric} vs Lambda: Maragal_8"
        )

        plot_box(
            results,
            "Lambda",
            metric,
            ylabel,
            f"maragal_8_{metric_name}_box.png",
            f"{metric} vs Lambda: Maragal_8"
        )

    plot_runtime_vs_residual(
        results,
        "Maragal_8"
    )

    # Additional plots directly relevant to the
    # PCG-versus-LSMR comparison.
    plot_runtime_comparison(comparison)
    plot_pcg_speedup(comparison)

    print()
    print("Post-processing complete.")
    print(f"Summaries saved to: {SUMMARY_DIR}")
    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
