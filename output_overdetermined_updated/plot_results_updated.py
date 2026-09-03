#!/usr/bin/env python

import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT_DIR = os.environ.get(
    "ERP_OUTPUT_DIR",
    "output_overdetermined_updated"
)

PLOT_DIR = os.path.join(
    INPUT_DIR,
    "plots"
)

SUMMARY_DIR = os.path.join(
    INPUT_DIR,
    "summaries"
)

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


def load_all_results():
    result_files = []

    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in files:
            if filename == "results_final.csv":
                result_files.append(
                    os.path.join(root, filename)
                )

    if not result_files:
        print("No completed result files found.")
        return pd.DataFrame()

    frames = []

    for filepath in sorted(result_files):
        try:
            df = pd.read_csv(filepath)

            if not df.empty:
                frames.append(df)
                print(
                    f"Loaded: {filepath} "
                    f"({len(df)} rows)"
                )

        except Exception as error:
            print(
                f"Could not read {filepath}: {error}"
            )

    if not frames:
        print("No valid result files found.")
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True
    )


def create_summary(results):
    if results.empty:
        return pd.DataFrame()

    columns = [
        "m",
        "n",
        "Condition Parameter",
        "Sparsity",
        "Sparsity Pattern",
        "Spectrum",
        "Lambda",
        "Solver"
    ]

    return (
        results
        .groupby(columns, as_index=False)
        .agg(
            Mean_Time=("Time (s)", "mean"),
            Std_Time=("Time (s)", "std"),
            Mean_Iterations=("Iterations", "mean"),
            Std_Iterations=("Iterations", "std"),
            Mean_Residual=("Residual", "mean"),
            Std_Residual=("Residual", "std"),
            Mean_Relative_Error=("Relative Error", "mean"),
            Std_Relative_Error=("Relative Error", "std")
        )
    )


def create_parameter_summary(results, parameter):
    if results.empty:
        return pd.DataFrame()

    return (
        results
        .groupby(
            [parameter, "Solver"],
            as_index=False
        )
        .agg(
            Mean_Time=("Time (s)", "mean"),
            Std_Time=("Time (s)", "std"),
            Mean_Iterations=("Iterations", "mean"),
            Std_Iterations=("Iterations", "std"),
            Mean_Residual=("Residual", "mean"),
            Std_Residual=("Residual", "std"),
            Mean_Relative_Error=("Relative Error", "mean"),
            Std_Relative_Error=("Relative Error", "std")
        )
    )


def save_summary(data, filename):
    if data.empty:
        return

    data.to_csv(
        os.path.join(SUMMARY_DIR, filename),
        index=False
    )


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_line(data, x, metric, ylabel, filename, title):
    if data.empty:
        return

    plt.figure(figsize=(9, 6))

    for solver in sorted(data["Solver"].unique()):
        subset = (
            data[data["Solver"] == solver]
            .groupby(x, as_index=False)[metric]
            .mean()
            .sort_values(x)
        )

        plt.plot(
            subset[x],
            subset[metric],
            marker="o",
            label=solver
        )

    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    save_plot(filename)


def plot_bar(data, x, metric, ylabel, filename, title):
    if data.empty:
        return

    pivot = (
        data
        .groupby([x, "Solver"])[metric]
        .mean()
        .unstack()
    )

    if pivot.empty:
        return

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Solver")

    save_plot(filename)


def plot_box(data, x, metric, ylabel, filename, title):
    if data.empty:
        return

    groups = []
    labels = []

    x_values = list(data[x].dropna().unique())

    try:
        x_values = sorted(x_values)
    except TypeError:
        pass

    for value in x_values:
        for solver in sorted(data["Solver"].unique()):
            values = (
                data[
                    (data[x] == value) &
                    (data["Solver"] == solver)
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

    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)

    save_plot(filename)


def plot_runtime_vs_error(data, experiment_name):
    if data.empty:
        return

    plt.figure(figsize=(9, 6))

    for solver in sorted(data["Solver"].unique()):
        subset = data[
            data["Solver"] == solver
        ]

        plt.scatter(
            subset["Time (s)"],
            subset["Relative Error"],
            label=solver,
            alpha=0.7
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Relative Error")
    plt.title(
        f"Runtime vs Relative Error: {experiment_name}"
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    filename = (
        experiment_name
        .lower()
        .replace(" ", "_")
        + "_runtime_vs_error.png"
    )

    save_plot(filename)


def run_experiment_plots(results, experiment_name, x):
    data = results[
        results["Experiment"] == experiment_name
    ].copy()

    if data.empty:
        print(
            f"No results for {experiment_name}"
        )
        return

    safe_name = (
        experiment_name
        .lower()
        .replace(" ", "_")
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
            "Relative Error",
            "Mean relative error",
            "relative_error"
        )
    ]

    for metric, ylabel, metric_name in metrics:

        plot_line(
            data,
            x,
            metric,
            ylabel,
            f"{safe_name}_{metric_name}_line.png",
            f"{metric} vs {x}: {experiment_name}"
        )

        plot_bar(
            data,
            x,
            metric,
            ylabel,
            f"{safe_name}_{metric_name}_bar.png",
            f"{metric} vs {x}: {experiment_name}"
        )

        plot_box(
            data,
            x,
            metric,
            ylabel,
            f"{safe_name}_{metric_name}_box.png",
            f"{metric} vs {x}: {experiment_name}"
        )

    plot_runtime_vs_error(
        data,
        experiment_name
    )


def main():
    print(f"Input directory: {INPUT_DIR}")
    print(f"Plot directory: {PLOT_DIR}")
    print(f"Summary directory: {SUMMARY_DIR}")
    print()

    results = load_all_results()

    if results.empty:
        print("No results available.")
        return

    summary = create_summary(results)
    save_summary(summary, "summary.csv")

    experiment_parameters = {
        "Condition Parameter": "Condition Parameter",
        "Sparsity": "Sparsity",
        "Matrix Size": "m",
        "Lambda": "Lambda",
        "Sparsity Pattern": "Sparsity Pattern",
        "Spectrum": "Spectrum"
    }

    for experiment_name, parameter in experiment_parameters.items():

        experiment_data = results[
            results["Experiment"] == experiment_name
        ]

        parameter_summary = create_parameter_summary(
            experiment_data,
            parameter
        )

        filename = (
            experiment_name
            .lower()
            .replace(" ", "_")
            + "_summary.csv"
        )

        save_summary(
            parameter_summary,
            filename
        )

        run_experiment_plots(
            results,
            experiment_name,
            parameter
        )

    print()
    print("Post-processing complete.")
    print(f"Summaries saved to: {SUMMARY_DIR}")
    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
