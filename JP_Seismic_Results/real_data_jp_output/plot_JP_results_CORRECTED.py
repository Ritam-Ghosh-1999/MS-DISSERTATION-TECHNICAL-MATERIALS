import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SOLVERS = ["CG", "PCG", "LSQR", "LSMR"]


def load_data(results_path, summary_path, runtime_path):
    results = pd.read_csv(results_path)
    summary = pd.read_csv(summary_path)
    runtime = pd.read_csv(runtime_path)

    summary["Solver"] = pd.Categorical(
        summary["Solver"], categories=SOLVERS, ordered=True
    )

    lambda_col = "Lambda Multiplier" if "Lambda Multiplier" in summary.columns else "Lambda"
    runtime_lambda_col = (
        "Lambda Multiplier"
        if "Lambda Multiplier" in runtime.columns
        else "Lambda"
    )

    summary["_PlotLambda"] = summary[lambda_col]
    runtime["_PlotLambda"] = runtime[runtime_lambda_col]

    summary = summary.sort_values(["_PlotLambda", "Solver"])
    runtime = runtime.sort_values("_PlotLambda")
    return results, summary, runtime


def save_line_plot(summary, y, ylabel, title, filename, log_y=False):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for solver in SOLVERS:
        d = summary[summary["Solver"] == solver].sort_values("Lambda")
        ax.plot(d["_PlotLambda"], d[y], marker="o", linewidth=1.8, markersize=4, label=solver)

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Regularisation parameter λ")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_bar(values, ylabel, title, filename, log_y=False):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    values = values.reindex(SOLVERS)

    ax.bar(values.index, values.values)
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_runtime_comparison(runtime, filename):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for solver in SOLVERS:
        ax.plot(
            runtime["_PlotLambda"],
            runtime[solver],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=solver,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Regularisation parameter λ")
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title("Solver runtime across regularisation parameters")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_speedup(runtime, filename):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    d = runtime.copy()
    x = list(range(len(d)))

    lsqr_speedup = 1 / d["LSMR_vs_LSQR_Time_Ratio"]
    cg_speedup = 1 / d["LSMR_vs_CG_Time_Ratio"]
    pcg_speedup = 1 / d["LSMR_vs_PCG_Time_Ratio"]

    width = 0.25

    ax.bar(
        [i - width for i in x],
        lsqr_speedup,
        width=width,
        label="vs LSQR",
    )
    ax.bar(
        x,
        cg_speedup,
        width=width,
        label="vs CG",
    )
    ax.bar(
        [i + width for i in x],
        pcg_speedup,
        width=width,
        label="vs PCG",
    )

    ax.axhline(1.0, linewidth=1.2, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{v:.0e}" for v in d["_PlotLambda"]]
    )
    ax.set_xlabel("Regularisation parameter λ")
    ax.set_ylabel("Competitor runtime / LSMR runtime")
    ax.set_title("LSMR speedup relative to competing solvers")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_speedup_line(runtime, filename):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    d = runtime.copy()

    ax.plot(
        d["_PlotLambda"],
        1 / d["LSMR_vs_LSQR_Time_Ratio"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="vs LSQR",
    )
    ax.plot(
        d["_PlotLambda"],
        1 / d["LSMR_vs_CG_Time_Ratio"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="vs CG",
    )
    ax.plot(
        d["_PlotLambda"],
        1 / d["LSMR_vs_PCG_Time_Ratio"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="vs PCG",
    )

    ax.axhline(1.0, linewidth=1.2, linestyle="--", label="Equal runtime")
    ax.set_xscale("log")
    ax.set_xlabel("Regularisation parameter λ")
    ax.set_ylabel("Competitor runtime / LSMR runtime")
    ax.set_title("LSMR speedup relative to competing solvers")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dissertation plots for the JP seismic tomography benchmark."
    )
    parser.add_argument(
        "--results",
        default="JP_lambda_sweep_results.csv",
        help="Detailed results CSV",
    )
    parser.add_argument(
        "--summary",
        default="JP_lambda_sweep_summary.csv",
        help="Summary CSV",
    )
    parser.add_argument(
        "--runtime",
        default="JP_solver_runtime_comparison.csv",
        help="Runtime comparison CSV",
    )
    parser.add_argument(
        "--output",
        default="JP_plots",
        help="Directory for generated plots",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results, summary, runtime = load_data(
        args.results, args.summary, args.runtime
    )

    save_line_plot(
        summary,
        "Mean_Time",
        "Mean runtime (s)",
        "Solver runtime across regularisation parameters",
        out / "01_runtime_vs_lambda.png",
    )

    save_line_plot(
        summary,
        "Mean_Iterations",
        "Mean iterations",
        "Solver iterations across regularisation parameters",
        out / "02_iterations_vs_lambda.png",
    )

    save_line_plot(
        summary,
        "Mean_Relative_Residual",
        "Mean relative residual",
        "Relative residual across regularisation parameters",
        out / "03_relative_residual_vs_lambda.png",
        log_y=True,
    )

    normal_residual_col = (
        "Mean_Relative_Normal_Equation_Residual"
        if "Mean_Relative_Normal_Equation_Residual" in summary.columns
        else "Mean_Relative_Normal_Residual"
    )

    save_line_plot(
        summary,
        normal_residual_col,
        "Mean relative normal-equation residual",
        "Relative normal-equation residual across regularisation parameters",
        out / "04_relative_normal_residual_vs_lambda.png",
        log_y=True,
    )

    save_line_plot(
        summary,
        "Mean_Objective",
        "Mean objective",
        "Objective value across regularisation parameters",
        out / "05_objective_vs_lambda.png",
        log_y=True,
    )

    mean_runtime = summary.groupby("Solver", observed=True)["Mean_Time"].mean()
    save_bar(
        mean_runtime,
        "Mean runtime (s)",
        "Average runtime across the λ sweep",
        out / "06_average_runtime_by_solver.png",
    )

    mean_iterations = summary.groupby("Solver", observed=True)["Mean_Iterations"].mean()
    save_bar(
        mean_iterations,
        "Mean iterations",
        "Average iterations across the λ sweep",
        out / "07_average_iterations_by_solver.png",
    )

    mean_rel_residual = summary.groupby("Solver", observed=True)[
        "Mean_Relative_Residual"
    ].mean()
    save_bar(
        mean_rel_residual,
        "Mean relative residual",
        "Average relative residual across the λ sweep",
        out / "08_average_relative_residual_by_solver.png",
        log_y=True,
    )

    save_speedup(runtime, out / "09_lsmr_speedup_vs_competitors.png")
    save_speedup_line(runtime, out / "10_lsmr_speedup_vs_competitors_line.png")



    print(f"Generated {len(list(out.glob('*.png')))} plots in: {out}")
    for p in sorted(out.glob("*.png")):
        print(p.name)


if __name__ == "__main__":
    main()
