import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_DIR = Path(".")
OUTPUT_DIR = Path("CT_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

summary = pd.read_csv(INPUT_DIR / "CT_solver_summary.csv")
runtime = pd.read_csv(INPUT_DIR / "CT_solver_runtime_comparison.csv")
results = pd.read_csv(INPUT_DIR / "CT_solver_results.csv")

solver_order = ["CG", "PCG", "LSQR", "LSMR"]
summary["Solver"] = pd.Categorical(summary["Solver"], categories=solver_order, ordered=True)
summary = summary.sort_values(["Lambda", "Solver"])

plt.figure(figsize=(10, 6))
for solver in solver_order:
    d = summary[summary["Solver"] == solver]
    plt.plot(d["Lambda"], d["Mean_Time"], marker="o", label=solver)
plt.xscale("log")
plt.xlabel("Regularisation parameter λ")
plt.ylabel("Mean runtime (s)")
plt.title("CT Solver Runtime vs Regularisation")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_runtime_vs_lambda.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for solver in solver_order:
    d = summary[summary["Solver"] == solver]
    plt.plot(d["Lambda"], d["Mean_Relative_Reconstruction_Error"], marker="o", label=solver)
plt.xscale("log")
plt.xlabel("Regularisation parameter λ")
plt.ylabel("Mean relative reconstruction error")
plt.title("Reconstruction Accuracy vs Regularisation")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_reconstruction_error_vs_lambda.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for solver in solver_order:
    d = summary[summary["Solver"] == solver]
    plt.plot(d["Lambda"], d["Mean_Relative_Residual"], marker="o", label=solver)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Regularisation parameter λ")
plt.ylabel("Mean relative residual")
plt.title("Data-Fit Residual vs Regularisation")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_residual_vs_lambda.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for solver in solver_order:
    d = summary[summary["Solver"] == solver]
    plt.plot(d["Lambda"], d["Mean_Relative_Normal_Residual"], marker="o", label=solver)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Regularisation parameter λ")
plt.ylabel("Mean relative normal-equation residual")
plt.title("Normal Residual vs Regularisation")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_normal_residual_vs_lambda.png", dpi=300)
plt.close()

best_lambda = summary.groupby("Solver")["Mean_Relative_Reconstruction_Error"].idxmin()
best = summary.loc[best_lambda].set_index("Solver").reindex(solver_order)

plt.figure(figsize=(9, 6))
plt.bar(best.index, best["Mean_Relative_Reconstruction_Error"])
plt.ylabel("Mean relative reconstruction error")
plt.title("Best Reconstruction Error Achieved by Each Solver")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_best_reconstruction_error_bar.png", dpi=300)
plt.close()

best_time_lambda = summary.groupby("Solver")["Mean_Time"].idxmin()
fastest = summary.loc[best_time_lambda].set_index("Solver").reindex(solver_order)

plt.figure(figsize=(9, 6))
plt.bar(fastest.index, fastest["Mean_Time"])
plt.ylabel("Minimum mean runtime (s)")
plt.title("Fastest Observed Runtime by Solver")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_fastest_runtime_bar.png", dpi=300)
plt.close()

lambda_for_comparison = runtime["Lambda"].iloc[0]
r = runtime.iloc[0]

comparisons = {
    "CG": r["LSQR_vs_CG_Time_Ratio"],
    "PCG": r["LSQR_vs_PCG_Time_Ratio"],
    "LSMR": r["LSQR_vs_LSMR_Time_Ratio"],
}

plt.figure(figsize=(9, 6))
plt.bar(list(comparisons.keys()), list(comparisons.values()))
plt.axhline(1.0, linewidth=1)
plt.ylabel("Competitor runtime / LSQR runtime")
plt.title(f"LSQR Runtime Comparison at λ = {lambda_for_comparison:g}")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_lsqr_runtime_ratio.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for solver in solver_order:
    d = summary[summary["Solver"] == solver]
    plt.errorbar(
        d["Mean_Iterations"],
        d["Mean_Relative_Reconstruction_Error"],
        xerr=d["Std_Iterations"],
        marker="o",
        linestyle="-",
        label=solver
    )
plt.xlabel("Mean iterations")
plt.ylabel("Mean relative reconstruction error")
plt.title("Iteration Count vs Reconstruction Error")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_iterations_vs_error.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for solver in solver_order:
    d = summary[summary["Solver"] == solver]
    plt.plot(d["Mean_Time"], d["Mean_Relative_Reconstruction_Error"], marker="o", label=solver)
plt.xlabel("Mean runtime (s)")
plt.ylabel("Mean relative reconstruction error")
plt.title("Runtime–Accuracy Trade-off")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_runtime_accuracy_tradeoff.png", dpi=300)
plt.close()

pivot = summary.pivot(index="Lambda", columns="Solver", values="Mean_Time").reindex(columns=solver_order)

ax = pivot.plot(kind="bar", figsize=(11, 6))
ax.set_xlabel("Regularisation parameter λ")
ax.set_ylabel("Mean runtime (s)")
ax.set_title("Runtime Comparison Across Solvers and λ")
ax.legend(title="Solver")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_runtime_grouped_bar.png", dpi=300)
plt.close()

with open(OUTPUT_DIR / "plot_summary.txt", "w") as f:
    f.write("CT solver plotting summary\n")
    f.write(f"Rows in results file: {len(results)}\n")
    f.write(f"Lambdas: {sorted(summary['Lambda'].unique())}\n")
    f.write(f"Solvers: {solver_order}\n")
    f.write(f"Image unknowns N: {results['N'].iloc[0]}\n")
    f.write(f"Measurements M: {results['M'].iloc[0]}\n")
    f.write(f"Views: {results['Views'].iloc[0]}\n")
    f.write("\nBest reconstruction-error row for each solver:\n")
    f.write(best[["Lambda", "Mean_Relative_Reconstruction_Error", "Mean_Time", "Mean_Iterations"]].to_string())
    f.write("\n\nFastest-runtime row for each solver:\n")
    f.write(fastest[["Lambda", "Mean_Time", "Mean_Relative_Reconstruction_Error", "Mean_Iterations"]].to_string())

print(f"Created {len(list(OUTPUT_DIR.glob('*.png')))} plots in {OUTPUT_DIR}/")
for p in sorted(OUTPUT_DIR.glob("*.png")):
    print(p.name)
