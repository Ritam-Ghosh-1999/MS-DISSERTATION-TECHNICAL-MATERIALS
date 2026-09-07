#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import lsmr

np.random.seed(42)

OUTPUT_DIR = os.environ.get(
    "ERP_OUTPUT_DIR",
    "output_test"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("Output directory:", OUTPUT_DIR)


# In[21]:



def generate_matrix(
    m,
    n,
    condition_parameter=10,
    sparsity=0.0,
    sparsity_pattern="random",
    spectrum="exponential",
    rng=None
):

    if rng is None:
        rng = np.random.default_rng(42)

    density = max(
        1.0 - sparsity,
        1e-4
    )

    if spectrum == "linear":

        scales = np.linspace(
            1.0,
            1.0 / condition_parameter,
            n
        )

    elif spectrum == "exponential":

        scales = np.geomspace(
            1.0,
            1.0 / condition_parameter,
            n
        )

    elif spectrum == "polynomial":

        x = np.linspace(
            0,
            1,
            n
        )

        scales = (
            1.0 / condition_parameter
            + (
                1.0
                - 1.0 / condition_parameter
            ) * (1 - x) ** 2
        )

    elif spectrum == "clustered":

        k = n // 3

        scales = np.concatenate([
            np.ones(k),
            np.linspace(
                0.3,
                3.0 / condition_parameter,
                k
            ),
            np.full(
                n - 2 * k,
                1.0 / condition_parameter
            )
        ])

    else:

        raise ValueError(
            "Unknown spectrum"
        )

    if sparsity_pattern == "random":

        A = sparse.random(
            m,
            n,
            density=density,
            format="csr",
            random_state=rng,
            data_rvs=rng.standard_normal
        )

    elif sparsity_pattern == "banded":

        target_nnz_per_row = max(
            1,
            int(round(density * n))
        )

        rows = []
        cols = []
        vals = []

        for i in range(m):

            if m > 1:
                centre = (
                    (n - 1) * i / (m - 1)
                )
            else:
                centre = 0.0

            half_width = (
                target_nnz_per_row // 2
            )

            centre_col = int(
                round(centre)
            )

            left = (
                centre_col
                - half_width
            )

            left = max(
                0,
                left
            )

            right = (
                left
                + target_nnz_per_row
            )

            if right > n:

                right = n

                left = max(
                    0,
                    right
                    - target_nnz_per_row
                )

            c = np.arange(
                left,
                right
            )

            rows.extend(
                [i] * len(c)
            )

            cols.extend(
                c
            )

            vals.extend(
                rng.standard_normal(
                    len(c)
                )
            )

        A = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(m, n)
        )

    elif sparsity_pattern == "block":

        block_rows = max(
            1,
            min(50, m // 20)
        )

        block_cols = max(
            1,
            min(50, n // 20)
        )

        n_block_rows = int(
            np.ceil(
                m / block_rows
            )
        )

        n_block_cols = int(
            np.ceil(
                n / block_cols
            )
        )

        total_blocks = (
            n_block_rows
            * n_block_cols
        )

        target_nnz = max(
            1,
            int(
                round(
                    density * m * n
                )
            )
        )

        block_area = (
            block_rows
            * block_cols
        )

        n_active_blocks = int(
            np.ceil(
                target_nnz
                / block_area
            )
        )

        n_active_blocks = min(
            n_active_blocks,
            total_blocks
        )

        active_block_ids = set()

        for col_block in range(
            n_block_cols
        ):

            row_block = (
                col_block
                % n_block_rows
            )

            block_id = (
                row_block
                * n_block_cols
                + col_block
            )

            active_block_ids.add(
                block_id
            )

        remaining_blocks = [
            block_id
            for block_id in range(
                total_blocks
            )
            if block_id
            not in active_block_ids
        ]

        remaining_needed = (
            n_active_blocks
            - len(active_block_ids)
        )

        if remaining_needed > 0:

            additional_blocks = rng.choice(
                remaining_blocks,
                size=remaining_needed,
                replace=False
            )

            active_block_ids.update(
                additional_blocks.tolist()
            )

        rows = []
        cols = []
        vals = []

        for block_id in active_block_ids:

            block_i = (
                block_id
                // n_block_cols
            )

            block_j = (
                block_id
                % n_block_cols
            )

            row_start = (
                block_i
                * block_rows
            )

            row_end = min(
                row_start
                + block_rows,
                m
            )

            col_start = (
                block_j
                * block_cols
            )

            col_end = min(
                col_start
                + block_cols,
                n
            )

            rr = np.arange(
                row_start,
                row_end
            )

            cc = np.arange(
                col_start,
                col_end
            )

            R, C = np.meshgrid(
                rr,
                cc,
                indexing="ij"
            )

            rows.extend(
                R.ravel()
            )

            cols.extend(
                C.ravel()
            )

            vals.extend(
                rng.standard_normal(
                    R.size
                )
            )

        A = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(m, n)
        )

    else:

        raise ValueError(
            "Unknown sparsity pattern"
        )

    A = (
        A
        @ sparse.diags(scales)
    )

    return A.tocsr()

def generate_problem(
    m,
    n,
    condition_parameter=10,
    sparsity=0.0,
    sparsity_pattern="random",
    spectrum="exponential",
    noise_level=0.01,
    random_seed=42
):

    seed_sequence = np.random.SeedSequence(random_seed)

    matrix_seed, x_seed, noise_seed = seed_sequence.spawn(3)

    matrix_rng = np.random.default_rng(matrix_seed)
    x_rng = np.random.default_rng(x_seed)
    noise_rng = np.random.default_rng(noise_seed)

    A = generate_matrix(
        m=m,
        n=n,
        condition_parameter=condition_parameter,
        sparsity=sparsity,
        sparsity_pattern=sparsity_pattern,
        spectrum=spectrum,
        rng=matrix_rng
    )

    x_true = x_rng.standard_normal(n)

    b = A @ x_true

    z = noise_rng.standard_normal(m)

    signal_norm = np.linalg.norm(b)

    noise = (
        noise_level
        * signal_norm
        * z
        / np.linalg.norm(z)
    )

    b = b + noise

    return A, b, x_true
# In[53]:


# Direct Solver

def solve_direct(A, b, x_true, lambda_=1.0):

    start = time.perf_counter()

    ATA = A.T @ A
    rhs = A.T @ b

    x = sparse.linalg.spsolve(
        ATA + lambda_ * sparse.eye(A.shape[1], format="csr"),
        rhs
    )

    runtime = time.perf_counter() - start

    residual = np.linalg.norm(A @ x - b)

    relative_error = (
        np.linalg.norm(x - x_true)
        / np.linalg.norm(x_true)
    )

    return {
        "Solver": "Direct",
        "Time (s)": runtime,
        "Iterations": 1,
        "Residual": residual,
        "Relative Error": relative_error
    }


# CG - Matrix-Free Iterative Solver

def solve_cg(A, b, x_true, lambda_=1.0):

    n = A.shape[1]

    iterations = [0]

    def callback(xk):
        iterations[0] += 1

    start = time.perf_counter()

    # Compute A^T b without forming A^T A
    rhs = A.T @ b

    # Matrix-free application of:
    #
    #     (A^T A + lambda I)x
    #
    # A^T A is NEVER explicitly constructed.
    def matvec(x):

        Ax = A @ x
        AT_Ax = A.T @ Ax

        return AT_Ax + lambda_ * x

    H = sparse.linalg.LinearOperator(
        shape=(n, n),
        matvec=matvec,
        dtype=A.dtype
    )

    x, info = cg(
        H,
        rhs,
        rtol=1e-8,
        atol=0.0,
        maxiter=5000,
        callback=callback
    )

    runtime = time.perf_counter() - start

    residual = np.linalg.norm(A @ x - b)

    relative_error = (
        np.linalg.norm(x - x_true)
        / np.linalg.norm(x_true)
    )

    return {
        "Solver": "CG",
        "Time (s)": runtime,
        "Iterations": iterations[0],
        "Residual": residual,
        "Relative Error": relative_error,
        "Info": info
    }

# LSQR

def solve_lsqr(A, b, x_true, lambda_=1.0):

    start = time.perf_counter()

    result = lsqr(
        A,
        b,
        damp=np.sqrt(lambda_),
        atol=1e-8,
        btol=1e-8,
        iter_lim=5000
    )

    runtime = time.perf_counter() - start

    x = result[0]

    residual = np.linalg.norm(A @ x - b)

    relative_error = (
        np.linalg.norm(x - x_true)
        / np.linalg.norm(x_true)
    )

    return {
        "Solver": "LSQR",
        "Time (s)": runtime,
        "Iterations": result[2],
        "Residual": residual,
        "Relative Error": relative_error
    }


# LSMR

def solve_lsmr(A, b, x_true, lambda_=1.0):

    start = time.perf_counter()

    result = lsmr(
        A,
        b,
        damp=np.sqrt(lambda_),
        atol=1e-8,
        btol=1e-8,
        maxiter=5000
    )

    runtime = time.perf_counter() - start

    x = result[0]

    residual = np.linalg.norm(A @ x - b)

    relative_error = (
        np.linalg.norm(x - x_true)
        / np.linalg.norm(x_true)
    )

    return {
        "Solver": "LSMR",
        "Time (s)": runtime,
        "Iterations": result[2],
        "Residual": residual,
        "Relative Error": relative_error
    }


SOLVER_FUNCTIONS = {
    "Direct": solve_direct,
    "CG": solve_cg,
    "LSQR": solve_lsqr,
    "LSMR": solve_lsmr
}


# In[10]:


def benchmark_solver(
    solver_function,
    A,
    b,
    x_true,
    lambda_=1.0
):

    return solver_function(
        A,
        b,
        x_true,
        lambda_
    )


def benchmark_single_solver(
    solver_name,
    m,
    n,
    condition_parameter=10,
    sparsity=0.0,
    sparsity_pattern="random",
    spectrum="exponential",
    lambda_=1.0,
    repeats=10,
    noise_level=0.01,
    random_seed=42
):

    if solver_name not in SOLVER_FUNCTIONS:

        raise ValueError(
            f"Unknown solver: {solver_name}. "
            f"Choose from {list(SOLVER_FUNCTIONS.keys())}"
        )

    solver_function = SOLVER_FUNCTIONS[solver_name]

    rows = []

    for run in range(repeats):

        A, b, x_true = generate_problem(
            m=m,
            n=n,
            condition_parameter=condition_parameter,
            sparsity=sparsity,
            sparsity_pattern=sparsity_pattern,
            spectrum=spectrum,
            noise_level=noise_level,
            random_seed=random_seed + run
        )

        result = benchmark_solver(
            solver_function=solver_function,
            A=A,
            b=b,
            x_true=x_true,
            lambda_=lambda_
        )

        result["Run"] = run + 1
        result["m"] = m
        result["n"] = n
        result["Condition Parameter"] = condition_parameter
        result["Sparsity"] = sparsity
        result["Sparsity Pattern"] = sparsity_pattern
        result["Spectrum"] = spectrum
        result["Lambda"] = lambda_

        rows.append(result)

    df = pd.DataFrame(rows)

    cols = [
        "m",
        "n",
        "Condition Parameter",
        "Sparsity",
        "Sparsity Pattern",
        "Spectrum",
        "Lambda",
        "Solver",
        "Run",
        "Time (s)",
        "Iterations",
        "Residual",
        "Relative Error"
    ]

    return df[cols]

# In[70]:


def average_results(results):

    return (
        results
        .groupby(
            [
                "Solver",
                "m",
                "n",
                "Condition Parameter",
                "Sparsity",
                "Sparsity Pattern",
                "Spectrum",
                "Lambda"
            ],
            as_index=False
        )
        .agg({
            "Time (s)": "mean",
            "Iterations": "mean",
            "Residual": "mean",
            "Relative Error": "mean"
        })
    )


# In[72]:


def parameter_sweep(
    m_values,
    n_values,
    condition_parameters,
    sparsity_values,
    lambda_values,
    sparsity_patterns=("random", "banded", "block"),
    spectra=("linear", "exponential", "polynomial", "clustered"),
    repeats=10,
    noise_level=0.01
):

    configurations = []

    experiment = 1

    for m in m_values:

        for n in n_values:

            for condition_parameter in condition_parameters:

                for sparsity in sparsity_values:

                    for lambda_ in lambda_values:

                        for sparsity_pattern in sparsity_patterns:

                            for spectrum in spectra:

                                configurations.append({
                                    "Experiment": experiment,
                                    "m": m,
                                    "n": n,
                                    "Condition Parameter": condition_parameter,
                                    "Sparsity": sparsity,
                                    "Sparsity Pattern": sparsity_pattern,
                                    "Spectrum": spectrum,
                                    "Lambda": lambda_,
                                    "Repeats": repeats,
                                    "Noise Level": noise_level
                                })

                                experiment += 1

    return configurations

# In[35]:


# Summary results

def create_summary(results):

    if results.empty:
        return pd.DataFrame()

    grouping_columns = [
        "m",
        "n",
        "Condition Parameter",
        "Sparsity",
        "Sparsity Pattern",
        "Spectrum",
        "Lambda",
        "Solver"
    ]

    summary = (
        results
        .groupby(
            grouping_columns,
            as_index=False
        )
        .agg(
            Mean_Time=("Time (s)", "mean"),
            Std_Time=("Time (s)", "std"),
            Mean_Iterations=("Iterations", "mean"),
            Mean_Residual=("Residual", "mean"),
            Mean_Relative_Error=("Relative Error", "mean")
        )
    )

    return summary


# In[37]:


# Condition Number Summary

def create_condition_summary(results):

    if results.empty:
        return pd.DataFrame()

    condition_summary = (
        results
        .groupby(
            [
                "Condition Parameter",
                "Solver"
            ],
            as_index=False
        )
        .agg(
            Mean_Time=("Time (s)", "mean"),
            Std_Time=("Time (s)", "std"),
            Mean_Iterations=("Iterations", "mean"),
            Mean_Residual=("Residual", "mean"),
            Mean_Relative_Error=("Relative Error", "mean")
        )
    )

    return condition_summary


# In[39]:


# Sparsity Summary

def create_sparsity_summary(results):

    if results.empty:
        return pd.DataFrame()

    sparsity_summary = (
        results
        .groupby(
            [
                "Sparsity",
                "Solver"
            ],
            as_index=False
        )
        .agg(
            Mean_Time=("Time (s)", "mean"),
            Std_Time=("Time (s)", "std"),
            Mean_Iterations=("Iterations", "mean"),
            Mean_Residual=("Residual", "mean"),
            Mean_Relative_Error=("Relative Error", "mean")
        )
    )

    return sparsity_summary


# In[41]:


# Matrix Size Summary

def create_size_summary(results):

    if results.empty:
        return pd.DataFrame()

    size_summary = (
        results
        .groupby(
            [
                "m",
                "n",
                "Solver"
            ],
            as_index=False
        )
        .agg(
            Mean_Time=("Time (s)", "mean"),
            Std_Time=("Time (s)", "std"),
            Mean_Iterations=("Iterations", "mean"),
            Mean_Residual=("Residual", "mean"),
            Mean_Relative_Error=("Relative Error", "mean")
        )
    )

    return size_summary


# In[43]:


# Lambda Summary

def create_lambda_summary(results):

    if results.empty:
        return pd.DataFrame()

    lambda_summary = (
        results
        .groupby(
            [
                "Lambda",
                "Solver"
            ],
            as_index=False
        )
        .agg(
            Mean_Time=("Time (s)", "mean"),
            Std_Time=("Time (s)", "std"),
            Mean_Iterations=("Iterations", "mean"),
            Mean_Residual=("Residual", "mean"),
            Mean_Relative_Error=("Relative Error", "mean")
        )
    )

    return lambda_summary


# In[45]:


# Spectrum Summary

def create_spectrum_summary(results):

    if results.empty:
        return pd.DataFrame()

    spectrum_summary = (
        results
        .groupby(
            [
                "Spectrum",
                "Solver"
            ],
            as_index=False
        )
        .agg(
            Mean_Time=("Time (s)", "mean"),
            Std_Time=("Time (s)", "std"),
            Mean_Iterations=("Iterations", "mean"),
            Mean_Residual=("Residual", "mean"),
            Mean_Relative_Error=("Relative Error", "mean")
        )
    )

    return spectrum_summary


# In[47]:


# Sparsity Pattern Summary

def create_pattern_summary(results):

    if results.empty:
        return pd.DataFrame()

    pattern_summary = (
        results
        .groupby(
            [
                "Sparsity Pattern",
                "Solver"
            ],
            as_index=False
        )
        .agg(
            Mean_Time=("Time (s)", "mean"),
            Std_Time=("Time (s)", "std"),
            Mean_Iterations=("Iterations", "mean"),
            Mean_Residual=("Residual", "mean"),
            Mean_Relative_Error=("Relative Error", "mean")
        )
    )

    return pattern_summary


# In[55]:


# Aggregate all completed SLURM results

def load_all_results(output_dir=OUTPUT_DIR):

    result_files = []

    for root, dirs, files in os.walk(output_dir):

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


def create_all_summaries(results):

    if results.empty:

        print("No results available for summary.")
        return {}

    summaries = {}

    summaries["summary"] = create_summary(results)

    summaries["condition_summary"] = (
        create_condition_summary(results)
    )

    summaries["sparsity_summary"] = (
        create_sparsity_summary(results)
    )

    summaries["size_summary"] = (
        create_size_summary(results)
    )

    summaries["lambda_summary"] = (
        create_lambda_summary(results)
    )

    summaries["spectrum_summary"] = (
        create_spectrum_summary(results)
    )

    summaries["pattern_summary"] = (
        create_pattern_summary(results)
    )

    for name, dataframe in summaries.items():

        filepath = os.path.join(
            OUTPUT_DIR,
            f"{name}.csv"
        )

        dataframe.to_csv(
            filepath,
            index=False
        )

        print()
        print("=" * 70)
        print(name)
        print("=" * 70)
        print(dataframe)

    return summaries


# In[ ]:


# Plotting

PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def save_plot(filename):

    filepath = os.path.join(
        PLOT_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_line(
    data,
    x,
    y,
    title,
    xlabel,
    ylabel,
    filename,
    log_y=False
):

    if data.empty:
        return

    plt.figure(figsize=(9, 6))

    for solver in sorted(data["Solver"].unique()):

        solver_data = (
            data[data["Solver"] == solver]
            .groupby(x, as_index=False)[y]
            .mean()
            .sort_values(x)
        )

        plt.plot(
            solver_data[x],
            solver_data[y],
            marker="o",
            label=solver
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()

    if log_y:
        plt.yscale("log")

    save_plot(filename)


def plot_bar(
    data,
    x,
    y,
    title,
    xlabel,
    ylabel,
    filename,
    log_y=False
):

    if data.empty:
        return

    plt.figure(figsize=(10, 6))

    pivot = (
        data
        .groupby([x, "Solver"])[y]
        .mean()
        .unstack()
    )

    pivot.plot(
        kind="bar",
        ax=plt.gca()
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend(
        title="Solver"
    )

    if log_y:
        plt.yscale("log")

    save_plot(filename)


def plot_box(
    data,
    x,
    y,
    title,
    xlabel,
    ylabel,
    filename
):

    if data.empty:
        return

    plt.figure(figsize=(11, 6))

    groups = []
    labels = []

    for value in sorted(data[x].unique()):

        subset = data[
            data[x] == value
        ]

        for solver in sorted(
            subset["Solver"].unique()
        ):

            values = subset[
                subset["Solver"] == solver
            ][y].dropna().values

            if len(values) > 0:

                groups.append(values)

                labels.append(
                    f"{value}\n{solver}"
                )

    if not groups:
        plt.close()
        return

    plt.boxplot(
        groups
    )

    plt.xticks(
        range(1, len(labels) + 1),
        labels,
        rotation=45
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    save_plot(filename)


def plot_iterations(
    data,
    x,
    title,
    xlabel,
    filename
):

    if data.empty:
        return

    plt.figure(figsize=(9, 6))

    for solver in sorted(
        data["Solver"].unique()
    ):

        solver_data = (
            data[data["Solver"] == solver]
            .groupby(
                x,
                as_index=False
            )["Iterations"]
            .mean()
            .sort_values(x)
        )

        plt.plot(
            solver_data[x],
            solver_data["Iterations"],
            marker="o",
            label=solver
        )

    plt.xlabel(xlabel)
    plt.ylabel("Mean iterations")
    plt.title(title)

    plt.legend()

    save_plot(filename)


def plot_relative_error(
    data,
    x,
    title,
    xlabel,
    filename
):

    if data.empty:
        return

    plt.figure(figsize=(9, 6))

    for solver in sorted(
        data["Solver"].unique()
    ):

        solver_data = (
            data[data["Solver"] == solver]
            .groupby(
                x,
                as_index=False
            )["Relative Error"]
            .mean()
            .sort_values(x)
        )

        plt.plot(
            solver_data[x],
            solver_data["Relative Error"],
            marker="o",
            label=solver
        )

    plt.xlabel(xlabel)
    plt.ylabel("Mean relative error")
    plt.title(title)

    plt.legend()

    save_plot(filename)


def plot_runtime_vs_error(
    results,
    experiment_name
):

    data = results[
        results["Experiment"] == experiment_name
    ]

    if data.empty:
        return

    plt.figure(figsize=(9, 6))

    for solver in sorted(
        data["Solver"].unique()
    ):

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
        f"Runtime vs Relative Error — {experiment_name}"
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.legend()

    save_plot(
        f"{experiment_name.lower().replace(' ', '_')}_runtime_vs_error.png"
    )


# In[25]:


# Experiment Definitions

EXPERIMENTS = {

    "Condition Parameter": {
        "m_values": [51000],
        "n_values": [5000],
        "condition_parameters": [10, 100, 1000, 10000, 20000],
        "sparsity_values": [0.95],
        "lambda_values": [1.0],
        "sparsity_patterns": ["random"],
        "spectra": ["exponential"]
    },

    "Sparsity": {
        "m_values": [20000],
        "n_values": [5000],
        "condition_parameters": [1000],
        "sparsity_values": [0.9, 0.95, 0.98, 0.99, 0.999],
        "lambda_values": [1.0],
        "sparsity_patterns": ["random"],
        "spectra": ["exponential"]
    },

    "Matrix Size": {
        "m_values": [20000, 30000, 25000],
        "n_values": [5000],
        "condition_parameters": [1000],
        "sparsity_values": [0.95],
        "lambda_values": [1.0],
        "sparsity_patterns": ["random"],
        "spectra": ["exponential"]
    },

    "Lambda": {
        "m_values": [20000],
        "n_values": [5000],
        "condition_parameters": [1000],
        "sparsity_values": [0.95],
        "lambda_values": [0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1],
        "sparsity_patterns": ["random"],
        "spectra": ["exponential"]
    },

    "Sparsity Pattern": {
        "m_values": [20000],
        "n_values": [5000],
        "condition_parameters": [1000],
        "sparsity_values": [0.95],
        "lambda_values": [1.0],
        "sparsity_patterns": [
            "random",
            "banded",
            "block"
        ],
        "spectra": ["exponential"]
    },

    "Spectrum": {
        "m_values": [20000],
        "n_values": [5000],
        "condition_parameters": [1000],
        "sparsity_values": [0.95],
        "lambda_values": [1.0],
        "sparsity_patterns": ["random"],
        "spectra": [
            "linear",
            "exponential",
            "polynomial",
            "clustered"
        ]
    }
}

print("Experiments defined:")

for name in EXPERIMENTS:
    print(" -", name)
# In[27]:


# Solver-specific benchmark runner



def run_solver_experiment(
    solver_name,
    experiment_name,
    m_values,
    n_values,
    condition_parameters,
    sparsity_values,
    lambda_values,
    sparsity_patterns,
    spectra,
    repeats=10,
    noise_level=0.01,
    random_seed=42
):

    if solver_name not in SOLVER_FUNCTIONS:
        raise ValueError(
            f"Unknown solver: {solver_name}. "
            f"Choose from {list(SOLVER_FUNCTIONS.keys())}"
        )

    solver_function = SOLVER_FUNCTIONS[solver_name]

    experiment_dir = os.path.join(
        OUTPUT_DIR,
        experiment_name,
        solver_name
    )

    os.makedirs(
        experiment_dir,
        exist_ok=True
    )

    checkpoint_file = os.path.join(
        experiment_dir,
        "results.csv"
    )

    rows = []

    experiment_number = 1

    for m in m_values:

        for n in n_values:

            for condition_parameter in condition_parameters:

                for sparsity in sparsity_values:

                    for lambda_ in lambda_values:

                        for sparsity_pattern in sparsity_patterns:

                            for spectrum in spectra:

                                print(
                                    f"{solver_name} | "
                                    f"{experiment_name} | "
                                    f"Experiment {experiment_number}"
                                )

                                for run in range(repeats):

                                    try:

                                        A, b, x_true = generate_problem(
                                            m=m,
                                            n=n,
                                            condition_parameter=condition_parameter,
                                            sparsity=sparsity,
                                            sparsity_pattern=sparsity_pattern,
                                            spectrum=spectrum,
                                            noise_level=noise_level,
                                            random_seed=random_seed + run
                                        )

                                        result = solver_function(
                                            A,
                                            b,
                                            x_true,
                                            lambda_
                                        )

                                        result["Experiment"] = (
                                            experiment_name
                                        )
                                        result["Experiment Number"] = (
                                            experiment_number
                                        )
                                        result["Run"] = run + 1
                                        result["m"] = m
                                        result["n"] = n

                                        result["Condition Parameter"] = (
                                            condition_parameter
                                        )
                                        result["Sparsity"] = sparsity
                                        result["Sparsity Pattern"] = (
                                            sparsity_pattern
                                        )
                                        result["Spectrum"] = spectrum
                                        result["Lambda"] = lambda_

                                        rows.append(result)

                                        print(
                                            f"  Run {run + 1}/{repeats} "
                                            f"completed"
                                        )

                                    except Exception as error:

                                        print(
                                            f"  Run {run + 1}/{repeats} "
                                            f"FAILED"
                                        )
                                        print(
                                            f"  Error: {error}"
                                        )

                                    current_results = pd.DataFrame(rows)

                                    current_results.to_csv(
                                        checkpoint_file,
                                        index=False
                                    )

                                experiment_number += 1

    results = pd.DataFrame(rows)

    final_file = os.path.join(
        experiment_dir,
        "results_final.csv"
    )

    results.to_csv(
        final_file,
        index=False
    )

    return results


# In[29]:


# Single Experiment Job

def run_single_job(
    experiment_name,
    solver_name,
    repeats=10,
    noise_level=0.01,
    random_seed=42
):

    if experiment_name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment: {experiment_name}. "
            f"Available experiments: {list(EXPERIMENTS.keys())}"
        )

    if solver_name not in SOLVER_FUNCTIONS:
        raise ValueError(
            f"Unknown solver: {solver_name}. "
            f"Available solvers: {list(SOLVER_FUNCTIONS.keys())}"
        )

    params = EXPERIMENTS[experiment_name]

    results = run_solver_experiment(
        solver_name=solver_name,
        experiment_name=experiment_name,
        m_values=params["m_values"],
        n_values=params["n_values"],
        condition_parameters=params["condition_parameters"],
        sparsity_values=params["sparsity_values"],
        lambda_values=params["lambda_values"],
        sparsity_patterns=params["sparsity_patterns"],
        spectra=params["spectra"],
        repeats=repeats,
        noise_level=noise_level,
        random_seed=random_seed
    )

    return results
# In[ ]:


# SLURM Job Entry Point

def main():

    parser = argparse.ArgumentParser(
        description="Run one solver for one experiment."
    )

    parser.add_argument(
        "--experiment",
        required=True,
        choices=list(EXPERIMENTS.keys())
    )

    parser.add_argument(
        "--solver",
        required=True,
        choices=list(SOLVER_FUNCTIONS.keys())
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=10
    )

    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Starting SLURM experiment")
    print("=" * 70)
    print(f"Experiment : {args.experiment}")
    print(f"Solver     : {args.solver}")
    print(f"Repeats    : {args.repeats}")
    print(f"Noise      : {args.noise_level}")
    print(f"Seed       : {args.random_seed}")
    print(f"Output     : {OUTPUT_DIR}")
    print("=" * 70)

    try:

        results = run_single_job(
            experiment_name=args.experiment,
            solver_name=args.solver,
            repeats=args.repeats,
            noise_level=args.noise_level,
            random_seed=args.random_seed
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("JOB FAILED")
        print("=" * 70)
        print(f"Error: {error}")
        raise

    print()
    print("=" * 70)
    print("Job completed")
    print("=" * 70)
    print(f"Rows produced: {len(results)}")


if __name__ == "__main__":
    main()


# In[ ]:




