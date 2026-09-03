# MSc Dissertation — Technical Materials

## Regularised Linear Systems and Iterative Solver Performance

Technical repository accompanying the MSc dissertation.

This repository contains the source code, experimental results, summary data,
and visualisations used in the dissertation study of regularised linear
least-squares problems and iterative solver performance.


## Overview

The computational study investigates how numerical solver behaviour changes
with problem characteristics and regularisation.

The experiments examine the effects of matrix size, conditioning, sparsity,
sparsity pattern, singular-value spectrum, and the regularisation parameter
lambda.

Performance is assessed using runtime, iteration count, residual measures,
and relative error.


## Mathematical Problem

The computational experiments consider regularised least-squares problems of
the form

$$
\min_x \|Ax-b\|_2^2 + \lambda\|x\|_2^2.
$$

The implementations compare direct and iterative approaches for solving the
resulting regularised systems.


## Numerical Solvers

The repository contains implementations and experiments involving:

| Solver | Role |
|---|---|
| Direct | Direct solution approach |
| CG | Conjugate Gradient applied to the regularised normal-equation operator |
| LSQR | Iterative least-squares solver |
| LSMR | Iterative least-squares solver |
| PCG | Preconditioned Conjugate Gradient |

The CG and PCG implementations use matrix-free application of the
normal-equation operator rather than explicitly forming the matrix
$A^T A$.


## Experimental Programme

The synthetic experiments investigate the influence of:

- **Matrix size**
- **Condition parameter**
- **Sparsity**
- **Sparsity pattern**
- **Singular-value spectrum**
- **Regularisation parameter lambda**

The experiments generate comparative results for the different numerical
solvers.

Key performance measures include:

- Runtime
- Iteration count
- Relative error
- Relative residual
- Normal-equation residual
- Objective value


## Repository Structure

### `output_overdetermined_updated/`

Source code, experimental results, summary data, and plots for the
overdetermined problem.

### `output_underdetermined_updated/`

Source code, experimental results, summary data, and plots for the
underdetermined problem.

### `output_small_matrices/`

Experimental results, summary tables, and plots for the small-matrix
experiments.

### `Maragal_8/`

Real-data experiments based on the Maragal-8 problem, including solver
comparisons, lambda sweeps, convergence information, and generated
visualisations.

### `JP_Seismic_Results/`

Real-data experiments using the Japan seismic tomography problem. The
implementation compares CG, PCG, LSQR, and LSMR and includes lambda-sweep
results, solver-runtime comparisons, and generated visualisations.


## Source Code

The principal Python implementations are contained within the corresponding
experiment folders.

The repository includes both computational scripts and plotting scripts used
to generate and analyse the reported results.


## Results and Visualisations

The repository preserves the generated computational outputs alongside the
source code.

### CSV files

CSV files contain detailed experimental results, solver measurements, and
summary statistics.

### PNG files

PNG files contain the plots generated from the experimental results,
including comparisons of runtime, iterations, residuals, relative error,
and other performance measures.


## Reproducibility

The repository is structured so that the principal computational scripts,
experimental results, summary data, and generated visualisations are retained
together.

This provides a record of the computational workflow supporting the
dissertation and allows the reported numerical experiments to be inspected
alongside their corresponding source code and outputs.


## Technical Submission

This repository forms the technical-materials component accompanying the
MSc dissertation.

**GitHub repository:**

https://github.com/Ritam-Ghosh-1999/MS-DISSERTATION-TECHNICAL-MATERIALS
