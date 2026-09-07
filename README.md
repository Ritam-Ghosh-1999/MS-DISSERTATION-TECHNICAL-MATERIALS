# MSc Dissertation - Technical Materials

## 1. Purpose of this repository

This repository contains the Python implementations and reproducibility documentation for the computational experiments used in my MSc dissertation.

The aim is to provide enough information for another researcher to understand what was tested, how the numerical experiments were carried out, what computing resources were required, and how the reported plots and summary data can be regenerated.

The repository is organised by experiment. Each experiment has its own folder containing:

1. an introduction to the problem and the purpose of the experiment;
2. step-by-step instructions for running the experiment;
3. a requirements and restrictions document describing the conditions that should be preserved;
4. the Python file used for the numerical experiment; and
5. the Python file used to generate the plots from the numerical results.

Generated CSV files and PNG figures are not intended to form part of the permanent repository. They are outputs of the supplied programs and can be recreated by following the instructions in the relevant folder.

---

## 2. How to use this repository

The recommended way to use the repository is to work through the experiment folders in the order shown below.

For **each folder**, the intended reading order is:

**1. Introduction PDF -> 2. Run Instructions PDF -> 3. Requirements and Restrictions PDF -> 4. Numerical Python file -> 5. Plotting Python file**

The PDFs should be read before running the code. The introduction explains what the experiment is testing, the run instructions explain how to execute it, and the requirements document identifies the settings that should not be changed if the intention is to reproduce the dissertation experiment.

The Python files are supplied after the documentation so that the implementation can be inspected with the experimental setup already understood.

---

# 3. Recommended experiment order

The folders should be presented in the following order:

### 01 - Overdetermined Systems

This is the first large synthetic-system experiment. It studies the behaviour of the solvers while varying properties such as matrix size, conditioning, sparsity, sparsity pattern, singular-value spectrum and regularisation.

Read:

1. `Overdetermined_Systems_Introduction.pdf`
2. `Overdetermined_Systems_Run_Instructions.pdf`
3. `Overdetermined_Systems_Requirements_and_Restrictions.pdf`
4. `Overdetermined.py`
5. `Overdetermined_plots.py`

### 02 - Small Matrices

This is the smaller-scale synthetic benchmark. It provides a more manageable setting in which the effects of matrix properties and regularisation can be examined before considering the larger computational experiments.

Read:

1. `Small_Matrices_Introduction.pdf`
2. `Small_Matrices_Run_Instructions.pdf`
3. `Small_Matrices_Requirements_and_Restrictions.pdf`
4. `Small_Matrices.py`
5. `Plot_Small_Matrices.py`

### 03 - Underdetermined Systems

This experiment considers systems with more unknowns than equations. The same general family of matrix characteristics is investigated, but under the underdetermined relationship between the number of equations and unknowns.

Read:

1. `Underdetermined_Introduction.pdf`
2. `Underdetermined_Run_Instructions.pdf`
3. `Underdetermined_Requirements_and_Restrictions.pdf`
4. `Underdetermined.py`
5. `Underdetermined_Plots.py`

### 04 - Maragal-8

This is a real sparse matrix experiment based on the Maragal_8 problem from the SuiteSparse Matrix Collection. It is used to examine solver behaviour on a large, rectangular, sparse and numerically rank-deficient problem.

Read:

1. `Maragal_8_Introduction.pdf`
2. `Maragal_8_Run_Instructions.pdf`
3. `Maragal_8_Requirements_and_Restrictions.pdf`
4. `Maragal_8.py`
5. `plot_Maragal_8.py`

### 05 - JP Seismic Tomography

This is a real-data experiment based on the Harvard Seismology JP matrix from the SuiteSparse Matrix Collection. The matrix represents a large sparse forward operator associated with seismic tomography.

Read:

1. `JP_Seismic_Introduction.pdf`
2. `JP_Seismic_Run_Instructions.pdf`
3. `JP_Seismic_Requirements_and_Restrictions.pdf`
4. `JP_Seismic_Tomography.py`
5. `plot_JP.py`

### 06 - Few-View Computed Tomography

This is the large-scale matrix-free CT experiment. It uses a synthetic few-view parallel-beam CT problem and compares the iterative methods on the resulting severely underdetermined reconstruction problem.

Read:

1. `CT_Introduction.pdf`
2. `CT_Run_Instructions.pdf`
3. `CT_Requirements_and_Restrictions.pdf`
4. `CT.py`
5. `plot_CT.py`

---

# 4. What each experiment folder contains

Every experiment folder follows the same basic documentation structure.

## Introduction

The introduction is intended to be read before any code is run. It explains the problem in plain language, gives the mathematical context, describes the data or synthetic problem being used, and explains what the experiment is intended to demonstrate.

It is written so that a reader does not need to be an expert programmer to understand the purpose of the experiment.

## Run Instructions

The run-instructions document gives the practical procedure for reproducing the experiment.

Depending on the experiment, this includes:

- preparing the Python files;
- checking the Python environment and required packages;
- running a local Python calculation where appropriate;
- logging into the University of Manchester CSF where HPC execution is required;
- submitting the numerical calculation through Slurm;
- checking the progress and completion of the calculation;
- locating the generated CSV files;
- running the plotting program;
- locating the generated PNG figures and summary files; and
- copying generated outputs back to a local machine when required.

The exact CSF login procedure and current resource limits are University-dependent and should be checked against the current University documentation.

## Requirements and Restrictions

This document records the numerical and computational conditions that define the experiment.

These include, where applicable:

- matrix dimensions;
- sparsity;
- condition parameters;
- sparsity patterns;
- singular-value spectra;
- regularisation parameters;
- noise levels;
- solver settings;
- tolerances and iteration limits;
- number of repeats;
- matrix-free restrictions;
- required datasets; and
- computational resource considerations.

A change to a numerical setting can produce a valid new experiment, but it should not be described as an exact reproduction of the dissertation experiment.

---

# 5. Numerical and plotting programs

The numerical Python program is responsible for carrying out the experiment and recording the numerical measurements.

The plotting Python program is separate. It reads the CSV results produced by the numerical program and recreates the plots and summary information used to inspect the experiment.

This separation is intentional. It means that the numerical calculation does not need to be repeated simply to change the presentation of an already completed set of results.

The exact command-line arguments, output directories and generated filenames are documented in the run-instructions PDF in each experiment folder and should be checked there rather than assumed to be identical across all experiments.

---

# 6. Computational environments

The experiments do not all require the same type of computer.

The small-matrix experiment is designed to be considerably more manageable and can be run in a normal Python/Anaconda environment where sufficient resources are available.

The larger experiments were designed for high-performance computing. In particular, the overdetermined and underdetermined synthetic benchmarks, Maragal-8, JP seismic tomography and the large CT experiment can require substantial memory and computation time.

For CSF-based experiments, the person reproducing the work must use the current University procedures for:

- obtaining CSF access;
- selecting an appropriate computing environment;
- determining available memory;
- determining the permitted CPU allocation;
- selecting the appropriate Slurm configuration; and
- submitting and monitoring jobs.

The dissertation experiments used substantial allocations. For the large direct-solver cases, less than 128 GB of memory was not sufficient in the tested setup. Resource limits and available allocations can change, so current University guidance should always be checked before submitting a job.

---

# 7. Matrix-free computations

Several experiments use matrix-free formulations.

Where the documentation states that a computation is matrix-free, this is part of the experimental design. The purpose is to apply the required linear operator without explicitly constructing the corresponding large dense normal-equation matrix.

For example, CG and PCG may apply the normal-equation operation through matrix-vector products involving `A` and its transpose rather than explicitly forming `A^T A`.

Replacing a matrix-free implementation with an explicitly formed matrix may therefore change both the computational problem and the memory requirements.

The requirements document for each experiment should be consulted before making such a change.

---

# 8. Generated outputs

The Python programs create numerical outputs during a run.

These may include CSV files containing:

- individual solver measurements;
- summary statistics;
- runtime comparisons; and
- other experiment-specific measurements.

The plotting programs then use the generated CSV files to produce PNG figures and, where implemented, additional summary information.

These generated files are not required to remain in the GitHub repository. They can be removed after inspection and regenerated from the supplied code.

Input datasets are different from generated outputs. Where an experiment requires an external dataset, such as a SuiteSparse Matrix Collection problem, the relevant requirements and run-instructions documents explain what is required.

---

# 9. Reproducibility principles

For a reproduction to be comparable with the dissertation results, the important experimental settings should be preserved.

In particular, the reader should not silently change:

- matrix dimensions;
- matrix or dataset identity;
- sparsity;
- conditioning;
- sparsity pattern;
- spectrum;
- regularisation values;
- noise level;
- solver configuration;
- stopping tolerance;
- maximum iterations;
- number of repeats; or
- matrix-free versus explicitly formed implementations.

Computing-resource settings are different. CPU counts, memory limits, queue or partition names, and other CSF/Slurm details depend on the resources available at the time of reproduction. These should be adapted to current University rules and recorded when reporting the reproduced results.

A reproduced experiment should therefore distinguish between:

**changes required by the current computing environment**, and

**changes to the numerical definition of the experiment**.

The latter can affect the scientific result and should be reported explicitly.

---

# 10. Repository structure

The final repository is intended to have the following structure:

```text
MS-DISSERTATION-TECHNICAL-MATERIALS/
|
|-- README.md
|
|-- 01_Overdetermined_Systems/
|   |-- 01_Overdetermined_Systems_Introduction.pdf
|   |-- 02_Overdetermined_Systems_Run_Instructions.pdf
|   |-- 03_Overdetermined_Systems_Requirements_and_Restrictions.pdf
|   |-- Overdetermined.py
|   `-- Overdetermined_plots.py
|
|-- 02_Small_Matrices/
|   |-- 01_Small_Matrices_Introduction.pdf
|   |-- 02_Small_Matrices_Run_Instructions.pdf
|   |-- 03_Small_Matrices_Requirements_and_Restrictions.pdf
|   |-- Small_Matrices.py
|   `-- Plot_Small_Matrices.py
|
|-- 03_Underdetermined_Systems/
|   |-- 01_Underdetermined_Introduction.pdf
|   |-- 02_Underdetermined_Run_Instructions.pdf
|   |-- 03_Underdetermined_Requirements_and_Restrictions.pdf
|   |-- Underdetermined.py
|   `-- Underdetermined_Plots.py
|
|-- 04_Maragal_8/
|   |-- 01_Maragal_8_Introduction.pdf
|   |-- 02_Maragal_8_Run_Instructions.pdf
|   |-- 03_Maragal_8_Requirements_and_Restrictions.pdf
|   |-- Maragal_8.py
|   `-- plot_Maragal_8.py
|
|-- 05_JP_Seismic/
|   |-- 01_JP_Seismic_Introduction.pdf
|   |-- 02_JP_Seismic_Run_Instructions.pdf
|   |-- 03_JP_Seismic_Requirements_and_Restrictions.pdf
|   |-- JP_Seismic_Tomography.py
|   `-- plot_JP.py
|
`-- 06_CT/
    |-- 01_CT_Introduction.pdf
    |-- 02_CT_Run_Instructions.pdf
    |-- 03_CT_Requirements_and_Restrictions.pdf
    |-- CT.py
    `-- plot_CT.py
```

The numbering is deliberate. GitHub displays directory contents alphabetically, so the numerical prefixes make the intended reading order visible without requiring the reader to infer it.

---

# 11. Relationship to the dissertation

The folders correspond to the computational experiments discussed in the dissertation. The dissertation provides the scientific discussion and interpretation of the results, while this repository provides the technical material needed to inspect and reproduce the computational work.

The repository should therefore be read together with the relevant dissertation sections rather than treated as a replacement for the dissertation.

For a first-time reader, the recommended route is:

**README -> experiment Introduction -> Run Instructions -> Requirements and Restrictions -> numerical code -> plotting code**

This order is intended to make the repository usable by readers with different levels of programming and computational experience.
