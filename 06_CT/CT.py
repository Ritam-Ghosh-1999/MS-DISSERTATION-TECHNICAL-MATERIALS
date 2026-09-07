#!/usr/bin/env python3
"""
Large-scale few-view CT benchmark: CG, PCG, LSQR, LSMR.

Matrix-free parallel-beam CT operator. No explicit CT system matrix is formed.

The experiment uses a few-view CT regime because published CT work reports
LSQR as better suited than LSMR for that application:
Chillaron Perez et al., Nuclear Science and Engineering 198(2), 193-206,
DOI: 10.1080/00295639.2023.2199677.

This script does NOT hard-code an LSQR win; it measures it.
"""

from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.sparse.linalg import LinearOperator, cg, lsqr, lsmr

SOLVERS = ["CG", "PCG", "LSQR", "LSMR"]


def shepp_logan(n):
    y, x = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    img = np.zeros((n, n), dtype=np.float64)
    ellipses = [
        (1.00, 0, 0, .69, .92, 0),
        (-.80, 0, -.018, .6624, .874, 0),
        (-.20, .22, 0, .11, .31, -18),
        (-.20, -.22, 0, .16, .41, 18),
        (.10, 0, .35, .21, .25, 0),
        (.10, 0, -.35, .21, .25, 0),
        (.10, -.08, .62, .046, .046, 0),
        (.10, .08, .62, .046, .046, 0),
        (.10, -.06, -.10, .046, .023, 0),
        (.10, .06, -.10, .046, .023, 0),
        (.10, 0, -.10, .023, .023, 0),
    ]
    for amp, x0, y0, a, b, deg in ellipses:
        t = np.deg2rad(deg)
        xr = (x-x0)*np.cos(t) + (y-y0)*np.sin(t)
        yr = -(x-x0)*np.sin(t) + (y-y0)*np.cos(t)
        mask = (xr/a)**2 + (yr/b)**2 <= 1
        img[mask] += amp
    img -= img.min()
    img /= max(img.max(), 1e-15)
    return img


class ParallelBeamCT:
    """Matrix-free parallel-beam CT using rotation + detector integration."""

    def __init__(self, n, angles):
        self.n = int(n)
        self.angles = np.asarray(angles, dtype=float)
        self.m = len(self.angles) * self.n
        self.shape = (self.m, self.n*self.n)

    def forward(self, x):
        image = np.asarray(x).reshape(self.n, self.n)
        sino = np.empty((len(self.angles), self.n), dtype=np.float64)
        for i, angle in enumerate(self.angles):
            rot = ndimage.rotate(
                image, float(angle), reshape=False, order=1,
                mode="constant", cval=0.0, prefilter=False
            )
            sino[i] = rot.sum(axis=0)
        return sino.ravel()

    def adjoint(self, y):
        sino = np.asarray(y).reshape(len(self.angles), self.n)
        image = np.zeros((self.n, self.n), dtype=np.float64)
        for i, angle in enumerate(self.angles):
            proj = np.broadcast_to(sino[i][None, :], (self.n, self.n))
            back = ndimage.rotate(
                proj, float(-angle), reshape=False, order=1,
                mode="constant", cval=0.0, prefilter=False
            )
            image += back
        return image.ravel()

    def operator(self):
        return LinearOperator(
            self.shape, matvec=self.forward, rmatvec=self.adjoint,
            dtype=np.float64
        )


def augmented_operator(A, lam):
    n = A.shape[1]
    s = math.sqrt(lam)

    def mv(x):
        return np.concatenate((A.matvec(x), s*x))

    def rmv(y):
        return A.rmatvec(y[:A.shape[0]]) + s*y[A.shape[0]:]

    return LinearOperator(
        (A.shape[0]+n, n), matvec=mv, rmatvec=rmv, dtype=np.float64
    )


def normal_operator(A, lam):
    n = A.shape[1]

    def mv(x):
        return A.rmatvec(A.matvec(x)) + lam*x

    return LinearOperator((n, n), matvec=mv, rmatvec=mv, dtype=np.float64)


def jacobi_preconditioner(A, lam, n, samples=8, seed=12345):
    """Matrix-free Hutchinson estimate of diag(A^T A + lambda I)."""
    rng = np.random.default_rng(seed)
    diag = np.zeros(n)

    for _ in range(samples):
        z = rng.choice(np.array([-1.0, 1.0]), size=n)
        Az = A.matvec(z)
        normal_z = A.rmatvec(Az) + lam*z
        diag += z*normal_z

    diag /= samples
    diag = np.maximum(diag, 1e-10)

    return LinearOperator(
        (n, n),
        matvec=lambda v: v/diag,
        rmatvec=lambda v: v/diag,
        dtype=np.float64
    )


def metrics(A, x, b, truth, lam):
    r = A.matvec(x) - b
    at_r = A.rmatvec(r)
    b_at = A.rmatvec(b)
    return {
        "Relative_Residual": np.linalg.norm(r)/max(np.linalg.norm(b), 1e-15),
        "Relative_Normal_Residual": (
            np.linalg.norm(at_r)/max(np.linalg.norm(b_at), 1e-15)
        ),
        "Objective": float(np.dot(r, r) + lam*np.dot(x, x)),
        "Relative_Reconstruction_Error": (
            np.linalg.norm(x-truth.ravel()) /
            max(np.linalg.norm(truth.ravel()), 1e-15)
        ),
    }


def run_solver(solver, A, b, truth, lam, tol, maxiter, M=None):
    n = A.shape[1]
    start = time.perf_counter()
    iterations = 0

    def callback(_):
        nonlocal iterations
        iterations += 1

    if solver in ("LSQR", "LSMR"):
        B = augmented_operator(A, lam)
        rhs = np.concatenate((b, np.zeros(n)))
        if solver == "LSQR":
            out = lsqr(
                B, rhs, atol=tol, btol=tol, conlim=1e12,
                iter_lim=maxiter, show=False
            )
            x, istop, iterations = out[0], out[1], int(out[2])
        else:
            out = lsmr(
                B, rhs, atol=tol, btol=tol, conlim=1e12,
                maxiter=maxiter, show=False
            )
            x, istop, iterations = out[0], out[1], int(out[2])
        info = 0 if istop in (1, 2) else int(istop)

    else:
        N = normal_operator(A, lam)
        rhs = A.rmatvec(b)
        x, info = cg(
            N, rhs, rtol=tol, atol=0.0, maxiter=maxiter,
            M=M if solver == "PCG" else None, callback=callback
        )

    elapsed = time.perf_counter() - start
    out = metrics(A, x, b, truth, lam)
    out.update({
        "Solver": solver,
        "Runtime_s": elapsed,
        "Iterations": iterations,
        "Converged": bool(info == 0),
        "Info": int(info),
    })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024,
                   help="Image side. 1024=1048576 unknowns; 2048=4194304." )
    p.add_argument("--views", type=int, default=64,
                   help="Few-view angular projections.")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--noise", type=float, default=0.002)
    p.add_argument("--lambdas", default="1e-6,1e-5,1e-4,1e-3,1e-2")
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--output", default="CT_real_world_output")
    p.add_argument("--seed", type=int, default=20260904)
    args = p.parse_args()

    if args.n < 64 or args.views < 8 or args.repeats < 1:
        raise ValueError("Use n>=64, views>=8 and repeats>=1.")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lambdas = np.array([float(v) for v in args.lambdas.split(",")])

    angles = np.linspace(0, 180, args.views, endpoint=False)
    truth = shepp_logan(args.n)
    ct = ParallelBeamCT(args.n, angles)
    A = ct.operator()

    print("="*72)
    print("LARGE-SCALE FEW-VIEW CT BENCHMARK")
    print("="*72)
    print(f"Image                  : {args.n} x {args.n}")
    print(f"Unknowns               : {args.n*args.n:,}")
    print(f"Views                  : {args.views}")
    print(f"Measurements           : {args.views*args.n:,}")
    print(f"m/n                    : {(args.views*args.n)/(args.n*args.n):.4f}")
    print(f"Noise                  : {args.noise}")
    print(f"Lambdas                : {lambdas}")
    print(f"Tolerance              : {args.tol}")
    print(f"Max iterations         : {args.maxiter}")
    print("Explicit matrix A      : NO")
    print("="*72)

    clean_b = A.matvec(truth.ravel())
    norm_b = np.linalg.norm(clean_b)
    rng = np.random.default_rng(args.seed)
    rows = []

    for lam in lambdas:
        print(f"\nLambda = {lam:g}")
        print("Building PCG preconditioner...")
        M = jacobi_preconditioner(A, lam, A.shape[1])

        for rep in range(1, args.repeats+1):
            noise = rng.standard_normal(clean_b.size)
            noise *= args.noise*norm_b/max(np.linalg.norm(noise), 1e-15)
            b = clean_b + noise
            print(f"  Repeat {rep}/{args.repeats}")

            for solver in SOLVERS:
                print(f"    {solver:4s} ...", end="", flush=True)
                result = run_solver(
                    solver, A, b, truth, lam, args.tol,
                    args.maxiter, M
                )
                result.update({
                    "Lambda": lam, "Repeat": rep,
                    "N": args.n*args.n, "M": args.views*args.n,
                    "Views": args.views, "Noise_Level": args.noise,
                    "Tolerance": args.tol, "Max_Iterations": args.maxiter
                })
                rows.append(result)
                print(
                    f" {result['Runtime_s']:.3f}s, "
                    f"{result['Iterations']} iters, "
                    f"res={result['Relative_Residual']:.3e}, "
                    f"err={result['Relative_Reconstruction_Error']:.3e}"
                )

    results = pd.DataFrame(rows)
    results.to_csv(out/"CT_solver_results.csv", index=False)

    summary = (
        results.groupby(["Lambda","Solver"], as_index=False)
        .agg(
            Mean_Time=("Runtime_s","mean"),
            Std_Time=("Runtime_s","std"),
            Mean_Iterations=("Iterations","mean"),
            Std_Iterations=("Iterations","std"),
            Convergence_Rate=("Converged","mean"),
            Mean_Relative_Residual=("Relative_Residual","mean"),
            Mean_Relative_Normal_Residual=(
                "Relative_Normal_Residual","mean"
            ),
            Mean_Objective=("Objective","mean"),
            Mean_Relative_Reconstruction_Error=(
                "Relative_Reconstruction_Error","mean"
            )
        )
    )
    summary.to_csv(out/"CT_solver_summary.csv", index=False)

    rt = summary.pivot(
        index="Lambda", columns="Solver", values="Mean_Time"
    ).reset_index()

    it = summary.pivot(
        index="Lambda", columns="Solver", values="Mean_Iterations"
    ).reset_index()

    for s in SOLVERS:
        if s not in rt:
            rt[s] = np.nan
        if s not in it:
            it[s] = np.nan

    rt["LSQR_vs_CG_Time_Ratio"] = rt["CG"]/rt["LSQR"]
    rt["LSQR_vs_PCG_Time_Ratio"] = rt["PCG"]/rt["LSQR"]
    rt["LSQR_vs_LSMR_Time_Ratio"] = rt["LSMR"]/rt["LSQR"]
    rt["LSQR_vs_CG_Iteration_Ratio"] = it["CG"]/it["LSQR"]
    rt["LSQR_vs_PCG_Iteration_Ratio"] = it["PCG"]/it["LSQR"]
    rt["LSQR_vs_LSMR_Iteration_Ratio"] = it["LSMR"]/it["LSQR"]

    rt.to_csv(out/"CT_solver_runtime_comparison.csv", index=False)

    meta = f"""Large-scale few-view CT benchmark

Problem: matrix-free parallel-beam X-ray CT reconstruction
Image: {args.n} x {args.n}
Unknowns: {args.n*args.n:,}
Views: {args.views}
Measurements: {args.views*args.n:,}
m/n: {(args.views*args.n)/(args.n*args.n):.6f}

Inverse problem:
    min_x ||Ax-b||_2^2 + lambda ||x||_2^2

Solvers:
    CG   = CG on regularised normal equations
    PCG  = diagonally preconditioned CG
    LSQR = Golub-Kahan least-squares
    LSMR = Golub-Kahan minimum-residual least-squares

A is matrix-free. No explicit CT system matrix is stored.

LSMR/LSQR comparison is motivated by:
Chillaron Perez, M., Vidal, V.E., Verdu, G.J., Quintana-Orti, G.
Few-View CT Image Reconstruction via Least-Squares Methods:
Assessment and Optimization.
Nuclear Science and Engineering, 198(2), 193-206 (2024).
DOI: 10.1080/00295639.2023.2199677

Important: the script does not force LSQR to win. The measured results
should determine the conclusion.

Speedup convention:
    competitor_time / LSQR_time
so >1 means LSQR is faster.
"""
    (out/"CT_experiment_metadata.txt").write_text(meta)

    print("\n"+"="*72)
    print("COMPLETE")
    print("="*72)
    for f in sorted(out.iterdir()):
        print(f.name)


if __name__ == "__main__":
    main()
