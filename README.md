# Numerical Experiments for Double Saddle-Point Systems

Python code accompanying **Block-Diagonal Schur-Complement Preconditioners for Non-Hermitian Double Saddle-Point Systems**.

The scripts reproduce the eigenvalue plots, GMRES convergence histories, and scaling experiments in Section 7. Iteration counts and spectral diagnostics are reproducible across machines; timing results depend on the computer and software environment.

## Files

| File | Purpose |
| --- | --- |
| `dsp.py` | Matrix construction, block preconditioners, right-preconditioned GMRES, and the timer |
| `run_experiments.py` | Runs the numerical experiments and saves figures and results |
| `results.txt` | Reference output from the authors' run reported in the manuscript |

Keep all three files in the same directory. Use the names `dsp.py`, `run_experiments.py`, and `results.txt` when uploading to the repository; remove download suffixes such as `(1)` or `(2)`. The driver imports `dsp` by this module name.

## Requirements and installation

Required packages: **NumPy**, **SciPy**, and **Matplotlib**. The optional package **threadpoolctl** is used only to record the BLAS thread count in `results.txt`; the script runs without it.

The reference results were obtained with Python 3.8.5, NumPy 1.24.4 and SciPy 1.10.1. The script writes the Python, NumPy and SciPy versions, the platform and the BLAS thread count at the top of `results.txt`, so every run documents its own environment.

Install the dependencies in your selected Python environment:

```sh
python -m pip install numpy scipy matplotlib threadpoolctl
```

To use the SciPy version reported in the manuscript, install it in a compatible Python environment:

```sh
python -m pip install "scipy==1.10.1"
```

In Visual Studio Code, use **Python: Select Interpreter** to select the same environment in which these packages are installed. Warnings such as `Import "numpy" could not be resolved` from Pylance mean that the editor uses a different interpreter; they do not affect running the script.

## Running the experiments

Open a terminal in the directory containing `run_experiments.py` and `dsp.py`, then run:

```sh
python -u run_experiments.py
```

For example, in Windows Command Prompt:

```bat
cd /d "C:\path\to\repository"
python -u run_experiments.py
```

The script runs all experiments without command-line arguments. The eigenvalue calculations may take some time before the first output appears, and the full run can take several minutes, depending on the machine. The final message is `done`. Figures are saved directly using Matplotlib's `Agg` backend; no plot windows need to be closed.

To run the scaling study for only some grid sizes (for example, for a quick test), set the environment variable `KS`:

Linux / macOS:

```sh
KS=16,32 python -u run_experiments.py
```

Windows Command Prompt:

```bat
set KS=16,32
python -u run_experiments.py
```

In Command Prompt, `set KS=` (with nothing after the equals sign) removes the variable again before the full run.

The manuscript results use the default `KS=16,32,64,128`. Because the right-hand sides are drawn from one random stream, running a subset of sizes changes the right-hand sides of the later runs, and iteration counts in the stabilized case can then differ by one or two from the reference.

## Generated output

Files are written to the **current working directory**:

| File | Contents |
| --- | --- |
| `fig_eigs_unstab.pdf` | Eigenvalue plots for the unstabilized system (Figure 1a) |
| `fig_eigs_stab.pdf` | Eigenvalue plots for the stabilized system (Figure 1b) |
| `fig_conv.pdf` | GMRES convergence histories (Figure 2) |
| `results.txt` | Environment, spectral diagnostics, iteration counts, timings, and final residuals |
| `results.json` | The scaling results in machine-readable form, one record per method and grid size |

Running the script again overwrites these files. Before rerunning, preserve the supplied reference output, for example by copying `results.txt` to `results_reference.txt`. The script does not generate the manuscript's LaTeX table; `results.json` is provided to make that step easy.

## Experiments

The code uses a fixed random seed of `2026` and considers both the unstabilized case (`D = E = 0`) and the stabilized case (`D, E != 0`).

1. **Spectral diagnostics:** eigenvalues at `k = 16`, the Bendixson strip, inertia of the Hermitian part, eigenvalue multiplicities of the ideally preconditioned matrix, the annihilating polynomial `q(M)`, and the errors `eps_1`, `eps_2` of the practical Schur-complement approximations.
2. **Scaling:** `k = 16, 32, 64, 128`, corresponding to system sizes `N = 799, 3263, 13183, 52991`.
3. **Convergence histories:** practical methods at `k = 64`, with ideal and unpreconditioned reference histories at `k = 32`.

The scaling study compares the ideal block-diagonal preconditioner, the practical block-diagonal and block-triangular preconditioners, a SIMPLE-type block-diagonal preconditioner, global ILU, and sparse direct LU. Unpreconditioned GMRES is also run for the two smaller sizes. The ideal preconditioner is computed only for `k <= 32` because it forms dense Schur complements.

GMRES is right-preconditioned and unrestarted, with a tolerance of `1e-8` and a maximum of 300 iterations. The Arnoldi basis is computed by classical Gram–Schmidt with one full reorthogonalization (CGS2), and the least-squares problem is updated by complex Givens rotations, so that the cost per iteration grows only linearly with the iteration number. One iteration is one Arnoldi step, that is, one matrix–vector product with `K`. The residual history is the GMRES least-squares residual, which with right preconditioning equals the residual of the original system; the final true relative residual is recomputed independently as `norm(b - K @ x) / norm(b)`.

## Reading the results

Scaling rows have the following columns:

```text
k  N | method : iters | setup cpu/wall | solve cpu/wall | total cpu/wall | true_res
```

- `iters`: GMRES iteration count. `>300` means the stopping tolerance was not reached within 300 iterations; `stallN` would indicate a breakdown of the Arnoldi process after `N` steps without convergence (this does not occur in the reference run).
- `setup cpu/wall`: CPU time and wall-clock time to construct and factorize the preconditioner.
- `solve cpu/wall`: CPU time and wall-clock time for the iterative solve.
- `total cpu/wall`: setup plus solve. For sparse direct LU, the total includes factorization and solution.
- `true_res`: final explicitly recomputed relative residual.

Use the **total wall-clock** value (the last time column) when transferring timings to the manuscript. Entries are rounded to two decimal places, so displayed setup and solve values may not sum exactly to the displayed total. A displayed `0.00` is a rounded measurement, not evidence of zero computational cost.

`ILU(K) global : failed (Factor is exactly singular)` indicates failure of the incomplete factorization with the selected parameters (`drop_tol=1e-4`, `fill_factor=10`). It does not by itself establish that the original matrix is singular. These cases are recorded as factorization failures (`f`) in the table.

## Reference results

The supplied `results.txt` ends with `done` and contains results for all configured sizes and both stabilization cases. Iteration counts (identical to Table 1 of the manuscript):

| Case | k | N | Ideal | Practical diagonal | Practical triangular | SIMPLE-type | Global ILU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unstabilized | 16 | 799 | 6 | 87 | 44 | 175 | 8 |
| Unstabilized | 32 | 3263 | 6 | 84 | 46 | 281 | f |
| Unstabilized | 64 | 13183 | -- | 82 | 46 | >300 | f |
| Unstabilized | 128 | 52991 | -- | 79 | 45 | >300 | f |
| Stabilized | 16 | 799 | 21 | 88 | 44 | 93 | 17 |
| Stabilized | 32 | 3263 | 21 | 92 | 46 | 146 | f |
| Stabilized | 64 | 13183 | -- | 96 | 48 | 224 | f |
| Stabilized | 128 | 52991 | -- | 96 | 48 | 300 | f |

Here `--` means not computed, `f` denotes an ILU factorization failure, and `>300` denotes failure to reach the GMRES stopping tolerance within the iteration limit. The stabilized SIMPLE-type run at `k = 128` converges at iteration **300** (true residual `8.6e-09`); it is not a `>300` entry. Unpreconditioned GMRES does not reach the tolerance at either tested size in either case.

For the unstabilized ideal matrix at `k = 16`, the script counts the computed eigenvalues lying within `1e-6` of each theoretical value:

| Eigenvalue | Multiplicity |
| ---: | ---: |
| 1 | 225 |
| (1 + sqrt 5)/2 ≈ 1.618034 | 191 |
| (1 − sqrt 5)/2 ≈ −0.618034 | 191 |
| 2 cos(π/7) ≈ 1.801938 | 64 |
| 2 cos(3π/7) ≈ 0.445042 | 64 |
| 2 cos(5π/7) ≈ −1.246980 | 64 |

All 799 eigenvalues are accounted for. The polynomial diagnostic is `norm(q(M), 2) = 9.24e-12`, where `q(x) = (x-1)(x^2-x-1)(x^3-x^2-2x+1)`. Both cases report Hermitian-part inertia of 544 positive and 255 negative eigenvalues. In the stabilized case exactly 225 eigenvalues equal 1, and the remaining ones have real parts in `[-0.987, 1.75]` and imaginary parts of modulus at most `0.163`.

Selected wall-clock totals from the reference run (seconds) at the finest grid:

| Case | Practical triangular | Sparse direct LU |
| --- | ---: | ---: |
| Unstabilized, k = 128 | 3.75 | 2.56 |
| Stabilized, k = 128 | 4.18 | 4.53 |

Refer to `results.txt` for all timings and final true residuals. Timings vary with hardware and library versions; exact numerical agreement across machines is not expected, but the iteration counts and spectral diagnostics should agree.

## Timing

All BLAS and LAPACK operations are restricted to **a single thread**. `run_experiments.py` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `NUMEXPR_NUM_THREADS` to `1` before NumPy is imported, and `dsp.py` does the same when it is imported on its own. If `threadpoolctl` is installed, `results.txt` confirms the setting with lines such as `[env] openblas (blas): num_threads=1`. This ensures that all methods are compared under the same conditions; with multithreaded BLAS, CPU time adds up the time of all threads, including idle threads that wait actively, and is then not comparable between methods.

Every timing is recorded twice, using the `Timer` class in `dsp.py`:

- **CPU time**, measured with `time.process_time()`;
- **wall-clock time**, measured with `time.perf_counter()`.

With a single thread the two agree closely (in the reference run within about 2%). The manuscript reports wall-clock times.

The measurements cover preconditioner setup, iterative solves, and sparse direct LU factorization and solution. They exclude construction of the system blocks, spectral diagnostics, plotting, and log-file output. The iterative solve timer includes the final true-residual computation inside `gmres_right()`. The direct LU timer stops before its residual is computed. The diagonal and triangular practical methods share one factorization setup in the driver; each method's reported total includes that setup cost once.
