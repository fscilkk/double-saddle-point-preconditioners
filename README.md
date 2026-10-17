# Numerical Experiments for Double Saddle-Point Systems

Python code accompanying **Block-Diagonal Schur-Complement Preconditioners for Non-Hermitian Double Saddle-Point Systems**.

The scripts reproduce the eigenvalue plots, GMRES convergence histories, and scaling experiments in Section 7. Timing results depend on the computer, software environment, and thread configuration.

## Files

| File | Purpose |
| --- | --- |
| `dsp.py` | Matrix construction, block preconditioners, and right-preconditioned GMRES |
| `run_experiments.py` | Runs the numerical experiments and saves figures and results |
| `results.txt` | Reference output from the author's CPU-time run |

Keep all three files in the same directory. Use the names `dsp.py`, `run_experiments.py`, and `results.txt` when uploading to the repository; remove download suffixes such as `(1)` or `(2)`. The driver imports `dsp` by this module name.

## Requirements and installation

Required packages: **NumPy**, **SciPy**, and **Matplotlib**.

The author's reported experiments used Python 3.8 and SciPy 1.10.1. Record the exact Python, NumPy, SciPy, and Matplotlib versions when reproducing the experiments; this information does not specify a fully pinned environment.

Install the dependencies in your selected Python environment:

```sh
python -m pip install numpy scipy matplotlib
```

To use the SciPy version reported in the manuscript, install it in a compatible Python environment:

```sh
python -m pip install "scipy==1.10.1"
```

In Visual Studio Code, use **Python: Select Interpreter** to select the same environment in which these packages are installed.

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

The script runs all experiments without additional command-line arguments. The eigenvalue calculations may take some time before the first output appears. The full run can take several minutes or longer, depending on the machine.

The final message is `done`. After that, the terminal prompt returns. Figures are saved directly using Matplotlib's `Agg` backend; no plot windows need to be closed.

## Generated output

Files are written to the **current working directory**:

| File | Contents |
| --- | --- |
| `fig_eigs_unstab.pdf` | Eigenvalue plots for the unstabilized system |
| `fig_eigs_stab.pdf` | Eigenvalue plots for the stabilized system |
| `fig_conv.pdf` | GMRES convergence histories |
| `results.txt` | Numerical diagnostics, iteration counts, timings, and final residuals |

Running the script again overwrites these files. Before rerunning, preserve the supplied reference output, for example by copying `results.txt` to `results_reference.txt`. The script does not automatically generate the manuscript's LaTeX table.

## Experiments

The code uses a fixed random seed of `2026` and considers both the unstabilized case (`D = E = 0`) and the stabilized case (`D, E != 0`).

1. **Spectral diagnostics:** eigenvalues at `k = 16`, the Bendixson strip, inertia of the Hermitian part, and diagnostics for the ideal and practical preconditioners.
2. **Scaling:** `k = 16, 32, 64, 128`, corresponding to system sizes `N = 799, 3263, 13183, 52991`.
3. **Convergence histories:** practical methods at `k = 64`, with ideal and unpreconditioned reference histories at `k = 32`.

The scaling study compares the ideal block-diagonal preconditioner, practical block-diagonal and block-triangular preconditioners, a SIMPLE-type block-diagonal preconditioner, global ILU, and sparse direct LU. Unpreconditioned GMRES is also run for the two smaller sizes. The ideal preconditioner is computed only for `k <= 32` because it forms dense Schur complements.

GMRES is right-preconditioned and unrestarted, with a tolerance of `1e-8` and a maximum of 300 iterations in the scaling study. The residual histories are based on the Arnoldi least-squares residuals; the final true relative residual is recomputed independently as `norm(b - K @ x) / norm(b)`.

## Reading the results

Scaling rows have the following columns:

```text
k  N  | method : iters  setup(s)  solve(s)  total(s)  true_res
```

- `iters`: GMRES iteration count; `>300` means the stopping tolerance was not reached within 300 iterations.
- `setup(s)`: time to construct and factorize the preconditioner.
- `solve(s)`: time for the iterative solve.
- `total(s)`: setup plus solve time. For sparse direct LU, the reported total includes factorization and solution.
- `true_res`: final explicitly recomputed relative residual.

Use the printed **total** column when transferring timings to the manuscript. Entries are rounded to two decimal places, so displayed setup and solve values may not sum exactly to the displayed total. A displayed `0.00` is a rounded measurement, not evidence of zero computational cost.

`ILU(K) failed: Factor is exactly singular` indicates failure of the incomplete factorization with the selected parameters (`drop_tol=1e-4`, `fill_factor=10`). It does not by itself establish that the original matrix is singular. Record these cases as factorization failures (`f`) in the table.

## Reference results

The supplied `results.txt` ends with `done` and contains results for all configured sizes and both stabilization cases. The following iteration counts summarize that run:

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

Here `--` means not computed, `f` denotes an ILU factorization failure, and `>300` denotes failure to reach the GMRES stopping tolerance within the iteration limit. The stabilized SIMPLE-type run at `k = 128` converges at iteration **300**; it is not a `>300` entry. Unpreconditioned GMRES does not reach the tolerance at either tested size in either case.

For the unstabilized ideal matrix at `k = 16`, the six reported eigenvalues and multiplicities are:

| Eigenvalue (rounded) | Multiplicity |
| ---: | ---: |
| -1.246980 | 64 |
| -0.618034 | 191 |
| 0.445042 | 64 |
| 1.000000 | 225 |
| 1.618034 | 191 |
| 1.801938 | 64 |

The reported polynomial diagnostic is `norm(q(M), 2) = 8.85e-12`, where `q(x) = (x-1)(x^2-x-1)(x^3-x^2-2x+1)`. Both cases report Hermitian-part inertia of 544 positive and 255 negative eigenvalues. These are numerical diagnostics for this example.

Refer to `results.txt` for the reference CPU times and final true residuals. Timings vary with hardware, library versions, and thread settings; exact numerical agreement across machines is not required.

## CPU timing

All reported timings are **CPU times in seconds**, measured using `time.process_time()` consistently in both `run_experiments.py` and `dsp.py`. This includes preconditioner setup, iterative solves, and sparse direct LU factorization and solution.

CPU time measures the CPU resources consumed by the Python process, including its computational threads. With multithreaded numerical libraries, CPU time can exceed elapsed wall-clock time. The `total(s)` column is setup plus solve CPU time; use this column for the manuscript's timing table.

These measurements exclude construction of the system blocks, spectral diagnostics, plotting, and log-file output. The iterative solve timer includes the final true-residual computation inside `gmres_right()`. The direct LU timer stops before its residual is computed. The diagonal and triangular practical methods share one factorization setup in the driver; each method's reported total includes that setup cost once.
