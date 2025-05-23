# MAM Method for Eigenvector Recovery

This repository contains all the necessary files to reproduce the experiments related to the MAM method for eigenvector recovery.

## Files

- `primes.txt`: A list of prime numbers less than 10,000. This file is required for constructing the $(K, \alpha)$-coherent matrix **M**.
- `Experiment_varying_rank.py`: This python script is needed for the experiments showing how the errors before and after recovery are affected by the rank of the original matrix **A**
- `Experiment_varying_sparsity.py`: This python script is needed for the experiments showing how the errors before and after recovery are affected by the sparsity of the original eigenvectors of the matrix **A**
- `Experiment_varying_rows.py`: A python script needed for the experiments showing how the errors before and after recovery are affected by the number of rows in the $(K,\alpha)$-coherent matrix **M**
- the additional .sh files were used to run the code on a HPC machine


##  Parameters

You can adjust the following parameters directly in `Exeperiment_varying_parameter.py`:

| Parameter        | Description                                                  | Example Value     |
|------------------|--------------------------------------------------------------|-------------------|
| `N`              | Size of the matrix                                            | `200`             |
| `s`              | Sparsity level of the eigenvectors                           | `3`               |
| `decay_type`     | Type of decay in the eigenvalues (e.g., exponential)         | `'expo'`          |
| `times_of_nnz`   | Controls extra nonzeros beyond the sparse signal             | `0`               |
| `ratio`          | Ratio for structured sparsity or overlap control             | `0`               |
| `support_type`   | Support configuration for sparse signals                     | `'disjoint'`      |
| `r_list`         | List of rank values to experiment with                       | `[6, 8, 10]`       |
| `K`              | Number of primes used in matrix construction (derived)       | '20' |
| `p_list`         | List of prime numbers for coherent matrix                    | `[11,13,17,19]` |
| `runs`           | Number of experimental runs                                  | `10`              |

These settings control the generation of the matrix A, its rank and the sparsity and decay of its eigenpairs. These parameters also dictate the number of trial we need to average over
## Usage
# How to run for instance the experiment with the varying rank: Simplest Case  in the case of varying rank
1. Clone the repo:
   ```bash
   git clone https://github.com/boahened/MAM_Method_for_Eigenvector_Recovery.git
   cd MAM_Method_for_Eigenvector_Recovery
2. Modify the parameters in the `def main()` in the script named`Experiment_varying_rank.py`

##  Output Description

After running the experiment, a `.csv` file is generated in the `figs_varying_rank/` directory. Each row corresponds to an eigenvector trial and contains the following columns:

| Column Name               | Description |
|---------------------------|-------------|
| `num_primes`              | Number of primes used to construct the $(K, \alpha)$-coherent matrix |
| `num_rows_m`              | Total number of rows in matrix $M$ (sum of selected primes) |
| `eigen_vector_index`      | Index of the eigenvector in the current trial |
| `true_eigenvalue`         | True eigenvalue of the synthetic matrix |
| `approx_eigenvalue`       | Approximated eigenvalue recovered by the algorithm |
| `l2_true_meas`            | $\ell_2$ norm of the true measurement vector |
| `l2_approx_meas`          | $\ell_2$ norm of the recovered measurement vector |
| `rel_l2_error`            | Relative $\ell_2$ error (before inversion) |
| `linf_error`              | $\ell_\infty$ error (before inversion) |
| `l2_true_eig`             | $\ell_2$ norm of the true eigenvector (should be close to 1) |
| `l2_approx_eig`           | $\ell_2$ norm of the recovered eigenvector |
| `rel_l2_inversion_error`  | Relative $\ell_2$ error (after inversion) |
| `linf_inversion_error`    | $\ell_\infty$ error (after inversion) |
| `index_difference`        | Count of mismatches in top-$s$ indices of true vs. recovered eigenvector |
| `beta`                    | Value of the $\beta$ parameter used in theoretical analysis |

---

##  Plots Generated

The experiment produces five key plots for analysis, saved in the `figs_varying_rank/` directory:

1. **Relative $\ell_2$ Error (before inversion)**
2. **$\ell_\infty$ Error (before inversion)**
3. **Relative $\ell_2$ Error (after inversion)**
4. **$\ell_\infty$ Error (after inversion)**
5. **$\beta$ Parameter Plot**

Plot filenames include key experimental parameters for easy identification and comparison.

