# MAM Method for Eigenvector Recovery

This repository contains all the necessary files to reproduce the experiments related to the MAM method for eigenvector recovery.

## Files

- `primes.txt`: A list of prime numbers less than 10,000. This file is required for constructing the $(K, \alpha)$-coherent matrix **M**.
- `Experiment_varying_rank.py`: This python script is needed for the experiments showing how the errors before and after recovery are affected by the rank of the original matrix **A**
- `Experiment_varying_sparsity.py`: This python script is needed for the experiments showing how the errors before and after recovery are affected by the sparsity of the original eigenvectors of the matrix **A**
- `Experiment_varying_rows.py`: A python script needed for the experiments showing how the errors before and after recovery are affected by the number of rows in the $(K,\alpha)$-coherent matrix **M**



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
| `K`              | Number of primes used in matrix construction (derived)       | Automatically set |
| `p_list`         | List of prime numbers for coherent matrix                    | Automatically set |
| `runs`           | Number of experimental runs per rank value                   | `10`              |

These settings control the synthetic data generation, matrix structure, and experiment repetitions.

## Usage
# How to run for instance the experiment in the case of varying rank
1. Clone the repo:
   ```bash
   git clone https://github.com/boahened/MAM_Method_for_Eigenvector_Recovery.git
   cd MAM_Method_for_Eigenvector_Recovery

2. Install dependencies
   
