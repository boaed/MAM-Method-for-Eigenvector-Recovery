# MAM Method for Eigenvector Recovery

This repository contains all the necessary files to reproduce the experiments related to the MAM method for eigenvector recovery.

## Files

- `primes.txt`: A list of prime numbers less than 10,000. This file is required for constructing the $(K, \alpha)$-coherent matrix **M**.
- `Experiment_varying_rank.py`: This python script is needed for the experiments showing how the errors before and after recovery are affected by the rank of the original matrix **A**
- `Experiment_varying_sparsity.py`: This python script is needed for the experiments showing how the errors before and after recovery are affected by the sparsity of the original eigenvectors of the matrix **A**
- `Experiment_varying_rows.py`: A python script needed for the experiments showing how the errors before and after recovery are affected by the number of rows in the $(K,\alpha)$-coherent matrix **M**

## Usage
# How to run for instance the experiment in the case of varying rank
1. Clone the repo:
   ```bash
   git clone https://github.com/boahened/MAM_Method_for_Eigenvector_Recovery.git
   cd MAM_Method_for_Eigenvector_Recovery

2. Install dependencies
   
