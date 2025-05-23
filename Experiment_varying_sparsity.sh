#!/bin/bash --login

#SBATCH --time=05:00:00
#SBATCH --mem=200GB
#SBATCH --ntasks=4


module purge
module load matplotlib/3.8.2-gfbf-2023b
#module load SciPy-bundle/2023.11-gfbf-2023b

cd /mnt/home/boahened/MAM_Simulate
python Experiment_varying_sparsity.py
