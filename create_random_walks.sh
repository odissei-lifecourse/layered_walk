#!/bin/bash
#
#SBATCH --job-name=create_random_walks
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH -p fat_rome
#SBATCH -e %x-%j.err
#SBATCH -o %x-%j.out


module purge 
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a
module load h5py/3.9.0-foss-2023a
module load numba/0.58.1-foss-2023a

source .venv/bin/activate 

python create_walks.py \
    --location snellius \
    --year 2010 \
    --n_walks 5 \
    --walk_len 10 \
    --dest layered_walks \
    --dry-run