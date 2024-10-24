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

cd /home/flavio/repositories/layered_walk

source 2023_snel_modules.sh
source .venv/bin/activate 

python create_walks.py \
    --location snellius \
    --year 2010 \
    --n_walks 5 \
    --walk_len 10 \
    --dest layered_walks \
    --dry-run
