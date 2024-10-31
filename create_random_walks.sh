#!/bin/bash
#
#SBATCH --job-name=lyr_wlk
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --mem=30G
#SBATCH -p rome
#SBATCH -e %x-%j.err
#SBATCH -o %x-%j.out

cd /home/flavio/repositories/layered_walk

source 2023_snel_modules.sh
source .venv/bin/activate 

export NUMEXPR_MAX_THREADS=32
python create_walks.py \
    --location snellius \
    --year 2010 \
    --n_walks 4 \
    --walk_len 20 \
    --iteration_name first_trial \
    --no-record_edge_types

python create_walks.py \
    --location snellius \
    --year 2010 \
    --n_walks 4 \
    --walk_len 40 \
    --iteration_name first_trial \
    --record_edge_types

