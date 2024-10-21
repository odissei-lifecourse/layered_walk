#!/bin/bash
#
#SBATCH --job-name=numba_check
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 64 
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH -p fat_rome
#SBATCH -e %x-%j.err
#SBATCH -o %x-%j.out

cd /home/fhafner/repositories/layered_walk/
source 2023_snel_modules.sh

source .venv/bin/activate 

python numba_check.py --location snellius 





