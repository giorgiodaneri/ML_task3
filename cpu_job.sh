#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --job-name=ML_task3_cpu

# Activate the virtual environment
micromamba activate env

n=5

for i in $(seq 1 $n); do
    python3 /home/users/gdaneri/ML_task3/P3.py
done