#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --time=0-04:00:00
#SBATCH -p batch
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --job-name=ML_task3_cpu

# activate the virtual environment, change with the name of your venv
micromamba activate env
# number of simulations
n=5
# numer of iterations
num_iterations=10
# create the file 'gpu_time_{num_iterations}.txt' if it does not exist, eliminate it and create a new one otherwise
if [ -f results/cpu_time_${num_iterations}.txt ]; then
    rm results/cpu_time_${num_iterations}.txt
fi
touch results/cpu_time_${num_iterations}.txt
# create the file kernel_{num_iterations}.txt if it does not exist, eliminate it and create a new one otherwise
if [ -f results/cpu_kernel_${num_iterations}.txt ]; then
    rm results/cpu_kernel_${num_iterations}.txt
fi
touch results/cpu_kernel_${num_iterations}.txt
for i in $(seq 1 $n); do
    python3 P3.py $num_iterations
done