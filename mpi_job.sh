#!/bin/bash -l
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH --time=0-10:00:00
#SBATCH -p batch
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --job-name=ML_task3_cpu_MPI

# activate the virtual environment, change with the name of your venv
ml mpi/OpenMPI
micromamba activate ml3

iterations=100

touch result.csv
echo "run, num_of_images, num_of_processes, proc_time, time" > result.csv


for i in 16 32 64 128 256; do
    for img in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 59000; do
        echo "Running with $i processes"
        # run it 30 times
        for j in {1..30}; do
            time=$(mpirun -n $i python3 src/P3_mpi.py $img $iterations TIMING)
            echo "$j, $img, $i, $time" >> result.csv
        done
    done
done



