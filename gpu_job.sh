#!/bin/bash
#SBATCH --job-name=ML_task3_gpu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=0-01:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal

HPCAI_ROOT="/work/projects/ulhpc-tutorials/PS10-Horovod/"

# MPI, CUDA, and compilers
module load toolchain/intelcuda

# CUDNN
export CUDNN_PATH=${HPCAI_ROOT}/soft/cudnn/install/cudnn-linux-x86_64-8.8.1.3_cuda11-archive
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib/:${LD_LIBRARY_PATH}
export CPATH=${CUDNN_PATH}/include/:${CPATH}

# NCCL
NCCL_DEBUG=INFO # allow to check if NCCL is active
# NCCL_ROOT=${HPCAI_ROOT}/miniconda/install/miniconda/lib/python3.10/site-packages/nvidia/nccl/ #nccl also present there
NCCL_ROOT=/work/projects/ulhpc-tutorials/PS10-Horovod/soft/nccl/install/nccl_2.17.1-1+cuda11.0_x86_64/
NCCL_INCLUDE_DIR=${NCCL_ROOT}/include/
NCCL_LIBRARY=${NCCL_ROOT}/lib/

LD_LIBRARY_PATH=${NCCL_ROOT}/lib/:${LD_LIBRARY_PATH}
CPATH=${NCCL_ROOT}/include/:${CPATH}

source /work/projects/ulhpc-tutorials/PS10-Horovod/env_ds.sh  # source python

source /home/users/gdaneri/.jaxenv/bin/activate

n=5

for i in $(seq 1 $n); do
    python3 /home/users/gdaneri/ML_task3/P3_gpu.py
done