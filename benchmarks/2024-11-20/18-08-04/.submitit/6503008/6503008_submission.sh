#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=6
#SBATCH --error=/oscar/home/hvanasse/stable-SSL/benchmarks/2024-11-20/18-08-04/.submitit/%j/%j_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/oscar/home/hvanasse/stable-SSL/benchmarks/2024-11-20/18-08-04/.submitit/%j/%j_0_log.out
#SBATCH --partition=gpu
#SBATCH --signal=USR2@120
#SBATCH --time=1000
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /oscar/home/hvanasse/stable-SSL/benchmarks/2024-11-20/18-08-04/.submitit/%j/%j_%t_log.out --error /oscar/home/hvanasse/stable-SSL/benchmarks/2024-11-20/18-08-04/.submitit/%j/%j_%t_log.err /oscar/home/hvanasse/stable-SSL/.venv/bin/python3 -u -m submitit.core._submit /oscar/home/hvanasse/stable-SSL/benchmarks/2024-11-20/18-08-04/.submitit/%j
