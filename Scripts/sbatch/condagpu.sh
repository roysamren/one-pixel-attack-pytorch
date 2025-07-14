#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1                   # Request 2 nodes
#SBATCH --ntasks=1                  # Run 5 tasks in parallel (distributed across 2 nodes)
#SBATCH --cpus-per-task=8           # Each task gets 8 CPUs
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcnx-delta-gpu
#SBATCH --job-name=deeponet
#SBATCH --time=00:20:00
#SBATCH --gpus-per-task=1           # 1 GPU per task
#SBATCH --gpus=1                    # Total 5 GPUs (distributed across nodes)
# module purge # drop modules and explicitly load the ones needed, including cuda
               # (good job metadata and reproducibility)
module load anaconda3_gpu