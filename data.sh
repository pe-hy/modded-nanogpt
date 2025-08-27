#!/bin/bash
#SBATCH --job-name=data_modded
#SBATCH --output=logs/data/data_%j.out
#SBATCH --error=logs/data/data_%j.err
#SBATCH --time=00:10:00
#SBATCH --account=OPEN-34-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --partition=qgpu

module load Anaconda3/2024.02-1
module load CUDA/12.4.0
source activate main_env

python data/cached_fineweb10B.py 8
