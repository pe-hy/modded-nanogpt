#!/bin/bash
#SBATCH --job-name=train_modded
#SBATCH --output=logs/train/train_%j.out
#SBATCH --error=logs/train/train_%j.err
#SBATCH --time=00:30:00
#SBATCH --account=OPEN-34-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --partition=qgpu

module load Anaconda3/2024.02-1
module load CUDA/12.4.0
source activate main_env

torchrun --standalone --nproc_per_node=8 train_gpt.py
