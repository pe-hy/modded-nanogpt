#!/bin/bash
#SBATCH --job-name=inference_modded
#SBATCH --output=logs/inference/inference_%j.out
#SBATCH --error=logs/inference/inference_%j.err
#SBATCH --time=00:10:00
#SBATCH --account=OPEN-34-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --partition=qgpu

module load Anaconda3/2024.02-1
module load CUDA/12.4.0
source activate main_env

python inference.py logs/f5a8803e-6188-4325-8f85-c2d552670f4b/state_step002020.pt --prompt "[BOS] 5 + 3 ="