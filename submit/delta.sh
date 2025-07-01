#!/bin/bash
#SBATCH --account bcuf-delta-gpu
#SBATCH -p gpuA40x4
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --time "00:30:00"
#SBATCH --constraint="scratch"
#SBATCH --cpus-per-task 16
#SBATCH --mem-per-cpu 4G
#SBATCH --array=0-20

cd /u/abhutani/story-surprisal/
source .venv/bin/activate
python src/run.py