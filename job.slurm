#!/bin/bash
#SBATCH --job-name=prj-645-g16        # Job name for identification
#SBATCH --partition=gpu               # Use the GPU partition
#SBATCH --gpus-per-node=1             # Request 1 GPU per node
#SBATCH --time=06:00:00               # Maximum runtime (adjust as needed)
#SBATCH --output=stdout_%j.out        # Standard output file
#SBATCH --error=stderror_%j.err       # Standard error file
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks=1                    # Single task (no MPI)
#SBATCH --cpus-per-task=1             # Reduce CPUs to 1 if not needed
#SBATCH --mem=16GB                    # Reduce memory allocation if sufficient

source ~/software/init-conda.sh
conda activate enel645

python train.py

conda deactivate
