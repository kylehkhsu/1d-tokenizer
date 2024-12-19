#!/bin/bash

#SBATCH --job-name=maskgit-vqgan-tokenize
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=viscam
#SBATCH --account=viscam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH --nodelist=viscam9

#source /svl/u/kylehsu/miniforge3/bin/activate && mamba activate 1d-tokenizer && cd /svl/u/kylehsu/code/1d-tokenizer

# Print job info
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Run your commands
python tokenize_imagenet_rar.py

# Clean up if necessary
echo "Job ended on $(hostname) at $(date)"
