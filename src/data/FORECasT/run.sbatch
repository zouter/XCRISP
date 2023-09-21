#!/bin/sh
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/cfseale/logs/slurm-%j.out
#SBATCH --error=/scratch/cfseale/logs/slurm-%j.err

srun python3 /scratch/cfseale/repos/repair-outcome-prediction/indel_analysis/data/download_from_ena.py