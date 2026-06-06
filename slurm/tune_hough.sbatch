#!/bin/bash
#! ==============================================================
#!  CSD3 CPU job — Hough parameter grid search (< 59 min, intr)
#!  Usage: sbatch slurm/tune_hough.sbatch
#! ==============================================================

#SBATCH -J hough_tune
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:59:00
#SBATCH --output=results/logs/hough_tune_%j.out
#SBATCH --error=results/logs/hough_tune_%j.err
#SBATCH --qos=intr

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs results/classical

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module use /usr/local/software/spack/spack-modules/a100-20221118/linux-rocky8-zen3
module load python/3.11.9/gcc/57ayfwme

source venv/bin/activate

echo "Job:    $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Python: $(which python)"

python scripts/evaluation/tune_hough.py

echo "Job finished."
