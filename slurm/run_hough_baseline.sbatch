#!/bin/bash
#! ==============================================================
#!  CSD3 CPU job — Hough baseline evaluation (< 59 min, intr)
#!  The Hough baseline is CPU-only and runs in ~5 min on 178 images.
#!  Usage: sbatch slurm/run_hough_baseline.sbatch
#! ==============================================================

#SBATCH -J hough_baseline
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/hough_%j.out
#SBATCH --error=results/logs/hough_%j.err
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

CONFIG="${CONFIG:-configs/experiments/hough_baseline.yaml}"
echo "Config: $CONFIG"

python -m src.classical.run_hough --config "$CONFIG"

echo "Job finished. Results in results/classical/"
