#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere job - compute per-image stats for a patch manifest
#!  Usage:
#!    MANIFEST=data/patches_test_full/manifest.csv \
#!      sbatch slurm/compute_image_stats.sbatch
#! ==============================================================

#SBATCH -J image_stats
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:59:00
#SBATCH --output=results/logs/compute_image_stats_%j.out
#SBATCH --error=results/logs/compute_image_stats_%j.err
#SBATCH --qos=intr

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load gcc/9 cuda/12.1 cudnn/8.9_cuda-12.1
module use /usr/local/software/spack/spack-modules/a100-20221118/linux-rocky8-zen3
module load python/3.11.9/gcc/57ayfwme

source venv/bin/activate

MANIFEST="${MANIFEST:?must be set; e.g. data/patches_test_full/manifest.csv}"

echo "Job:      $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Manifest: $MANIFEST"

python scripts/compute_image_stats.py --manifest "$MANIFEST"

echo "Job finished."
