#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere job - calibrate signal-dependent noise from empty patches
#!  Usage:
#!    MANIFEST=data/patches/manifest.csv \
#!      sbatch slurm/compute_background_noise_stats.sbatch
#! ==============================================================

#SBATCH -J noise_stats
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:59:00
#SBATCH --output=results/logs/compute_background_noise_stats_%j.out
#SBATCH --error=results/logs/compute_background_noise_stats_%j.err
#SBATCH --qos=intr

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs results/classical

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load gcc/9 cuda/12.1 cudnn/8.9_cuda-12.1
module use /usr/local/software/spack/spack-modules/a100-20221118/linux-rocky8-zen3
module load python/3.11.9/gcc/57ayfwme

source venv/bin/activate

MANIFEST="${MANIFEST:?must be set; e.g. data/patches/manifest.csv}"
OUT="${OUT:-results/classical/background_noise_calibration.json}"

echo "Job:      $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Manifest: $MANIFEST"
echo "Out:      $OUT"

python scripts/compute_background_noise_stats.py --manifest "$MANIFEST" --out "$OUT"

echo "Job finished."
