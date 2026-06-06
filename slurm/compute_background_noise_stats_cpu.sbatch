#!/bin/bash
#! ==============================================================
#!  CSD3 CPU job - calibrate signal-dependent noise from empty train patches
#!  Must run after build_patches_cpu.sbatch and compute_image_stats_cpu.sbatch.
#!  Usage:
#!    MANIFEST=data/patches/manifest.csv \
#!      sbatch slurm/compute_background_noise_stats_cpu.sbatch
#! ==============================================================

#SBATCH -J noise_stats
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:59:00
#SBATCH --output=results/logs/compute_background_noise_stats_%j.out
#SBATCH --error=results/logs/compute_background_noise_stats_%j.err
#SBATCH --qos=cpu1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs results/classical

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

source venv/bin/activate

MANIFEST="${MANIFEST:?must be set; e.g. data/patches/manifest.csv}"
OUT="${OUT:-results/classical/background_noise_calibration.json}"

echo "Job:      $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Manifest: $MANIFEST"
echo "Out:      $OUT"

python scripts/data/compute_background_noise_stats.py --manifest "$MANIFEST" --out "$OUT"

echo "Job finished."
