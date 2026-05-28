#!/bin/bash
#! ==============================================================
#!  CSD3 CPU job - compute per-image stats for a patch manifest
#!  Usage:
#!    MANIFEST=data/patches/manifest.csv \
#!      sbatch slurm/compute_image_stats_cpu.sbatch
#! ==============================================================

#SBATCH -J image_stats
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:59:00
#SBATCH --output=results/logs/compute_image_stats_%j.out
#SBATCH --error=results/logs/compute_image_stats_%j.err
#SBATCH --qos=cpu1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

source venv/bin/activate

MANIFEST="${MANIFEST:?must be set; e.g. data/patches/manifest.csv}"

echo "Job:      $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Manifest: $MANIFEST"

python scripts/compute_image_stats.py --manifest "$MANIFEST"

echo "Job finished."
