#!/bin/bash
#! ==============================================================
#!  CSD3 CPU job - audit patch manifest before final training/evaluation
#!  Usage:
#!    MANIFEST=data/patches/manifest.csv \
#!      sbatch slurm/audit_manifest_cpu.sbatch
#! ==============================================================

#SBATCH -J audit_manifest
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=results/logs/audit_manifest_%j.out
#SBATCH --error=results/logs/audit_manifest_%j.err
#SBATCH --qos=cpu1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

source venv/bin/activate

MANIFEST="${MANIFEST:?must be set; e.g. data/patches/manifest.csv}"
SAMPLE_ROWS="${SAMPLE_ROWS:-200}"

echo "Job:         $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Manifest:    $MANIFEST"
echo "Sample rows: $SAMPLE_ROWS"

python scripts/audit/audit_manifest.py \
    --manifest "$MANIFEST" \
    --sample-rows "$SAMPLE_ROWS" \
    --strict-stats

echo "Job finished."
