#!/bin/bash
#! CSD3 Ampere job: score the completed blinded self-reannotation once.

#SBATCH -J gold_audit_eval
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/gold_audit_eval_%j.out
#SBATCH --error=results/logs/gold_audit_eval_%j.err
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

ANNOTATION_DIR="${ANNOTATION_DIR:?set ANNOTATION_DIR}"
VERDICTS_CSV="${VERDICTS_CSV:?set VERDICTS_CSV}"
CROP_DIR="${CROP_DIR:-data/gold/audit_crops}"
SEALED_MANIFEST="${SEALED_MANIFEST:-data/gold/sealed_crop_manifest.json}"
CHECKPOINT="${CHECKPOINT:-results/checkpoints/model-best.pth}"
PATCH_DIR="${PATCH_DIR:-data/patches}"
OUT="${OUT:-results/classical/gold_audit_eval.json}"

test -d data/Processed
test -f "$PATCH_DIR/manifest.csv"
test -f "$SEALED_MANIFEST"
test -f "$CHECKPOINT"

python scripts/evaluation/gold_audit_eval.py \
  --annotation_dir "$ANNOTATION_DIR" \
  --verdicts_csv "$VERDICTS_CSV" \
  --crop_dir "$CROP_DIR" \
  --sealed_manifest "$SEALED_MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --threshold 0.45 \
  --patch_dir "$PATCH_DIR" \
  --out "$OUT"

echo "Reannotation audit finished: $OUT"
