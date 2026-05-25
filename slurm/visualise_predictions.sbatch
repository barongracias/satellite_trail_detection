#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere job - prediction visualisation figures
#!  Usage:
#!    CHECKPOINT=results/checkpoints/unet_ablation_0_best.pth \
#!    THRESHOLD=0.47 \
#!    HOUGH_JSON=results/classical/hough_postprocess_unet_ablation_0.json \
#!      sbatch slurm/visualise_predictions.sbatch
#! ==============================================================

#SBATCH -J visualise_predictions
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=results/logs/visualise_predictions_%j.out
#SBATCH --error=results/logs/visualise_predictions_%j.err
#SBATCH --qos=intr

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs results/figures/predictions

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load gcc/9 cuda/12.1 cudnn/8.9_cuda-12.1
module use /usr/local/software/spack/spack-modules/a100-20221118/linux-rocky8-zen3
module load python/3.11.9/gcc/57ayfwme

source venv/bin/activate

CHECKPOINT="${CHECKPOINT:?must be set}"
THRESHOLD="${THRESHOLD:?must be set}"
HOUGH_JSON="${HOUGH_JSON:?must be set}"
PATCH_DIR="${PATCH_DIR:-data/patches}"
OUT_DIR="${OUT_DIR:-results/figures/predictions}"
TAG="${TAG:-}"

TAG_ARG=()
if [ -n "$TAG" ]; then
    TAG_ARG=(--tag "$TAG")
fi

echo "Job:       $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Checkpoint: $CHECKPOINT"
echo "Threshold:  $THRESHOLD"
echo "Hough JSON: $HOUGH_JSON"
echo "Patch dir:  $PATCH_DIR"
echo "Out dir:    $OUT_DIR"

python scripts/visualise_predictions.py \
    --checkpoint "$CHECKPOINT" \
    --patch_dir "$PATCH_DIR" \
    --threshold "$THRESHOLD" \
    --hough_json "$HOUGH_JSON" \
    --out_dir "$OUT_DIR" \
    "${TAG_ARG[@]}"

echo "Job finished."
