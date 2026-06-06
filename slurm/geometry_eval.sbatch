#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job — geometry-aware test evaluation
#!  (exact metrics + centerline Dice + FP distance-to-mask histogram).
#!  Validation-only selection must already be locked; this re-scores the
#!  locked checkpoint's test predictions at its validation-optimal threshold.
#!  Usage:
#!    CHECKPOINT=results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth \
#!    THRESHOLD=0.45 \
#!    TAG=t44_s2804 \
#!      sbatch slurm/geometry_eval.sbatch
#! ==============================================================

#SBATCH -J geometry_eval
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/geometry_eval_%j.out
#SBATCH --error=results/logs/geometry_eval_%j.err
#SBATCH --qos=gpu1

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

echo "Job: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

CHECKPOINT="${CHECKPOINT:?must be set; e.g. results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth}"
THRESHOLD="${THRESHOLD:?must be set; the validation-optimal threshold for the checkpoint}"
PATCH_DIR="${PATCH_DIR:-data/patches}"
TAG="${TAG:-}"
THRESHOLD_SOURCE="${THRESHOLD_SOURCE:-}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi

echo "Checkpoint: $CHECKPOINT  (threshold=$THRESHOLD, tag=${TAG:-<from stem>})"
echo "Threshold source: ${THRESHOLD_SOURCE:-<none>}"

EXTRA_ARGS=()
if [ -n "$TAG" ]; then
    EXTRA_ARGS+=(--tag "$TAG")
fi
if [ -n "$THRESHOLD_SOURCE" ]; then
    EXTRA_ARGS+=(--threshold-source "$THRESHOLD_SOURCE")
fi

python scripts/evaluation/geometry_eval.py \
    --checkpoint "$CHECKPOINT" \
    --patch_dir "$PATCH_DIR" \
    --threshold "$THRESHOLD" \
    "${EXTRA_ARGS[@]}"

echo "Job finished."
