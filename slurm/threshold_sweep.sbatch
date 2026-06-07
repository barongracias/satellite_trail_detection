#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job — interactive queue (< 59 min)
#!  Usage:
#!    sbatch slurm/threshold_sweep.sbatch
#!    CHECKPOINT=results/checkpoints/model-best.pth \
#!      sbatch slurm/threshold_sweep.sbatch
#! ==============================================================

#SBATCH -J threshold_sweep
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=results/logs/threshold_sweep_%j.out
#SBATCH --error=results/logs/threshold_sweep_%j.err
#SBATCH --qos=intr

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs results/figures results/classical

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load gcc/9 cuda/12.1 cudnn/8.9_cuda-12.1
module use /usr/local/software/spack/spack-modules/a100-20221118/linux-rocky8-zen3
module load python/3.11.9/gcc/57ayfwme

source venv/bin/activate

echo "Job:    $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CHECKPOINT="${CHECKPOINT:-results/checkpoints/model-best.pth}"
PATCH_DIR="${PATCH_DIR:-data/patches}"
TAG="${TAG:-}"
THRESHOLD="${THRESHOLD:-}"
VAL_ONLY="${VAL_ONLY:-0}"

echo "Checkpoint: $CHECKPOINT"
echo "Patch dir:  $PATCH_DIR"
echo "Tag:        ${TAG:-<auto-derived from checkpoint stem>}"
echo "Threshold:  ${THRESHOLD:-<val PR sweep>}"
echo "Val only:   $VAL_ONLY"

TAG_ARG=()
if [ -n "$TAG" ]; then
    TAG_ARG=(--tag "$TAG")
fi
THRESHOLD_ARG=()
if [ -n "$THRESHOLD" ]; then
    THRESHOLD_ARG=(--threshold "$THRESHOLD")
fi
VAL_ONLY_ARG=()
if [ "$VAL_ONLY" = "1" ] || [ "$VAL_ONLY" = "true" ] || [ "$VAL_ONLY" = "TRUE" ]; then
    VAL_ONLY_ARG=(--val-only)
fi

python scripts/evaluation/threshold_sweep.py \
    --checkpoint "$CHECKPOINT" \
    --patch_dir "$PATCH_DIR" \
    --num_workers 0 \
    "${TAG_ARG[@]}" \
    "${THRESHOLD_ARG[@]}" \
    "${VAL_ONLY_ARG[@]}"

echo "Job finished."
