#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job — ensemble (+ optional TTA) evaluation
#!  Set TTA=1 to enable 4-view test-time augmentation.
#!  Usage:
#!    CHECKPOINTS="ckpt1,ckpt2,ckpt3,ckpt4,ckpt5" \
#!    OUT=results/classical/ensemble_t44.json \
#!    TTA=1 \
#!      sbatch slurm/ensemble_eval.sbatch
#! ==============================================================

#SBATCH -J ensemble_eval
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=results/logs/ensemble_eval_%j.out
#SBATCH --error=results/logs/ensemble_eval_%j.err
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

CHECKPOINTS="${CHECKPOINTS:?must be set; comma-separated checkpoint paths}"
OUT="${OUT:?must be set; e.g. results/classical/ensemble_t44.json}"
PATCH_DIR="${PATCH_DIR:-data/patches}"
TTA="${TTA:-0}"
THRESHOLD="${THRESHOLD:-}"

echo "Checkpoints: $CHECKPOINTS"
echo "Out:         $OUT"
echo "TTA:         $TTA"
echo "Threshold:   ${THRESHOLD:-<val PR sweep>}"

EXTRA_ARGS=()
if [ "$TTA" = "1" ] || [ "$TTA" = "true" ] || [ "$TTA" = "TRUE" ]; then
    EXTRA_ARGS+=(--tta)
fi
if [ -n "$THRESHOLD" ]; then
    EXTRA_ARGS+=(--threshold "$THRESHOLD")
fi

python scripts/evaluation/ensemble_eval.py \
    --checkpoints "$CHECKPOINTS" \
    --patch_dir "$PATCH_DIR" \
    --out "$OUT" \
    "${EXTRA_ARGS[@]}"

echo "Job finished."
