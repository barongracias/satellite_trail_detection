#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job — interactive queue (< 59 min)
#!  Usage:
#!    CHECKPOINT=results/checkpoints/unet_<tag>_best.pth \
#!    THRESHOLD=0.72 \
#!      sbatch slurm/bootstrap_metrics.sbatch
#! ==============================================================

#SBATCH -J bootstrap_metrics
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=results/logs/bootstrap_metrics_%j.out
#SBATCH --error=results/logs/bootstrap_metrics_%j.err
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

echo "Job:    $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CHECKPOINT="${CHECKPOINT:?must be set; e.g. results/checkpoints/unet_<tag>_best.pth}"
THRESHOLD="${THRESHOLD:?must be set; extract from results/classical/threshold_sweep_<tag>.json optimal_threshold}"
PATCH_DIR="${PATCH_DIR:-data/patches}"
N_RESAMPLES="${N_RESAMPLES:-1000}"
TAG="${TAG:-}"
OUT="${OUT:-}"

echo "Checkpoint:  $CHECKPOINT"
echo "Threshold:   $THRESHOLD"
echo "Patch dir:   $PATCH_DIR"
echo "N resamples: $N_RESAMPLES"
echo "Tag:         ${TAG:-<auto-derived from checkpoint stem>}"
echo "Out:         ${OUT:-<auto-derived from tag>}"

EXTRA_ARGS=()
if [ -n "$TAG" ]; then
    EXTRA_ARGS+=(--tag "$TAG")
fi
if [ -n "$OUT" ]; then
    EXTRA_ARGS+=(--out "$OUT")
fi

python scripts/evaluation/bootstrap_metrics.py \
    --checkpoint "$CHECKPOINT" \
    --threshold "$THRESHOLD" \
    --patch_dir "$PATCH_DIR" \
    --n_resamples "$N_RESAMPLES" \
    "${EXTRA_ARGS[@]}"

echo "Job finished."
