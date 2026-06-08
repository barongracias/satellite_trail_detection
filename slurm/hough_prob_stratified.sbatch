#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job — probability-stratified Hough (approximation)
#!  Compares pre-Hough / binary Hough / stratified Hough recall on the
#!  reconstructed test canvases. Does NOT modify the locked Hough path.
#!  Usage:
#!    CHECKPOINT=results/checkpoints/model-best.pth \
#!    THRESHOLD=0.45 \
#!    TAG=t44_s2804 \
#!      sbatch slurm/hough_prob_stratified.sbatch
#! ==============================================================

#SBATCH -J hough_strat
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/hough_strat_%j.out
#SBATCH --error=results/logs/hough_strat_%j.err
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

CHECKPOINT="${CHECKPOINT:?must be set; e.g. results/checkpoints/model-best.pth}"
THRESHOLD="${THRESHOLD:?must be set; the validation-optimal threshold for the checkpoint}"
PATCH_DIR="${PATCH_DIR:-data/patches}"
STRATA="${STRATA:-0.1,0.3,0.5,0.7,0.9}"
TAG="${TAG:-}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi

echo "Checkpoint: $CHECKPOINT  (threshold=$THRESHOLD, strata=$STRATA, tag=${TAG:-<from stem>})"

TAG_ARG=()
if [ -n "$TAG" ]; then
    TAG_ARG=(--tag "$TAG")
fi

python scripts/evaluation/hough_prob_stratified.py \
    --checkpoint "$CHECKPOINT" \
    --patch_dir "$PATCH_DIR" \
    --threshold "$THRESHOLD" \
    --strata "$STRATA" \
    "${TAG_ARG[@]}"

echo "Job finished."
