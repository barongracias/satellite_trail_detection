#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job — Step 14 boundary-tolerant evaluation
#!  Usage:
#!    CHECKPOINT=results/checkpoints/model-best.pth \
#!    THRESHOLD=0.45 \
#!      sbatch slurm/boundary_tolerant_eval.sbatch
#! ==============================================================

#SBATCH -J boundary_tol
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=results/logs/boundary_tolerant_%j.out
#SBATCH --error=results/logs/boundary_tolerant_%j.err
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
TOLERANCES="${TOLERANCES:-0,1,2,3}"
OUT="${OUT:-}"

echo "Checkpoint: $CHECKPOINT"
echo "Threshold:  $THRESHOLD"
echo "Tolerances: $TOLERANCES"

OUT_ARG=()
if [ -n "$OUT" ]; then
    OUT_ARG=(--out "$OUT")
fi

python scripts/evaluation/boundary_tolerant_eval.py \
    --checkpoint "$CHECKPOINT" \
    --patch_dir "$PATCH_DIR" \
    --threshold "$THRESHOLD" \
    --tolerances "$TOLERANCES" \
    "${OUT_ARG[@]}"

echo "Job finished."
