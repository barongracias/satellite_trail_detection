#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job — hard-negative mining (train split only)
#!  Runs the winner U-Net over empty-mask train patches at its
#!  validation-optimal threshold and records the patches it fires
#!  on as hard negatives. Train-only: no val/test patches are read,
#!  so this cannot leak selection information. The mined manifest
#!  feeds the disabled-by-default hard_negative_manifest training
#!  path; it does not alter any locked checkpoint on its own.
#!  Usage:
#!    CHECKPOINT=results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth \
#!    THRESHOLD=0.45 \
#!    OUT=results/classical/hard_negative_train_t44_s2804.json \
#!      sbatch slurm/mine_hard_negatives.sbatch
#! ==============================================================

#SBATCH -J mine_hard_neg
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/mine_hard_neg_%j.out
#SBATCH --error=results/logs/mine_hard_neg_%j.err
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

echo "Job:    $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

CHECKPOINT="${CHECKPOINT:-results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth}"
THRESHOLD="${THRESHOLD:-0.45}"
PATCH_DIR="${PATCH_DIR:-data/patches}"
OUT="${OUT:-results/classical/hard_negative_train_t44_s2804.json}"
TOP_K="${TOP_K:-1000}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi

echo "Checkpoint: $CHECKPOINT  (threshold=$THRESHOLD)"
echo "Patch dir:  $PATCH_DIR"
echo "Out:        $OUT  (top_k=$TOP_K)"

python scripts/mine_hard_negatives.py \
    --checkpoint "$CHECKPOINT" \
    --threshold "$THRESHOLD" \
    --patch_dir "$PATCH_DIR" \
    --top_k "$TOP_K" \
    --out "$OUT"

echo "Job finished."
