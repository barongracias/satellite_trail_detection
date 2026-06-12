#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job — export blinded M9.4 audit crops.
#!  Inference is used ONLY to classify mask-empty patches as
#!  fired/quiet for the FP and decoy strata; the exported crops
#!  contain raw image pixels only.
#!  Usage:
#!    sbatch slurm/export_audit_crops.sbatch
#! ==============================================================

#SBATCH -J export_audit_crops
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/export_audit_crops_%j.out
#SBATCH --error=results/logs/export_audit_crops_%j.err
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

echo "Job: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

CHECKPOINT="${CHECKPOINT:-results/checkpoints/model-best.pth}"
SEED="${SEED:-2804}"

echo "Checkpoint: $CHECKPOINT  Seed: $SEED"

python scripts/evaluation/export_audit_crops.py \
    --checkpoint "$CHECKPOINT" \
    --seed "$SEED"

echo "Job finished."
