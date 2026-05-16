#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job — interactive queue (< 59 min)
#!  Usage:
#!    sbatch slurm/threshold_sweep.sbatch
#!    CHECKPOINT=results/checkpoints/unet_baseline_best.pth \
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

CONFIG="${CONFIG:-configs/experiments/unet_baseline.yaml}"
CHECKPOINT="${CHECKPOINT:-results/checkpoints/unet_baseline_best.pth}"
PATCH_DIR="${PATCH_DIR:-data/patches}"

echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Patch dir:  $PATCH_DIR"

python scripts/threshold_sweep.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --patch_dir "$PATCH_DIR"

echo "Job finished."
