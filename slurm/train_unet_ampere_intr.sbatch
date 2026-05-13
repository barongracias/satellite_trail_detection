#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job submission script — interactive queue
#!  Satellite Trail Segmentation (U-Net)
#! ==============================================================

#SBATCH -J unet_satellite_train
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=results/logs/unet_%j.out
#SBATCH --error=results/logs/unet_%j.err
#SBATCH --qos=intr

set -euo pipefail

#! ======= Working directory ====================================
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs results/checkpoints

#! ======= Load Modules ========================================
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load gcc/9 cuda/12.1 cudnn/8.9_cuda-12.1
module use /usr/local/software/spack/spack-modules/a100-20221118/linux-rocky8-zen3
module load python/3.11.9/gcc/57ayfwme

#! ======= Activate Virtual Environment =========================
source venv/bin/activate

#! ======= Diagnostics ==========================================
echo "Job:      $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "Submit:   $SLURM_SUBMIT_DIR"
echo "Python:   $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
nvidia-smi

#! ======= Run Training =========================================
echo "Starting U-Net training..."
DATA_ROOT="${DATA_ROOT:-data/subset/processed}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/checkpoints}"
LOG_DIR="${LOG_DIR:-results/logs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-unet_ampere_intr}"

TRAIN_ARGS=(
  --data-root "$DATA_ROOT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --auto-pos-weight
  --checkpoint-dir "$CHECKPOINT_DIR"
  --log-dir "$LOG_DIR"
  --experiment-name "$EXPERIMENT_NAME"
)
[ -n "${MAX_STEPS:-}" ]        && TRAIN_ARGS+=(--max-steps "$MAX_STEPS")
[ -n "${EVAL_MAX_BATCHES:-}" ] && TRAIN_ARGS+=(--eval-max-batches "$EVAL_MAX_BATCHES")

python -m src.training.train_unet "${TRAIN_ARGS[@]}"

echo "Job finished."
