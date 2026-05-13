#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job submission script
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
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/unet_%j.out
#SBATCH --error=results/logs/unet_%j.err
#SBATCH --qos=GPU

#! ======= Load Modules ========================================
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load gcc/9 cuda/12.1 cudnn

#! ======= Activate Virtual Environment =========================
source ~/astro_env/bin/activate

#! ======= Diagnostics ==========================================
echo "Running on: $SLURM_NODELIST"
echo "Python: $(which python)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi

#! ======= Run Training =========================================
echo "Starting U-Net training..."
DATA_ROOT="${DATA_ROOT:-data/subset/processed}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/checkpoints}"
LOG_DIR="${LOG_DIR:-results/logs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-unet_ampere_long}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-}"

if [ -n "${MAX_STEPS:-}" ] && [ -n "$EVAL_MAX_BATCHES" ]; then
  python -m src.training.train_unet \
    --data-root "$DATA_ROOT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --auto-pos-weight \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    --experiment-name "$EXPERIMENT_NAME" \
    --max-steps "$MAX_STEPS" \
    --eval-max-batches "$EVAL_MAX_BATCHES"
elif [ -n "${MAX_STEPS:-}" ]; then
  python -m src.training.train_unet \
    --data-root "$DATA_ROOT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --auto-pos-weight \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    --experiment-name "$EXPERIMENT_NAME" \
    --max-steps "$MAX_STEPS"
else
  python -m src.training.train_unet \
    --data-root "$DATA_ROOT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --auto-pos-weight \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    --experiment-name "$EXPERIMENT_NAME"
fi

echo "Job finished."
