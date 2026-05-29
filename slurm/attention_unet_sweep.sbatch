#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job — long queue (< 12 hr)
#!  Usage:
#!    sbatch slurm/attention_unet_sweep.sbatch
#!    N_TRIALS=50 sbatch slurm/attention_unet_sweep.sbatch
#!    SKIP_RETRAIN=1 sbatch slurm/attention_unet_sweep.sbatch
#! ==============================================================

#SBATCH -J attention_unet_sweep
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/attention_unet_sweep_%j.out
#SBATCH --error=results/logs/attention_unet_sweep_%j.err
#SBATCH --qos=gpu1

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

python -c "import optuna, sqlalchemy, alembic" 2>/dev/null \
    || pip install --quiet "optuna[storages]"

echo "Job:    $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CONFIG="${CONFIG:-configs/experiments/attention_unet_noise_base.yaml}"
PATCH_DIR="${PATCH_DIR:-data/patches}"
N_TRIALS="${N_TRIALS:-45}"
STUDY_NAME="${STUDY_NAME:-attention_unet_f1}"
SKIP_RETRAIN="${SKIP_RETRAIN:-0}"

echo "Config:     $CONFIG"
echo "Patch dir:  $PATCH_DIR"
echo "N trials:   $N_TRIALS"
echo "Skip retrain: $SKIP_RETRAIN"

SKIP_RETRAIN_ARG=()
if [[ "$SKIP_RETRAIN" == "1" || "$SKIP_RETRAIN" == "true" || "$SKIP_RETRAIN" == "TRUE" ]]; then
    SKIP_RETRAIN_ARG=(--skip-retrain)
fi

python -m src.training.sweep \
    --config "$CONFIG" \
    --patch_dir "$PATCH_DIR" \
    --n_trials "$N_TRIALS" \
    --study_name "$STUDY_NAME" \
    "${SKIP_RETRAIN_ARG[@]}"

echo "Job finished."
