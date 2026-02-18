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
python src/training/train_unet.py --config configs/unet.yaml

echo "Job finished."