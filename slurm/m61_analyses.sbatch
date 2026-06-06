#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job - M6.1 locked-winner A-MVP + B only (< 59 min)
#!  Usage:
#!    sbatch slurm/m61_analyses.sbatch
#! ==============================================================

#SBATCH -J m61_analyses
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=results/logs/m61_analyses_%j.out
#SBATCH --error=results/logs/m61_analyses_%j.err
#SBATCH --qos=intr

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs results/classical results/figures

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

echo "Running M6.1 A-MVP faint-streak diagnostic..."
python scripts/figures/faint_streak_analysis.py

echo "Running M6.1 B Hough gap bridge figure..."
python scripts/figures/hough_gap_figure.py

echo "Job finished."
