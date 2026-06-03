#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job - qualitative cold-DECam locked-winner demo
#!  Usage:
#!    sbatch slurm/decam_cold_inference.sbatch
#! ==============================================================

#SBATCH -J decam_cold
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=results/logs/decam_cold_%j.out
#SBATCH --error=results/logs/decam_cold_%j.err
#SBATCH --qos=intr

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs results/classical results/figures data/external/decam/raw

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

python scripts/decam_cold_inference.py

echo "Job finished."
