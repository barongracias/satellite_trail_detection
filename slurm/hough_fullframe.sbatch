#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job — dense full-frame Hough evaluation.
#!  Inference-only on the locked winner; retires the patch-canvas caveat by
#!  scoring the SAME Hough scheme on a dense full-frame prediction instead of
#!  the 1:3 sampled-patch canvas. All Hough params/thresholds are read from
#!  the committed parity JSON inside the script (kept fixed).
#!  Usage:
#!    CHECKPOINT=results/checkpoints/model-best.pth \
#!      sbatch slurm/hough_fullframe.sbatch
#! ==============================================================

#SBATCH -J hough_fullframe
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=results/logs/hough_fullframe_%j.out
#SBATCH --error=results/logs/hough_fullframe_%j.err
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CHECKPOINT="${CHECKPOINT:-results/checkpoints/model-best.pth}"
OUT="${OUT:-results/classical/hough_fullframe_winner_t44_s2804.json}"

echo "Checkpoint: $CHECKPOINT"
echo "Out:        $OUT"

python scripts/evaluation/hough_fullframe.py \
    --checkpoint "$CHECKPOINT" \
    --out "$OUT"

echo "Job finished."
