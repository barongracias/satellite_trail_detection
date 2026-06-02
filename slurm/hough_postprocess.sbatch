#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job — interactive queue (< 59 min)
#!  Usage:
#!    CHECKPOINT=results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth \
#!    THRESHOLD=0.45 \
#!      sbatch slurm/hough_postprocess.sbatch
#! ==============================================================

#SBATCH -J hough_postprocess
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=results/logs/hough_postprocess_%j.out
#SBATCH --error=results/logs/hough_postprocess_%j.err
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

echo "Job:    $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CHECKPOINT="${CHECKPOINT:-results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth}"
PATCH_DIR="${PATCH_DIR:-data/patches}"
THRESHOLD="${THRESHOLD:?must be set; extract from results/classical/threshold_sweep_<tag>.json optimal_threshold (default 0.5 silently corrupted Hough across all ablations)}"
OUT="${OUT:-}"

echo "Checkpoint: $CHECKPOINT"
echo "Patch dir:  $PATCH_DIR"
echo "Threshold:  $THRESHOLD"
echo "Out:        ${OUT:-<auto-derived from checkpoint stem>}"

OUT_ARG=()
if [ -n "$OUT" ]; then
    OUT_ARG=(--out "$OUT")
fi

python scripts/hough_postprocess.py \
    --checkpoint "$CHECKPOINT" \
    --patch_dir "$PATCH_DIR" \
    --threshold "$THRESHOLD" \
    "${OUT_ARG[@]}"

echo "Job finished."
