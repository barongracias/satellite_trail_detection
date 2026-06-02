#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job — Step 7 two-stage evaluation
#!  (classifier gate -> U-Net -> Hough). Test-side; the M5.6 winner
#!  must already be selected/recorded. Gate defaults to the deployed
#!  epoch-30 classifier at t=0.18; do not silently fall back to
#!  classifier_base_best.pth, which is a different operating point. UNET +
#!  UNET_THRESHOLD must be set per the chosen winner seed.
#!  Usage:
#!    UNET=results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth \
#!    UNET_THRESHOLD=0.45 \
#!    OUT=results/classical/two_stage_t44_s2804.json \
#!    BASELINE=results/classical/threshold_sweep_winner_t44_s2804.json \
#!    SWEEP=1 \
#!      sbatch slurm/two_stage.sbatch
#! ==============================================================

#SBATCH -J two_stage
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=results/logs/two_stage_%j.out
#SBATCH --error=results/logs/two_stage_%j.err
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

CLASSIFIER="${CLASSIFIER:-results/checkpoints/classifier_base_latest.pth}"
CLF_THRESHOLD="${CLF_THRESHOLD:-0.18}"
UNET="${UNET:?must be set; e.g. results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth}"
UNET_THRESHOLD="${UNET_THRESHOLD:?must be set; the validation-optimal threshold for the winner seed}"
MANIFEST="${MANIFEST:-data/patches/manifest.csv}"
SPLIT="${SPLIT:-test}"
OUT="${OUT:?must be set; e.g. results/classical/two_stage_t44_s2804.json}"
BASELINE="${BASELINE:-}"
SWEEP="${SWEEP:-0}"

if [ ! -f "$CLASSIFIER" ]; then
    echo "ERROR: classifier checkpoint not found: $CLASSIFIER" >&2
    exit 2
fi
if [ ! -f "$UNET" ]; then
    echo "ERROR: U-Net checkpoint not found: $UNET" >&2
    exit 2
fi

echo "Classifier:     $CLASSIFIER  (t=$CLF_THRESHOLD)"
echo "U-Net:          $UNET  (t=$UNET_THRESHOLD)"
echo "Manifest/split: $MANIFEST / $SPLIT"
echo "Out:            $OUT"
echo "Baseline:       ${BASELINE:-<none>}"
echo "Joint sweep:    $SWEEP"

EXTRA_ARGS=()
if [ -n "$BASELINE" ]; then
    EXTRA_ARGS+=(--baseline "$BASELINE")
fi
if [ "$SWEEP" = "1" ] || [ "$SWEEP" = "true" ] || [ "$SWEEP" = "TRUE" ]; then
    EXTRA_ARGS+=(--sweep)
fi

python -m src.inference.two_stage \
    --classifier "$CLASSIFIER" \
    --clf-threshold "$CLF_THRESHOLD" \
    --unet "$UNET" \
    --unet-threshold "$UNET_THRESHOLD" \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --out "$OUT" \
    "${EXTRA_ARGS[@]}"

echo "Job finished."
