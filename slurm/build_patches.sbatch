#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere job - patch dataset build
#!  Usage:
#!    OUT_DIR=data/patches_test_full SPLITS=test NO_NEG_SAMPLING=1 \
#!      sbatch slurm/build_patches.sbatch
#! ==============================================================

#SBATCH -J build_patches
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/build_patches_%j.out
#SBATCH --error=results/logs/build_patches_%j.err
#SBATCH --qos=gpu1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load gcc/9 cuda/12.1 cudnn/8.9_cuda-12.1
module use /usr/local/software/spack/spack-modules/a100-20221118/linux-rocky8-zen3
module load python/3.11.9/gcc/57ayfwme

source venv/bin/activate

DATA_ROOT="${DATA_ROOT:-data/Processed}"
OUT_DIR="${OUT_DIR:-data/patches}"
PATCH_SIZE="${PATCH_SIZE:-528}"
STRIDE="${STRIDE:-528}"
SEED="${SEED:-2804}"
SPLITS="${SPLITS:-train,val,test}"
NO_NEG_SAMPLING="${NO_NEG_SAMPLING:-0}"

if [ -e "$OUT_DIR" ] && [ "${ALLOW_EXISTING:-0}" != "1" ]; then
    echo "Refusing to overwrite existing OUT_DIR=$OUT_DIR. Set ALLOW_EXISTING=1 to override." >&2
    exit 2
fi

CMD=(
    python scripts/build_patch_dataset.py
    --data-root "$DATA_ROOT"
    --out-dir "$OUT_DIR"
    --patch-size "$PATCH_SIZE"
    --stride "$STRIDE"
    --seed "$SEED"
    --splits "$SPLITS"
)

if [ "$NO_NEG_SAMPLING" = "1" ] || [ "$NO_NEG_SAMPLING" = "true" ] || [ "$NO_NEG_SAMPLING" = "TRUE" ]; then
    CMD+=(--no-neg-sampling)
fi

echo "Job:       $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Data root: $DATA_ROOT"
echo "Out dir:   $OUT_DIR"
echo "Splits:    $SPLITS"
echo "No neg:    $NO_NEG_SAMPLING"
echo "Command:   ${CMD[*]}"

"${CMD[@]}"

echo "Job finished."
