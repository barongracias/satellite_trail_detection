#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job — additive diagnostics for submission polish.
#!  Inference-only on already-locked checkpoints. Does NOT retrain,
#!  re-tune, reselect, or modify any committed checkpoint/JSON.
#!
#!  Exp 6 — Hough line-thickness decomposition (single locked model):
#!    re-runs the EXACT committed parity invocation
#!    (patch_dir=data/patches_test_full, threshold=0.45) at line_thickness
#!    1 and 2 (thickness 3 == committed parity result).
#!
#!  Exp 7 — 5-seed bands on locked-model diagnostics:
#!    boundary_tolerant already exists committed for all 5 seeds, so only
#!    fp_decompose + geometry_eval are run here for the 4 non-s2804 seeds,
#!    each at that seed's OWN validation-optimal threshold.
#!
#!  Usage:  sbatch slurm/diag_exp6_exp7.sbatch
#! ==============================================================

#SBATCH -J diag_exp6_exp7
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=results/logs/diag_exp6_exp7_%j.out
#SBATCH --error=results/logs/diag_exp6_exp7_%j.err
#SBATCH --qos=gpu1

set -uo pipefail

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

CKPT_DIR=results/checkpoints
OUT_DIR=results/classical

# step name; command... — runs only if target (last token after --out / derived) absent.
declare -i N_OK=0 N_FAIL=0 N_SKIP=0

run_step () {  # $1=label  $2=out_path  $3...=command
    local label="$1"; local out="$2"; shift 2
    if [ -e "$out" ]; then
        echo ">>> SKIP  [$label] target already exists: $out"
        N_SKIP+=1
        return 0
    fi
    echo ">>> RUN   [$label] -> $out"
    echo "    \$ $*"
    if "$@"; then
        echo ">>> OK    [$label]"
        N_OK+=1
    else
        echo ">>> FAIL  [$label] (exit $?)"
        N_FAIL+=1
    fi
}

# ---------------------------------------------------------------
# Exp 6 — Hough line-thickness decomposition (parity canvas)
#   Parity canvas == data/patches_test_full (natural class dist,
#   built with --no-neg-sampling). threshold 0.45, all other Hough
#   params at script defaults (== committed parity JSON).
#   model-best.pth is sha256-identical to
#   unet_paper_arch_noise_topk_t44_s2804_best.pth (committed parity ckpt).
# ---------------------------------------------------------------
for TH in 1 2; do
    OUT="$OUT_DIR/hough_postprocess_winner_t44_s2804_parity_thick${TH}.json"
    run_step "hough_parity_thick${TH}" "$OUT" \
        python scripts/evaluation/hough_postprocess.py \
            --checkpoint "$CKPT_DIR/model-best.pth" \
            --patch_dir data/patches_test_full \
            --threshold 0.45 \
            --line_thickness "${TH}" \
            --out "$OUT"
done

# ---------------------------------------------------------------
# Exp 7 — fp_decompose + geometry_eval for the 4 non-s2804 seeds,
#   each at its OWN validation-optimal threshold.
#   (boundary_tolerant already committed for all 5 seeds; s2804
#    fp_decompose + geometry_eval already committed.)
#   seed:threshold  s7=0.41 s42=0.50 s13=0.44 s1234=0.45
# ---------------------------------------------------------------
for PAIR in "7:0.41" "42:0.50" "13:0.44" "1234:0.45"; do
    S="${PAIR%%:*}"
    THR="${PAIR##*:}"
    CKPT="$CKPT_DIR/unet_paper_arch_noise_topk_t44_s${S}_best.pth"
    SWEEP="$OUT_DIR/threshold_sweep_winner_t44_s${S}.json"

    if [ ! -f "$CKPT" ]; then
        echo ">>> FAIL  [seed s${S}] checkpoint missing: $CKPT"
        N_FAIL+=1
        continue
    fi

    FP_OUT="$OUT_DIR/fp_decomposition_unet_paper_arch_noise_topk_t44_s${S}.json"
    run_step "fp_decompose_s${S}" "$FP_OUT" \
        python scripts/evaluation/fp_decompose.py \
            --checkpoint "$CKPT" \
            --patch_dir data/patches \
            --threshold "$THR" \
            --out "$FP_OUT"

    GEO_OUT="$OUT_DIR/geometry_eval_t44_s${S}.json"
    run_step "geometry_eval_s${S}" "$GEO_OUT" \
        python scripts/evaluation/geometry_eval.py \
            --checkpoint "$CKPT" \
            --patch_dir data/patches \
            --threshold "$THR" \
            --threshold-source "$SWEEP" \
            --tag "t44_s${S}" \
            --out "$GEO_OUT"
done

echo "=============================================================="
echo "SUMMARY  ok=$N_OK  fail=$N_FAIL  skip=$N_SKIP"
echo "=============================================================="
echo "Job finished."
