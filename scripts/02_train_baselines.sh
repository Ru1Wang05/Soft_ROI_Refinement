#!/bin/bash
# Train + evaluate the CNN/Swin baselines (U-Net, Attention U-Net, UNet++,
# Swin-Unet) under the identical protocol as Adjusted TransUNet.
#
# Usage: 02_train_baselines.sh [arch] [seed ...]
#        (default arch: all of "unet attunet unetpp swinunet";
#         default seeds: 42 123 456 789 1024)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[ -f configs/paths.sh ] && source configs/paths.sh
DATA_ROOT="${DATA_ROOT:-./data/fetalheart/train_npz}"
LIST_DIR="${LIST_DIR:-./lists/fetalheart}"
SWIN_PRETRAINED="${SWIN_PRETRAINED:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

ARCHS=("${1:-}")
[ -n "${ARCHS[0]}" ] || ARCHS=(unet attunet unetpp swinunet)
[ $# -gt 0 ] && shift
SEEDS=("$@")
[ ${#SEEDS[@]} -gt 0 ] || SEEDS=(42 123 456 789 1024)

mkdir -p logs baselines_results/test baselines_results/boundary

for ARCH in "${ARCHS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN="baseline_${ARCH}_384_ep70_seed${SEED}"
    CKPT_DIR="optimization_runs/${RUN}"
    CHECKPOINT="${CKPT_DIR}/best_dice.pth"
    TEST_JSON="baselines_results/test/${RUN}.json"
    BOUNDARY_DIR="baselines_results/boundary/${RUN}"

    test ! -e "$CKPT_DIR" || { echo "refusing to overwrite $CKPT_DIR" >&2; exit 3; }
    test ! -e "$TEST_JSON" || { echo "refusing to overwrite $TEST_JSON" >&2; exit 3; }

    echo "START arch=$ARCH seed=$SEED run=$RUN"
    TRAIN_ARGS=(
      --model "$ARCH"
      --root_path "$DATA_ROOT" --list_dir "$LIST_DIR"
      --snapshot_path "$CKPT_DIR"
      --num_classes 12 --img_size 384 --batch_size 4 --max_epochs 70
      --base_lr 0.01 --seed "$SEED" --loss_mode dice_ce
      --sampling uniform --augmentation legacy --val_every 2
      --num_workers 4 --amp
    )
    # swinunet: use the official Swin-T checkpoint when provided.
    if [ "$ARCH" = "swinunet" ] && [ -n "$SWIN_PRETRAINED" ] && [ -f "$SWIN_PRETRAINED" ]; then
      TRAIN_ARGS+=(--swin_pretrained "$SWIN_PRETRAINED")
    fi

    "$PYTHON_BIN" train.py "${TRAIN_ARGS[@]}"
    test -f "$CHECKPOINT" || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 4; }

    "$PYTHON_BIN" evaluate.py \
      --root_path "$DATA_ROOT" --list_dir "$LIST_DIR" \
      --checkpoint "$CHECKPOINT" --output "$TEST_JSON" \
      --model_type "baseline_${ARCH}" --split test --img_size 384 \
      --num_classes 12

    "$PYTHON_BIN" evaluate_boundary.py \
      --checkpoint "$CHECKPOINT" --output_dir "$BOUNDARY_DIR" --run_name "$RUN" \
      --data_dir "$DATA_ROOT" --list_dir "$LIST_DIR" --split test \
      --num_classes 12 --img_size 384 --seed "$SEED" \
      --model_type "baseline_${ARCH}" --boundary_metric_radius 2

    test -f "$BOUNDARY_DIR/case_metrics.csv" || { echo "missing boundary results" >&2; exit 5; }
    echo "DONE arch=$ARCH seed=$SEED run=$RUN"
  done
done
