#!/bin/bash
# Train and evaluate one stage of the main-table progression on the fixed split.
#
# Stages (paper Table: Official raw -> Corrected Dice -> ImageNet-21K ->
#         Adjusted TransUNet -> ROI Refiner):
#   raw        official TransUNet recipe, from scratch, 224, legacy loss
#   corrected  + corrected foreground-Dice + CE loss
#   pretrain   + ImageNet-21k encoder initialisation
#   adjusted   Adjusted TransUNet: 384 input, 70 epochs (main baseline)
#   roi        Adjusted TransUNet + Soft ROI Refiner (proposed)
#
# Usage: 01_train_progression.sh <stage> [seed ...]
#        (default seeds: 42 123 456 789 1024)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[ -f configs/paths.sh ] && source configs/paths.sh
DATA_ROOT="${DATA_ROOT:-./data/fetalheart/train_npz}"
LIST_DIR="${LIST_DIR:-./lists/fetalheart}"
PRETRAINED="${PRETRAINED:-./pretrained/R50+ViT-B_16.npz}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STAGE="${1:?usage: $0 <raw|corrected|pretrain|adjusted|roi> [seed ...]}"
shift || true
SEEDS=("$@")
[ ${#SEEDS[@]} -gt 0 ] || SEEDS=(42 123 456 789 1024)

mkdir -p logs optimization_runs seed_extension_results/test seed_extension_results/boundary \
  raw_test_results raw_boundary_results factor_test_results factor_boundary_results \
  final_test_results final_boundary_results

for SEED in "${SEEDS[@]}"; do
  case "$STAGE" in
    raw)
      RUN="official_raw_224_legacy_nopretrain_seed${SEED}"; IMG=224; MODEL_TYPE=original
      TRAIN=("$PYTHON_BIN" train_fetalheart_raw.py
        --root_path "$DATA_ROOT" --list_dir "$LIST_DIR"
        --snapshot_path "optimization_runs/${RUN}"
        --num_classes 12 --max_epochs 50 --batch_size 8 --base_lr 0.01
        --img_size 224 --seed "$SEED" --n_skip 3 --vit_name R50-ViT-B_16
        --vit_patches_size 16 --num_workers 4 --amp)
      ;;
    corrected)
      RUN="factor_corrected_nopretrain_224_seed${SEED}"; IMG=224; MODEL_TYPE=original
      TRAIN=("$PYTHON_BIN" train_fetalheart_factorial.py
        --root_path "$DATA_ROOT" --list_dir "$LIST_DIR"
        --snapshot_path "optimization_runs/${RUN}" --loss_mode dice_ce
        --num_classes 12 --max_epochs 50 --batch_size 8 --base_lr 0.01
        --img_size 224 --seed "$SEED" --n_skip 3 --vit_name R50-ViT-B_16
        --vit_patches_size 16 --num_workers 4 --amp)
      ;;
    pretrain)
      RUN="factor_legacy_pretrain_224_seed${SEED}"; IMG=224; MODEL_TYPE=original
      TRAIN=("$PYTHON_BIN" train_fetalheart_factorial.py
        --root_path "$DATA_ROOT" --list_dir "$LIST_DIR"
        --snapshot_path "optimization_runs/${RUN}" --loss_mode legacy_dice_ce
        --pretrained_path "$PRETRAINED"
        --num_classes 12 --max_epochs 50 --batch_size 8 --base_lr 0.01
        --img_size 224 --seed "$SEED" --n_skip 3 --vit_name R50-ViT-B_16
        --vit_patches_size 16 --num_workers 4 --amp)
      ;;
    adjusted)
      RUN="dicefix_384_ep70_seed${SEED}"; IMG=384; MODEL_TYPE=original
      TRAIN=("$PYTHON_BIN" train_fetalheart_opt.py
        --root_path "$DATA_ROOT" --list_dir "$LIST_DIR"
        --snapshot_path "optimization_runs/${RUN}" --pretrained_path "$PRETRAINED"
        --num_classes 12 --img_size 384 --batch_size 4 --max_epochs 70
        --base_lr 0.01 --vit_name R50-ViT-B_16 --n_skip 3 --seed "$SEED"
        --loss_mode dice_ce --sampling uniform --augmentation legacy
        --val_every 2 --num_workers 4 --amp)
      ;;
    roi)
      RUN="roi_refiner_384_ep70_seed${SEED}"; IMG=384; MODEL_TYPE=roi_refiner
      TRAIN=("$PYTHON_BIN" train_fetalheart_roi_refiner.py
        --root_path "$DATA_ROOT" --list_dir "$LIST_DIR"
        --snapshot_path "optimization_runs/${RUN}" --pretrained_path "$PRETRAINED"
        --num_classes 12 --img_size 384 --batch_size 4 --max_epochs 70
        --base_lr 0.01 --vit_name R50-ViT-B_16 --n_skip 3 --seed "$SEED"
        --roi_loss_weight 0.2 --augmentation legacy --val_every 2
        --num_workers 4 --amp)
      ;;
    *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
  esac

  CHECKPOINT="optimization_runs/${RUN}/best_dice.pth"
  test ! -e "optimization_runs/${RUN}" || { echo "refusing to overwrite optimization_runs/${RUN}" >&2; exit 3; }

  # Results layout expected by summarize_aish_5seed_statistics.py: extension
  # seeds live under seed_extension_results/, the three original seeds under
  # the per-stage result dirs with their table-level names.
  case "$SEED" in 789|1024)
      TEST_JSON="seed_extension_results/test/${RUN}.json"
      BOUNDARY_DIR="seed_extension_results/boundary/${RUN}" ;;
    *)
      case "$STAGE" in
        raw)       BASE_NAME="$RUN" ;;
        corrected) BASE_NAME="$RUN" ;;
        pretrain)  BASE_NAME="$RUN" ;;
        adjusted)  BASE_NAME="original_seed${SEED}" ;;
        roi)       BASE_NAME="roi_refiner_seed${SEED}" ;;
      esac
      case "$STAGE" in
        raw)       TEST_DIR=raw_test_results;    BOUNDARY_ROOT=raw_boundary_results ;;
        corrected) TEST_DIR=factor_test_results; BOUNDARY_ROOT=factor_boundary_results ;;
        pretrain)  TEST_DIR=factor_test_results; BOUNDARY_ROOT=factor_boundary_results ;;
        adjusted)  TEST_DIR=final_test_results;  BOUNDARY_ROOT=final_boundary_results ;;
        roi)       TEST_DIR=final_test_results;  BOUNDARY_ROOT=final_boundary_results ;;
      esac
      TEST_JSON="${TEST_DIR}/${BASE_NAME}.json"
      BOUNDARY_DIR="${BOUNDARY_ROOT}/${BASE_NAME}" ;;
  esac
  test ! -e "$TEST_JSON" || { echo "refusing to overwrite $TEST_JSON" >&2; exit 3; }

  echo "START stage=$STAGE seed=$SEED run=$RUN"
  "${TRAIN[@]}"
  test -f "$CHECKPOINT" || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 4; }

  "$PYTHON_BIN" evaluate_fetalheart_final_models.py \
    --root_path "$DATA_ROOT" --list_dir "$LIST_DIR" \
    --checkpoint "$CHECKPOINT" --output "$TEST_JSON" \
    --model_type "$MODEL_TYPE" --split test --img_size "$IMG" \
    --num_classes 12 --vit_name R50-ViT-B_16 --vit_patches_size 16 --n_skip 3

  "$PYTHON_BIN" eval_boundary_search.py \
    --checkpoint "$CHECKPOINT" --output_dir "$BOUNDARY_DIR" --run_name "$RUN" \
    --data_dir "$DATA_ROOT" --list_dir "$LIST_DIR" --split test \
    --num_classes 12 --img_size "$IMG" --vit_name R50-ViT-B_16 \
    --n_skip 3 --vit_patches_size 16 --seed "$SEED" \
    --model_type "$MODEL_TYPE" --boundary_metric_radius 2

  test -f "$BOUNDARY_DIR/case_metrics.csv" || { echo "missing boundary results" >&2; exit 5; }
  echo "DONE stage=$STAGE seed=$SEED run=$RUN"
done
