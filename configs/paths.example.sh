# Copy this file to configs/paths.sh and adjust the paths for your machine.
# All scripts under scripts/ source configs/paths.sh when present; otherwise
# the defaults below are used.

# Directory containing one .npz per case (keys: image, label).
export DATA_ROOT="${DATA_ROOT:-./data/fetalheart/train_npz}"

# Directory containing the fixed split lists: train.txt, val.txt, test.txt.
export LIST_DIR="${LIST_DIR:-./lists/fetalheart}"

# ImageNet-21k pretrained R50+ViT-B/16 weights (see scripts/00_download_pretrained.sh).
export PRETRAINED="${PRETRAINED:-./pretrained/R50+ViT-B_16.npz}"

# Optional: official Swin-T ImageNet checkpoint for the swinunet baseline.
export SWIN_PRETRAINED="${SWIN_PRETRAINED:-}"

# Python interpreter to use (defaults to `python` from the environment).
export PYTHON_BIN="${PYTHON_BIN:-python}"
