#!/bin/bash
# Download pretrained encoder weights used by the TransUNet experiments.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[ -f configs/paths.sh ] && source configs/paths.sh
PRETRAINED="${PRETRAINED:-./pretrained/R50+ViT-B_16.npz}"

mkdir -p "$(dirname "$PRETRAINED")"
if [ ! -f "$PRETRAINED" ]; then
  echo "Downloading ImageNet-21k R50+ViT-B/16 weights..."
  wget -c "https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz" -O "$PRETRAINED"
else
  echo "Already present: $PRETRAINED"
fi

# If the Google Storage link is dead, the TransUNet README mirrors the file at
# https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd
# (file: model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz).

# Optional: official Swin-T ImageNet checkpoint for the swinunet baseline.
SWIN_PRETRAINED="${SWIN_PRETRAINED:-}"
if [ -n "$SWIN_PRETRAINED" ] && [ ! -f "$SWIN_PRETRAINED" ]; then
  mkdir -p "$(dirname "$SWIN_PRETRAINED")"
  wget -c "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth" -O "$SWIN_PRETRAINED"
fi
echo "Done."
