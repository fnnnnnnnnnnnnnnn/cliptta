#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/media/fnn/cliptta/ttavlm/data}"
SAVE_ROOT="${SAVE_ROOT:-/media/fnn/cliptta/ttavlm/result/visda_vit_osda}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
KNOWN_RATIO="${KNOWN_RATIO:-0.5}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-10}"

SOURCE_CKPT="${SAVE_ROOT}/source/train/vit_b16_known${KNOWN_RATIO}_seed${SEED}.pt"
mkdir -p "${SAVE_ROOT}"

if [[ ! -f "${SOURCE_CKPT}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" python -m ttavlm.vit_osda source-train \
    --dataset visda \
    --source_domain train \
    --dataroot "${DATA_ROOT}" \
    --save_root "${SAVE_ROOT}" \
    --known_class_ratio "${KNOWN_RATIO}" \
    --epochs "${SOURCE_EPOCHS}" \
    --batch_size 64 \
    --workers 4 \
    --lr 1e-4 \
    --seed "${SEED}" \
    --output "${SOURCE_CKPT}"
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -m ttavlm.vit_osda adapt-eval \
  --dataset visda \
  --source_domain train \
  --target_domain validation \
  --dataroot "${DATA_ROOT}" \
  --save_root "${SAVE_ROOT}" \
  --known_class_ratio "${KNOWN_RATIO}" \
  --batch_size 64 \
  --workers 4 \
  --seed "${SEED}" \
  --source_checkpoint "${SOURCE_CKPT}"
